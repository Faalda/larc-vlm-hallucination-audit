"""C1 extensions requested in review: (A) paired, image-clustered DeltaAUC with a
GTC-vs-alignment contrast; (B) sensitivity to human-adjudicated labels; (C) synthetic
ground-truth category-drop perturbation (no known bug) with a model-free coupled probe.

Usage: PYTHONPATH=src python scripts/run_c1_extensions.py
"""
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# bbox_to_patch_indices is pure numpy; avoid requiring torch for this CPU-only analysis.
try:
    import torch  # noqa: F401
    from larc_audit.llava_geometry import bbox_to_patch_indices
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")
    from larc_audit.llava_geometry import bbox_to_patch_indices
    del sys.modules["torch"]

ROOT = Path(__file__).resolve().parents[1]
RES, DATA = ROOT / "results", ROOT / "data"
N_BOOT = 2000


def auc(y, s):
    """Mann-Whitney AUC with tie-averaged ranks; y=1 is hallucinated."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def oof(X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def positional(base, other, key):
    gb, go = defaultdict(list), defaultdict(list)
    for r in base:
        gb[r["image_id"]].append(r)
    for r in other:
        go[r["image_id"]].append(r)
    out = []
    for iid in gb:
        for rb, ro in zip(gb[iid], go[iid]):
            assert rb["word"] == ro["word"]
            out.append(ro[key])
    return np.array(out)


def ci(v):
    return np.nanpercentile(v, [2.5, 97.5])


def load():
    recs = json.load(open(RES / "pilot_records_caption_corrected.json"))
    perhead = json.load(open(RES / "alignment_perhead_full_records.json"))
    ids = [r["image_id"] for r in recs]
    n_runs = 1 + sum(ids[i] != ids[i - 1] for i in range(1, len(ids)))
    assert n_runs == len(set(ids)), "records not image-contiguous"
    gtc = np.array([r["by_layer"]["20"]["gtc_any_object"] for r in recs])
    al = positional(recs, perhead, "align_perhead")
    return recs, gtc, al


def part_a(recs, gtc, al):
    print("=" * 70, "\n(A) Paired image-clustered DeltaAUC\n" + "=" * 70)
    y_o = np.array([0 if r["grounded_original"] else 1 for r in recs])
    y_c = np.array([0 if r["grounded_corrected"] else 1 for r in recs])
    img = np.array([r["image_id"] for r in recs])
    feats = {"GTC": gtc[:, None], "alignment": al[:, None]}
    p_o = {k: oof(X, y_o) for k, X in feats.items()}
    p_c = {k: oof(X, y_c) for k, X in feats.items()}
    uniq = np.unique(img)
    idx_by = {u: np.where(img == u)[0] for u in uniq}
    rng = np.random.RandomState(0)
    deltas = {k: [] for k in feats}
    contrast = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, len(uniq))
        idx = np.concatenate([idx_by[u] for u in pick])
        d = {}
        for k in feats:
            d[k] = auc(y_c[idx], p_c[k][idx]) - auc(y_o[idx], p_o[k][idx])
            deltas[k].append(d[k])
        contrast.append(d["GTC"] - d["alignment"])
    for k in feats:
        a_o, a_c = auc(y_o, p_o[k]), auc(y_c, p_c[k])
        lo, hi = ci(deltas[k])
        print(f"{k:15s} orig={a_o:.3f} corr={a_c:.3f} dAUC={a_c - a_o:+.3f} [{lo:+.3f},{hi:+.3f}]")
    d_pt = (auc(y_c, p_c["GTC"]) - auc(y_o, p_o["GTC"])) - (auc(y_c, p_c["alignment"]) - auc(y_o, p_o["alignment"]))
    lo, hi = ci(contrast)
    print(f"contrast dAUC(GTC)-dAUC(align) = {d_pt:+.3f} [{lo:+.3f},{hi:+.3f}]")


def part_b(recs, gtc, al):
    print("=" * 70, "\n(B) Human-adjudicated label sensitivity\n" + "=" * 70)
    aud = [json.loads(l) for l in open(RES / "e4_label_audit" / "audit_labels.jsonl")]
    lut = defaultdict(list)
    for i, r in enumerate(recs):
        lut[(r["image_id"], r["word"].lower())].append(i)
    lut_cat = defaultdict(list)
    for i, r in enumerate(recs):
        lut_cat[(r["image_id"], r["category"])].append(i)
    y = np.array([0 if r["grounded_corrected"] else 1 for r in recs])
    keep = np.ones(len(recs), bool)
    n_join = n_amb = 0
    changed = 0
    for a in aud:
        cand = lut.get((a["image_id"], a["mention"].lower())) or lut_cat.get((a["image_id"], a["category"]), [])
        cand = [i for i in cand if not recs[i]["grounded_original"]]
        if not cand:
            continue
        n_join += 1
        i = cand[0]
        lab = a["final_label"]
        if lab == "AMBIGUOUS":
            keep[i] = False
            n_amb += 1
        else:
            new = 1 if lab == "NOT_VISIBLE" else 0
            changed += int(new != y[i])
            y[i] = new
    print(f"audit items joined: {n_join}/{len(aud)}; ambiguous dropped: {n_amb}; labels changed vs caption-corrected: {changed}")
    y_c = np.array([0 if r["grounded_corrected"] else 1 for r in recs])
    img = np.array([r["image_id"] for r in recs])
    feats = {"GTC": gtc[:, None], "alignment": al[:, None], "GTC+alignment": np.stack([gtc, al], 1)}
    uniq = np.unique(img[keep])
    idx_by = {u: np.where((img == u) & keep)[0] for u in uniq}
    for k, X in feats.items():
        Xk, yk, yck = X[keep], y[keep], y_c[keep]
        p = oof(Xk, yk)
        pc = oof(Xk, yck)
        rng = np.random.RandomState(0)
        bs = []
        for _ in range(N_BOOT):
            idx = np.concatenate([idx_by[u] for u in rng.choice(uniq, len(uniq))])
            sub = np.where(keep)[0]
            pos = {g: j for j, g in enumerate(sub)}
            j = np.array([pos[g] for g in idx])
            bs.append(auc(yk[j], p[j]))
        lo, hi = ci(bs)
        print(f"{k:15s} human-adj AUC={auc(yk, p):.3f} [{lo:.3f},{hi:.3f}]  caption-corr (same items)={auc(yck, pc):.3f}  "
              f"halluc: {int(yk.sum())}/{len(yk)}")


def part_c(recs, al):
    print("=" * 70, "\n(C) Synthetic GT category-drop (model-free coupled probe vs alignment)\n" + "=" * 70)
    meta = {m["image_id"]: m for m in json.load(open(DATA / "pilot_index.json")) + json.load(open(DATA / "pilot_index_extra.json"))}
    corr_cats = {int(k): set(v) for k, v in json.load(open(RES / "corrected_gt_categories.json")).items()}
    grid = np.array([(i // 24, i % 24) for i in range(576)], dtype=np.float32)
    patch_cache = {}

    def cat_patches(iid, cat):
        key = (iid, cat)
        if key not in patch_cache:
            m = meta[iid]
            s = set()
            for o in m["objects"]:
                if o["category"] == cat:
                    s.update(bbox_to_patch_indices(o["bbox"], m["width"], m["height"], 336, {"width": 336, "height": 336}, 14))
            patch_cache[key] = s
        return patch_cache[key]

    def dist_map(iid, dropped):
        ps = set()
        for c in {o["category"] for o in meta[iid]["objects"]} - dropped:
            ps |= cat_patches(iid, c)
        if not ps:
            return None
        t = grid[sorted(ps)]
        return np.min(np.linalg.norm(grid[:, None] - t[None], axis=-1), axis=1)

    cache_path = RES / "l20_attention_cache.npz"
    attn = np.load(cache_path)["attn"] if cache_path.exists() else None

    imgs = sorted({r["image_id"] for r in recs})
    img_idx = defaultdict(list)
    for i, r in enumerate(recs):
        img_idx[r["image_id"]].append(i)
    inst = {iid: sorted({o["category"] for o in meta[iid]["objects"]}) for iid in imgs}
    cc = {iid: corr_cats.get(iid, set()) | set(inst[iid]) for iid in imgs}
    cat_of = np.array([r["category"] for r in recs])
    align = np.asarray(al)
    n_seeds = 30
    base = None
    for k in [0.0, 0.05, 0.10, 0.20, 0.40]:
        a_g, a_r, a_a, frac = [], [], [], []
        for s in range(n_seeds if k > 0 else 1):
            rng = np.random.RandomState(1000 + s)
            y = np.zeros(len(recs), dtype=int)
            g = np.zeros(len(recs))
            r_gtc = np.full(len(recs), np.nan)
            for iid in imgs:
                drop = {c for c in sorted(cc[iid]) if rng.rand() < k}
                if set(inst[iid]) <= drop:  # keep one annotated category so GTC stays defined
                    drop.discard(inst[iid][rng.randint(len(inst[iid]))])
                d = dist_map(iid, drop)
                for i in img_idx[iid]:
                    y[i] = int(cat_of[i] in drop or cat_of[i] not in cc[iid])
                    g[i] = np.nan if d is None else d.mean()
                    if attn is not None and d is not None:
                        r_gtc[i] = attn[i] @ d
            ok = np.isfinite(g)  # a kept category whose box is fully cropped leaves GTC undefined
            a_g.append(auc(y[ok], g[ok]))
            a_a.append(auc(y[ok], -align[ok]))
            ok_r = np.isfinite(r_gtc)
            a_r.append(auc(y[ok_r], r_gtc[ok_r]) if ok_r.any() else np.nan)
            frac.append(y.mean())
        def fmt(v):
            return f"{np.nanmean(v):.3f} [{np.nanpercentile(v, 2.5):.3f},{np.nanpercentile(v, 97.5):.3f}]"
        print(f"k={k:4.2f} halluc={np.mean(frac):.3f}  coupled(real GTC L20)={fmt(a_r)}  coupled(unif-GTC)={fmt(a_g)}"
              f"  annotation-free(alignment)={fmt(a_a)}")


def main():
    recs, gtc, al = load()
    part_a(recs, gtc, al)
    part_b(recs, gtc, al)
    part_c(recs, al)


if __name__ == "__main__":
    main()
