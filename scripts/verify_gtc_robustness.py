"""Run robustness checks for GTC and alignment.

Checks include image-attention mass, image-grouped cross-validation,
replacement-token grounding, box geometry, held-out layer selection, AUPRC and
operating points, error strata, Gemma attention statistics, and a model-removed
uniform-attention geometry control.

Usage: python scripts/verify_gtc_robustness_corrected.py
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import bbox_to_patch_indices

ROOT = Path(__file__).resolve().parents[1]
from bootstrap_utils import bootstrap_auc_image_level  # noqa: E402
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_auc_ordinal(y_true, scores):
    # FIXED 2026-10-07: ties now get 0.5 credit (standard Mann-Whitney U / AUC
    # convention), matching sklearn.roc_auc_score. The previous version gave
    # ties zero credit, understating AUC for any tie-heavy feature (confirmed
    # wrong for object-count: 0.635 reported vs. 0.541 correct).
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    concordant = 0.0
    for y_i, s_i in zip(y_true, scores):
        if y_i == 1:
            for y_j, s_j in zip(y_true, scores):
                if y_j == 0:
                    if s_i > s_j:
                        concordant += 1.0
                    elif s_i == s_j:
                        concordant += 0.5
    return concordant / (n_pos * n_neg)


def load_table1_population():
    r = json.load(open(RESULTS_DIR / "pilot_records_caption_corrected.json"))
    y = [0 if x["grounded"] else 1 for x in r]
    gtc20 = [x["by_layer"]["20"]["gtc_any_object"] for x in r]
    align8 = [x["by_layer"]["8"]["align_cos"] for x in r]
    mass = [x["image_mass_total"] for x in r]
    groups = [x["image_id"] for x in r]
    return r, y, gtc20, align8, mass, groups


def check_mass_decomposition(y, gtc20, mass):
    print("=" * 78)
    print("(1) MASS-DECOMPOSITION CHECK")
    print("=" * 78)
    auc_gtc = compute_auc_ordinal(y, gtc20)
    auc_mass = compute_auc_ordinal(y, mass)
    corr = float(np.corrcoef(gtc20, mass)[0, 1])
    print(f"GTC L20 AUC (sanity check vs. paper's stated 0.635): {max(auc_gtc,1-auc_gtc):.4f}")
    print(f"image_mass_total alone AUC: {max(auc_mass,1-auc_mass):.4f}")
    print(f"corr(GTC_L20, image_mass_total): {corr:.4f}")
    print(f"mean image_mass_total (sanity check vs. paper's stated 0.154): {np.mean(mass):.4f}\n")


def check_image_grouped_cv(y, gtc20, align8, groups):
    print("=" * 78)
    print("(2) IMAGE-GROUPED CROSS-VALIDATION CHECK")
    print("=" * 78)
    y = np.array(y)
    groups = np.array(groups)

    def run_cv(X, splitter, **kw):
        Xs = StandardScaler().fit_transform(X)
        oof = np.zeros(len(y))
        for tr, te in splitter.split(Xs, y, **kw):
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
            clf.fit(Xs[tr], y[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return roc_auc_score(y, oof)

    gtc_arr = np.array(gtc20).reshape(-1, 1)
    align_arr = np.array(align8).reshape(-1, 1)
    combined = np.hstack([gtc_arr, align_arr])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    gkf = GroupKFold(n_splits=5)

    for name, X in [("GTC L20", gtc_arr), ("alignment L8", align_arr), ("combined", combined)]:
        strat = run_cv(X, skf)
        grouped = run_cv(X, gkf, groups=groups)
        print(f"{name:15s}: stratified={strat:.4f}  image-grouped={grouped:.4f}  "
              f"delta={grouped-strat:+.4f}")
    print()


def check_replacement_grounding():
    print("=" * 78)
    print("(3) REPLACEMENT-TOKEN GROUNDING CHECK")
    print("=" * 78)
    recs = json.load(open(RESULTS_DIR / "intervention_records_v2.json"))
    pi1 = json.load(open(DATA_DIR / "pilot_index.json"))
    pi2 = json.load(open(DATA_DIR / "pilot_index_extra.json"))
    gt = {}
    for r in pi1 + pi2:
        gt[r["image_id"]] = {o["category"] for o in r["objects"]}

    halluc = [r for r in recs if not r["grounded"]]
    flip_rate = sum(r["restrict_flip"] for r in halluc) / len(halluc)
    print(f"n hallucinated mentions: {len(halluc)}")
    print(f"restrict_flip rate (sanity check vs. paper's stated 0.264): {flip_rate:.4f}")

    flipped = [r for r in halluc if r["restrict_flip"]]
    n_grounded_replacement = 0
    for r in flipped:
        img_cats = gt.get(r["image_id"], set())
        mentions = find_object_mentions(r["restrict_new_word"])
        if any(c in img_cats for c, _, _, _ in mentions):
            n_grounded_replacement += 1

    frac_of_flips = n_grounded_replacement / len(flipped)
    frac_of_all_halluc = n_grounded_replacement / len(halluc)
    print(f"n flipped: {len(flipped)}")
    print(f"flips landing on a token naming a real image object: "
          f"{n_grounded_replacement}/{len(flipped)} = {frac_of_flips:.4f}")
    print(f"=> actually-grounded-replacement rate over ALL hallucinated mentions: "
          f"{frac_of_all_halluc:.4f} ({flip_rate:.4f} x {frac_of_flips:.4f})\n")


def check_box_geometry_confound(y, gtc20, align8, groups):
    print("=" * 78)
    print("(4) BOX-COUNT / BOX-AREA CONFOUND CHECK")
    print("=" * 78)
    pi1 = json.load(open(DATA_DIR / "pilot_index.json"))
    pi2 = json.load(open(DATA_DIR / "pilot_index_extra.json"))
    meta = {m["image_id"]: m for m in pi1 + pi2}

    y = np.array(y)
    gtc20 = np.array(gtc20)
    align8 = np.array(align8)
    n_boxes = np.array([len(meta[iid]["objects"]) for iid in groups], dtype=float)
    total_area_frac = np.array([
        sum(bb[2] * bb[3] for bb in (o["bbox"] for o in meta[iid]["objects"]))
        / (meta[iid]["width"] * meta[iid]["height"])
        for iid in groups
    ])

    def residualize(signal):
        A = np.column_stack([n_boxes, total_area_frac, np.ones_like(signal)])
        coef, *_ = np.linalg.lstsq(A, signal, rcond=None)
        return signal - A @ coef

    print(f"corr(GTC, n_boxes)={np.corrcoef(gtc20, n_boxes)[0,1]:.4f}  "
          f"corr(GTC, box_area_frac)={np.corrcoef(gtc20, total_area_frac)[0,1]:.4f}")
    print(f"corr(alignment, n_boxes)={np.corrcoef(align8, n_boxes)[0,1]:.4f}  "
          f"corr(alignment, box_area_frac)={np.corrcoef(align8, total_area_frac)[0,1]:.4f}")

    auc_nboxes = compute_auc_ordinal(list(y), list(n_boxes))
    auc_area = compute_auc_ordinal(list(y), list(total_area_frac))
    print(f"n_boxes alone AUC: {max(auc_nboxes,1-auc_nboxes):.4f}")
    print(f"box_area_frac alone AUC: {max(auc_area,1-auc_area):.4f}")

    for name, signal in [("GTC", gtc20), ("alignment", align8)]:
        auc_raw = compute_auc_ordinal(list(y), list(signal))
        resid = residualize(signal)
        auc_resid = compute_auc_ordinal(list(y), list(resid))
        print(f"{name}: raw AUC={max(auc_raw,1-auc_raw):.4f}  "
              f"residualized on (n_boxes, area) AUC={max(auc_resid,1-auc_resid):.4f}")
    print()


def check_independent_layer_selection(r):
    print("=" * 78)
    print("(5) INDEPENDENT LAYER SELECTION / HELD-OUT BATCH CHECK")
    print("=" * 78)
    b1 = json.load(open(RESULTS_DIR / "pilot_records_v3_chair.json"))
    b1_imgs = {x["image_id"] for x in b1}
    batch1 = [x for x in r if x["image_id"] in b1_imgs]
    batch2 = [x for x in r if x["image_id"] not in b1_imgs]
    print(f"batch1 n={len(batch1)}  batch2 n={len(batch2)}")

    def auc_at_layer(recs, layer, key):
        avail = [x for x in recs if str(layer) in x["by_layer"]]
        y = [0 if x["grounded"] else 1 for x in avail]
        vals = [x["by_layer"][str(layer)][key] for x in avail]
        auc = compute_auc_ordinal(y, vals)
        return max(auc, 1 - auc)

    layers = sorted({int(k) for x in batch1 for k in x["by_layer"].keys()})
    best_gtc_layer = max(layers, key=lambda L: auc_at_layer(batch1, L, "gtc_any_object"))
    best_align_layer = max(layers, key=lambda L: auc_at_layer(batch1, L, "align_cos"))
    print(f"Selected on batch1 only: GTC->L{best_gtc_layer} "
          f"(AUC on batch1={auc_at_layer(batch1, best_gtc_layer, 'gtc_any_object'):.4f}), "
          f"alignment->L{best_align_layer} "
          f"(AUC on batch1={auc_at_layer(batch1, best_align_layer, 'align_cos'):.4f})")

    y2 = [0 if x["grounded"] else 1 for x in batch2]
    gtc2 = [x["by_layer"][str(best_gtc_layer)]["gtc_any_object"] for x in batch2]
    align2 = [x["by_layer"][str(best_align_layer)]["align_cos"] for x in batch2]
    img_groups2 = {}
    for i, x in enumerate(batch2):
        img_groups2.setdefault(x["image_id"], []).append(i)

    for name, layer, vals in [("GTC", best_gtc_layer, gtc2), ("alignment", best_align_layer, align2)]:
        auc = compute_auc_ordinal(y2, vals)
        feat = vals if auc >= 0.5 else [-v for v in vals]
        boot = bootstrap_auc_image_level(y2, feat, img_groups2, n_bootstrap=2000, seed=42)
        print(f"{name} L{layer} on held-out batch2 (n={len(batch2)}): "
              f"AUC={max(auc,1-auc):.4f} CI=[{boot['ci_lower']:.4f},{boot['ci_upper']:.4f}]")
    print()


def check_auprc(y, gtc20, align8, groups):
    print("=" * 78)
    print("(6) AUPRC / OPERATING-POINT PRECISION-RECALL CHECK")
    print("=" * 78)
    y = np.array(y)
    groups = np.array(groups)
    base_rate = y.mean()
    print(f"base rate: {base_rate:.4f}")

    gtc_arr = np.array(gtc20).reshape(-1, 1)
    align_arr = -np.array(align8).reshape(-1, 1)  # lower alignment = riskier
    combined = np.hstack([gtc_arr, align_arr])

    def cv_scores(X):
        gkf = GroupKFold(n_splits=5)
        Xs = StandardScaler().fit_transform(X)
        oof = np.zeros(len(y))
        for tr, te in gkf.split(Xs, y, groups):
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
            clf.fit(Xs[tr], y[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return oof

    for name, X in [("GTC", gtc_arr), ("alignment", align_arr), ("combined", combined)]:
        oof = cv_scores(X)
        ap = average_precision_score(y, oof)
        prec, rec, _ = precision_recall_curve(y, oof)
        i50 = np.argmin(np.abs(rec[:-1] - 0.5))
        i30 = np.argmin(np.abs(rec[:-1] - 0.3))
        print(f"{name:10s}: AUPRC={ap:.4f} (lift={ap/base_rate:.2f}x)  "
              f"precision@recall~0.5={prec[i50]:.4f}  precision@recall~0.3={prec[i30]:.4f}")
    print()


def check_error_analysis(r, y, gtc20, align8, groups):
    print("=" * 78)
    print("(7) ERROR ANALYSIS: HIGH-CONFIDENCE FALSE POSITIVES/NEGATIVES")
    print("=" * 78)
    pi1 = json.load(open(DATA_DIR / "pilot_index.json"))
    pi2 = json.load(open(DATA_DIR / "pilot_index_extra.json"))
    meta = {m["image_id"]: m for m in pi1 + pi2}

    y = np.array(y)
    groups = np.array(groups)
    gtc_arr = np.array(gtc20).reshape(-1, 1)
    align_arr = -np.array(align8).reshape(-1, 1)
    X = np.hstack([gtc_arr, align_arr])

    gkf = GroupKFold(n_splits=5)
    Xs = StandardScaler().fit_transform(X)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(Xs, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
        clf.fit(Xs[tr], y[tr])
        oof[te] = clf.predict_proba(Xs[te])[:, 1]

    thr_hi = np.quantile(oof, 0.9)
    thr_lo = np.quantile(oof, 0.1)
    fp_idx = [i for i in range(len(r)) if y[i] == 0 and oof[i] >= thr_hi]
    fn_idx = [i for i in range(len(r)) if y[i] == 1 and oof[i] <= thr_lo]
    print(f"n FP (grounded, high predicted risk)={len(fp_idx)}  "
          f"n FN (hallucinated, low predicted risk)={len(fn_idx)}")

    from collections import Counter, defaultdict
    fp_cats = Counter(r[i]["category"] for i in fp_idx)
    print(f"FP top categories: {fp_cats.most_common(6)}")

    def pos(rec):
        cap, w = rec["caption"].lower(), rec["word"].lower()
        idx = cap.find(w)
        return idx / len(cap) if idx != -1 and len(cap) > 0 else None

    def area_frac(rec):
        m = meta.get(rec["image_id"])
        objs = [o for o in m["objects"] if o["category"] == rec["category"]]
        if not objs:
            return None
        img_area = m["width"] * m["height"]
        return sum(bb[2] * bb[3] for bb in (o["bbox"] for o in objs)) / img_area if img_area else None

    fp_area = [a for i in fp_idx if (a := area_frac(r[i])) is not None]
    all_grounded_area = [a for x in r if x["grounded"] and (a := area_frac(x)) is not None]
    print(f"mean GT box area frac: all grounded={np.mean(all_grounded_area):.4f}  "
          f"FP={np.mean(fp_area):.4f}")

    by_img = defaultdict(list)
    for x in r:
        by_img[x["image_id"]].append(x["grounded"])
    mixed_images = {iid for iid, gs in by_img.items() if len(set(gs)) > 1}
    all_mixed_rate = sum(1 for x in r if x["image_id"] in mixed_images) / len(r)
    fp_mixed = sum(1 for i in fp_idx if r[i]["image_id"] in mixed_images) / len(fp_idx)
    fn_mixed = sum(1 for i in fn_idx if r[i]["image_id"] in mixed_images) / len(fn_idx)
    print(f"mixed-image rate: overall={all_mixed_rate:.4f}  FP={fp_mixed:.4f}  FN={fn_mixed:.4f}")

    all_pos = [p for x in r if (p := pos(x)) is not None]
    fp_pos = [p for i in fp_idx if (p := pos(r[i])) is not None]
    fn_pos = [p for i in fn_idx if (p := pos(r[i])) is not None]
    print(f"mean position: all={np.mean(all_pos):.4f}  FP={np.mean(fp_pos):.4f}  "
          f"FN={np.mean(fn_pos):.4f}")
    print()


def check_gemma_attention_statistics():
    print("=" * 78)
    print("(8) GEMMA MECHANISM CHECK: coarse attention statistics across architectures")
    print("=" * 78)
    llava = json.load(open(RESULTS_DIR / "pilot_records_caption_corrected.json"))
    pali = json.load(open(RESULTS_DIR / "paligemma_records.json"))
    clipgemma = json.load(open(RESULTS_DIR / "llava_gemma_records.json"))
    for name, recs in [("LLaVA (Vicuna-7B)", llava), ("PaliGemma (Gemma-2B)", pali),
                        ("CLIP+Gemma (Gemma-2B)", clipgemma)]:
        mass = [x["image_mass_total"] for x in recs]
        ent = [x["entropy"] for x in recs]
        maxa = [x["max_attn"] for x in recs]
        print(f"{name:24s}: n={len(recs)}  mass={np.mean(mass):.4f}  "
              f"entropy={np.mean(ent):.4f}  max_attn={np.mean(maxa):.4f}")
    print()


GRID_SIZE = 24
SHORTEST_EDGE = 336
CROP_SIZE = {"width": 336, "height": 336}
PATCH_SIZE = 14
_COORDS = np.array([(i // GRID_SIZE, i % GRID_SIZE) for i in range(GRID_SIZE * GRID_SIZE)],
                    dtype=np.float32)


def uniform_gtc_for_image(img_meta):
    all_gt_patches = set()
    for obj in img_meta["objects"]:
        idxs = bbox_to_patch_indices(obj["bbox"], img_meta["width"], img_meta["height"],
                                      SHORTEST_EDGE, CROP_SIZE, PATCH_SIZE)
        all_gt_patches.update(idxs)
    if not all_gt_patches:
        return None
    target_coords = _COORDS[sorted(all_gt_patches)]
    dists = np.min(np.linalg.norm(_COORDS[:, None, :] - target_coords[None, :, :], axis=-1), axis=1)
    return float(np.mean(dists))  # uniform attention (1/N each) -> plain mean distance


def check_geometry_only_control(r):
    print("=" * 78)
    print("(9) GEOMETRY-ONLY CONTROL: uniform attention, no model, no forward pass")
    print("=" * 78)
    pi1 = json.load(open(DATA_DIR / "pilot_index.json"))
    pi2 = json.load(open(DATA_DIR / "pilot_index_extra.json"))
    meta = {m["image_id"]: m for m in pi1 + pi2}

    img_cache = {}
    y, real_gtc, unif_gtc, image_ids, align_vals = [], [], [], [], []
    for x in r:
        iid = x["image_id"]
        if iid not in img_cache:
            img_cache[iid] = uniform_gtc_for_image(meta[iid])
        u = img_cache[iid]
        if u is None:
            continue
        y.append(0 if x["grounded"] else 1)
        real_gtc.append(x["by_layer"]["20"]["gtc_any_object"])
        unif_gtc.append(u)
        image_ids.append(iid)
        align_vals.append(x["by_layer"]["8"]["align_cos"])

    real_gtc = np.array(real_gtc)
    unif_gtc = np.array(unif_gtc)
    align_vals = np.array(align_vals)

    auc_unif = compute_auc_ordinal(y, list(unif_gtc))
    print(f"n={len(y)}  uniform-attention GTC AUC={max(auc_unif,1-auc_unif):.4f}")
    print(f"corr(real GTC L20, uniform-attention GTC)={np.corrcoef(real_gtc, unif_gtc)[0,1]:.4f}")
    print(f"corr(alignment L8, uniform-attention GTC)={np.corrcoef(align_vals, unif_gtc)[0,1]:.4f}")

    A = np.column_stack([unif_gtc, np.ones_like(unif_gtc)])
    coef, *_ = np.linalg.lstsq(A, real_gtc, rcond=None)
    resid = real_gtc - A @ coef
    auc_resid = compute_auc_ordinal(y, list(resid))
    auc_raw = compute_auc_ordinal(y, list(real_gtc))
    print(f"real GTC raw AUC={max(auc_raw,1-auc_raw):.4f}  "
          f"residualized on uniform-attention GTC AUC={max(auc_resid,1-auc_resid):.4f}")

    img_groups = defaultdict(list)
    for i, iid in enumerate(image_ids):
        img_groups[iid].append(i)
    feat = list(resid) if auc_resid >= 0.5 else list(-resid)
    boot = bootstrap_auc_image_level(y, feat, img_groups, n_bootstrap=2000, seed=42)
    print(f"residualized CI=[{boot['ci_lower']:.4f},{boot['ci_upper']:.4f}]")
    feat_u = list(unif_gtc) if auc_unif >= 0.5 else list(-unif_gtc)
    boot_u = bootstrap_auc_image_level(y, feat_u, img_groups, n_bootstrap=2000, seed=42)
    print(f"uniform-attention-alone CI=[{boot_u['ci_lower']:.4f},{boot_u['ci_upper']:.4f}]")
    print()


def main():
    r, y, gtc20, align8, mass, groups = load_table1_population()
    print(f"Loaded {len(r)} mentions ({sum(y)} hallucinated) from "
          f"pilot_records_caption_corrected.json\n")
    check_mass_decomposition(y, gtc20, mass)
    check_image_grouped_cv(y, gtc20, align8, groups)
    check_replacement_grounding()
    check_box_geometry_confound(y, gtc20, align8, groups)
    check_independent_layer_selection(r)
    check_auprc(y, gtc20, align8, groups)
    check_error_analysis(r, y, gtc20, align8, groups)
    check_gemma_attention_statistics()
    check_geometry_only_control(r)


if __name__ == "__main__":
    main()
