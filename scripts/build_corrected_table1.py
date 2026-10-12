"""Full reframe (Option B): recompute Table 1's headline numbers on corrected labels
(grounded_corrected, not the original instance-only grounded field), using the exact
same methodology as analyze_final.py (the canonical Table-1-generating script):
StratifiedKFold(5), StandardScaler + LogisticRegression, 2000-sample bootstrap CI.

Covers:
  - GTC layer sweep + best layer (reusing analyze_final.analyze on corrected labels)
  - alignment (head-averaged approximation) layer sweep + best layer (same call)
  - GTC + alignment (head-averaged) combined (same call)
  - per-head alignment alone (L8) and GTC(L20) + per-head alignment combined,
    merged onto the corrected-label population via the verified per-image
    positional join (alignment_perhead_full_records.json has the same per-image
    mention order as pilot_records_caption_corrected.json -- checked, 0 mismatches
    across all 592 images / 1649 mentions)
  - self-GTC alone (same positional join against selfgtc_paper1_records.json)

Usage: python build_corrected_table1.py
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from analyze_final import analyze, LAYERS


def positional_join(base_records, other_records, other_key):
    """Join `other_records` onto `base_records` by (image_id, position-within-image),
    verifying category/word agreement first (the project's standing discipline:
    never join repeated-category mentions by name alone)."""
    def group(records):
        g = defaultdict(list)
        for r in records:
            g[r["image_id"]].append(r)
        return g

    gb, go = group(base_records), group(other_records)
    assert set(gb.keys()) == set(go.keys()), "image sets differ"
    out = []
    for iid in gb:
        lb, lo = gb[iid], go[iid]
        assert len(lb) == len(lo), f"length mismatch on image {iid}"
        for rb, ro in zip(lb, lo):
            assert rb["category"] == ro["category"] and rb["word"] == ro["word"], \
                f"content mismatch on image {iid}"
            out.append(ro[other_key])
    return out


def boot_ci(y, probs, n_boot=2000, seed=0):
    n = len(y)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], probs[idx]))
    return np.percentile(boots, [2.5, 97.5])


def cv_auc(X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probs)
    lo, hi = boot_ci(y, probs, seed=seed)
    return auc, lo, hi


def main():
    records = json.load(open("results/pilot_records_caption_corrected.json"))
    for r in records:
        r["grounded"] = r["grounded_corrected"]

    print("=" * 70)
    print("Block A: analyze_final.py's own layer-sweep + combined pipeline,")
    print("         re-run verbatim on corrected labels")
    print("=" * 70)
    best_gtc, best_align = analyze(records, "n=1649, CORRECTED labels")

    print("\n" + "=" * 70)
    print("Block B: per-head alignment reconstruction + self-GTC, corrected labels")
    print("=" * 70)
    y = np.array([1 if not r["grounded"] else 0 for r in records])
    n_halluc = int(y.sum())
    print(f"n={len(y)}, hallucinated={n_halluc} ({n_halluc/len(y)*100:.1f}%)")

    perhead_full = json.load(open("results/alignment_perhead_full_records.json"))
    align_perhead = np.array(positional_join(records, perhead_full, "align_perhead"))
    auc, lo, hi = cv_auc(align_perhead.reshape(-1, 1), y)
    print(f"  per-head alignment (L8) alone        AUC={auc:.4f} CI=[{lo:.4f},{hi:.4f}]")

    gtc_l20 = np.array([r["by_layer"]["20"]["gtc_any_object"] for r in records])
    auc_gtc20, lo20, hi20 = cv_auc(gtc_l20.reshape(-1, 1), y)
    print(f"  GTC L20 alone (fixed layer, for combo) AUC={auc_gtc20:.4f} CI=[{lo20:.4f},{hi20:.4f}]")

    combo = np.stack([gtc_l20, align_perhead], axis=1)
    auc_c, lo_c, hi_c = cv_auc(combo, y)
    print(f"  GTC(L20) + per-head alignment combined AUC={auc_c:.4f} CI=[{lo_c:.4f},{hi_c:.4f}]")

    selfgtc_recs = json.load(open("results/selfgtc_paper1_records.json"))
    self_gtc = np.array(positional_join(records, selfgtc_recs, "self_gtc"))
    auc_sg, lo_sg, hi_sg = cv_auc(self_gtc.reshape(-1, 1), y)
    print(f"  self-GTC (box-free) alone              AUC={auc_sg:.4f} CI=[{lo_sg:.4f},{hi_sg:.4f}]")

    # AUPRC / precision-at-50%-recall for the per-head alignment probe, corrected labels
    from sklearn.metrics import average_precision_score
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    probs = cross_val_predict(pipe, align_perhead.reshape(-1, 1), y, cv=cv, method="predict_proba")[:, 1]
    auprc = average_precision_score(y, probs)
    order = np.argsort(-probs)
    y_sorted = y[order]
    k = int(np.ceil(0.5 * n_halluc))
    # precision at the recall-0.5 operating point: smallest prefix capturing k positives
    cum_pos = np.cumsum(y_sorted)
    idx50 = np.searchsorted(cum_pos, k)
    precision_at_50recall = cum_pos[idx50] / (idx50 + 1)
    print(f"  per-head alignment AUPRC={auprc:.4f}, precision@50% recall={precision_at_50recall:.4f}")


if __name__ == "__main__":
    main()
