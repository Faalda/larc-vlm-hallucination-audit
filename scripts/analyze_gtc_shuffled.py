"""Analyze the shuffled-attention control under corrected labels.

Reports AUC for real (spatially intact) versus shuffled-attention
(magnitude-preserving, spatially scrambled) GTC and a paired image-grouped
bootstrap of their difference.

Usage: python analyze_gtc_shuffled.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from bootstrap_utils import compute_auc_ordinal, bootstrap_auc_image_level  # noqa: E402


def main():
    r = json.load(open(ROOT / "results" / "gtc_shuffled_attention_records.json"))
    corrected = json.load(open(ROOT / "results" / "pilot_records_caption_corrected.json"))
    corrected_by_category = {
        (x["image_id"], x["category"]): x["grounded_corrected"]
        for x in corrected
    }
    y = [
        0 if corrected_by_category[(x["image_id"], x["category"])] else 1
        for x in r
    ]
    real = [x["real_gtc"] for x in r]
    shuffled = [x["shuffled_gtc_mean"] for x in r]

    img_groups = defaultdict(list)
    for i, x in enumerate(r):
        img_groups[x["image_id"]].append(i)

    print(f"n={len(r)} mentions, {sum(y)} hallucinated, {len(img_groups)} images "
            f"(corrected labels; 150-image subset)")

    for name, vals in [("real (spatially intact) GTC L20", real),
                        ("shuffled-attention GTC (mean of 20 shuffles/mention)", shuffled)]:
        auc = compute_auc_ordinal(y, vals)
        auc_use = max(auc, 1 - auc)
        feat = vals if auc >= 0.5 else [-v for v in vals]
        boot = bootstrap_auc_image_level(y, feat, img_groups, n_bootstrap=2000, seed=42)
        print(f"  {name:50s} AUC={auc_use:.4f} CI=[{boot['ci_lower']:.4f},{boot['ci_upper']:.4f}]")

    print(f"\nmean real GTC={np.mean(real):.4f}  mean shuffled GTC={np.mean(shuffled):.4f}  "
          f"corr(real,shuffled)={np.corrcoef(real, shuffled)[0,1]:.4f}")

    images = list(img_groups.keys())
    random.seed(42)
    diffs = []
    for _ in range(2000):
        sampled = [random.choice(images) for _ in range(len(images))]
        idxs = [i for img in sampled for i in img_groups[img]]
        y_b = [y[i] for i in idxs]
        real_b = [real[i] for i in idxs]
        shuf_b = [shuffled[i] for i in idxs]
        a_real = compute_auc_ordinal(y_b, real_b)
        a_shuf = compute_auc_ordinal(y_b, shuf_b)
        diffs.append(max(a_real, 1 - a_real) - max(a_shuf, 1 - a_shuf))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    a_real_pt = compute_auc_ordinal(y, real)
    a_shuf_pt = compute_auc_ordinal(y, shuffled)
    point_diff = max(a_real_pt, 1 - a_real_pt) - max(a_shuf_pt, 1 - a_shuf_pt)
    print(f"\nPaired bootstrap, AUC(real) - AUC(shuffled): point={point_diff:.4f} "
          f"CI=[{lo:.4f},{hi:.4f}]  P(real>shuffled)={(diffs>0).mean():.4f}")


if __name__ == "__main__":
    main()
