"""Analyze the released 200-item multi-rater label audit.

The protocol was fixed before annotation. This script reports:

  - raw contingency table for the first two raters (not kappa alone)
  - Cohen's kappa between rater A and rater B
  - two FP-rate definitions, each with a CI:
      strict      = VISIBLE / 200
      adjudicated = VISIBLE / (VISIBLE + NOT_VISIBLE)   [AMBIGUOUS excluded]
  - Wilson and exact (Clopper-Pearson) binomial CIs for both FP rates
    - disagreements between A and B adjudicated by rater C

Final label per item: A==B -> that label. A!=B and C present -> C's label.
A!=B and C absent -> left out of the final FP-rate count, reported as
"unresolved disagreement," not silently dropped.

Usage: python analyze_e4_label_audit.py

Reads results/e4_label_audit/audit_labels.jsonl. The public file omits
captions, model features, timestamps, and annotator identities.
"""
import json
import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "e4_label_audit"
AUDIT = OUT_DIR / "audit_labels.jsonl"
VALID = {"VISIBLE", "NOT_VISIBLE", "AMBIGUOUS"}


def read_audit():
    return [
        json.loads(line)
        for line in AUDIT.read_text().splitlines()
        if line.strip()
    ]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def clopper_pearson_ci(k, n, alpha=0.05):
    from scipy import stats
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def cohens_kappa(labels_a, labels_b, ids):
    cats = sorted(VALID)
    n = len(ids)
    agree = sum(1 for i in ids if labels_a[i] == labels_b[i])
    po = agree / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for i in ids if labels_a[i] == c) / n
        pb = sum(1 for i in ids if labels_b[i] == c) / n
        pe += pa * pb
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def main():
    rows = read_audit()
    ids = sorted(r["annotation_id"] for r in rows)
    by_id = {r["annotation_id"]: r for r in rows}
    a = {i: by_id[i]["label_a"] for i in ids}
    b = {i: by_id[i]["label_b"] for i in ids}
    c = {i: by_id[i]["label_c"] for i in ids if by_id[i]["label_c"]}

    invalid = [
        i for i in ids
        if a[i] not in VALID or b[i] not in VALID
        or (i in c and c[i] not in VALID)
    ]
    if invalid:
        raise ValueError(f"Invalid labels for {len(invalid)} audit items")

    print(f"n = {len(ids)} items, both raters complete\n")

    # Raw contingency table, A vs B
    cats = sorted(VALID)
    print("Contingency table (rows = rater A, cols = rater B):")
    header = "        " + "".join(f"{c2:>13}" for c2 in cats)
    print(header)
    for c1 in cats:
        row = [sum(1 for i in ids if a[i] == c1 and b[i] == c2) for c2 in cats]
        print(f"{c1:>8}" + "".join(f"{v:>13}" for v in row))
    print()

    kappa = cohens_kappa(a, b, ids)
    agree_n = sum(1 for i in ids if a[i] == b[i])
    print(f"Raw agreement: {agree_n}/{len(ids)} ({agree_n/len(ids)*100:.1f}%)")
    print(f"Cohen's kappa (A vs B): {kappa:.3f}\n")

    # Resolve final label per item
    final = {}
    unresolved = []
    for i in ids:
        if a[i] == b[i]:
            final[i] = a[i]
        elif i in c:
            final[i] = c[i]
        else:
            unresolved.append(i)
    print(f"Disagreements: {len(ids) - agree_n}; adjudicated by rater C: "
          f"{len(ids) - agree_n - len(unresolved)}; unresolved (no rater C "
          f"judgment, excluded from final counts): {len(unresolved)}\n")

    resolved_ids = [i for i in ids if i in final]
    n_visible = sum(1 for i in resolved_ids if final[i] == "VISIBLE")
    n_not_visible = sum(1 for i in resolved_ids if final[i] == "NOT_VISIBLE")
    n_ambiguous = sum(1 for i in resolved_ids if final[i] == "AMBIGUOUS")
    n_resolved = len(resolved_ids)
    print(f"Final labels (n={n_resolved} resolved of {len(ids)} sampled): "
          f"VISIBLE={n_visible}, NOT_VISIBLE={n_not_visible}, "
          f"AMBIGUOUS={n_ambiguous}")

    strict_rate = n_visible / len(ids)
    strict_wilson = wilson_ci(n_visible, len(ids))
    print(f"\nStrict FP rate = VISIBLE / {len(ids)} (sampled, not resolved, "
          f"denominator): {strict_rate:.3f} "
          f"[Wilson {strict_wilson[0]:.3f}, {strict_wilson[1]:.3f}]")

    adj_denom = n_visible + n_not_visible
    if adj_denom:
        adj_rate = n_visible / adj_denom
        adj_wilson = wilson_ci(n_visible, adj_denom)
        print(f"Adjudicated FP rate = VISIBLE / (VISIBLE+NOT_VISIBLE) "
              f"(n={adj_denom}, ambiguous excluded): {adj_rate:.3f} "
              f"[Wilson {adj_wilson[0]:.3f}, {adj_wilson[1]:.3f}]")

    try:
        strict_cp = clopper_pearson_ci(n_visible, len(ids))
        print(f"Strict FP rate, exact (Clopper-Pearson) CI: "
              f"[{strict_cp[0]:.3f}, {strict_cp[1]:.3f}]")
        if adj_denom:
            adj_cp = clopper_pearson_ci(n_visible, adj_denom)
            print(f"Adjudicated FP rate, exact (Clopper-Pearson) CI: "
                  f"[{adj_cp[0]:.3f}, {adj_cp[1]:.3f}]")
    except ImportError:
        print("(scipy not available -- exact binomial CI skipped, Wilson CI "
              "above is the one to use)")

    print(f"\nCompare to original 50-item single-rater estimate: 74.0% "
          f"[Wilson 0.604, 0.841].")


if __name__ == "__main__":
    main()
