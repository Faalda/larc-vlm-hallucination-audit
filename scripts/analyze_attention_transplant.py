"""Analysis for run_attention_transplant.py's output. Reports flip rates (Wilson CI) for
transplanting a real, naturalistic grounded-mention attention pattern vs. the same pattern
spatially shuffled, and an exact McNemar paired test (both conditions applied to every mention).

Usage: python analyze_attention_transplant.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((center - margin) / denom, (center + margin) / denom)


def binom_pmf(k, n, p=0.5):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    target = binom_pmf(k, n)
    return sum(binom_pmf(i, n) for i in range(n + 1) if binom_pmf(i, n) <= target * (1 + 1e-9))


def main():
    r = json.load(open(ROOT / "results" / "attention_transplant_records.json"))
    n = len(r)
    donor_flips = sum(1 for x in r if x["transplant_donor"]["flip"])
    shuf_flips = sum(1 for x in r if x["transplant_shuffled"]["flip"])
    donor_grounded = sum(1 for x in r if x["transplant_donor"]["grounded_replacement"])
    shuf_grounded = sum(1 for x in r if x["transplant_shuffled"]["grounded_replacement"])

    print(f"n={n} hallucinated mentions in 'mixed' images (83 images, at least one grounded "
          f"mention as attention donor, same image)")
    for name, k in [("transplant_donor (real, naturalistic pattern)", donor_flips),
                    ("transplant_shuffled (same pattern, spatially scrambled)", shuf_flips)]:
        lo, hi = wilson_ci(k, n)
        print(f"  {name}: flip rate {k}/{n} = {k/n:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"\n  grounded-replacement: donor={donor_grounded}/{n}  shuffled={shuf_grounded}/{n}")

    both = sum(1 for x in r if x["transplant_donor"]["flip"] and x["transplant_shuffled"]["flip"])
    donor_only = sum(1 for x in r if x["transplant_donor"]["flip"] and not x["transplant_shuffled"]["flip"])
    shuf_only = sum(1 for x in r if not x["transplant_donor"]["flip"] and x["transplant_shuffled"]["flip"])
    neither = n - both - donor_only - shuf_only
    p = mcnemar_exact_p(donor_only, shuf_only)
    print(f"\nPaired 2x2: both={both} donor_only={donor_only} shuffled_only={shuf_only} neither={neither}")
    print(f"Exact McNemar p-value (donor vs shuffled flip rate, paired): {p:.6f}")


if __name__ == "__main__":
    main()
