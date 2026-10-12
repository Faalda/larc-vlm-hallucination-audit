"""Analysis for run_attention_steering_multilayer.py's output. Reports flip rates (Wilson CI),
an exact McNemar paired test for steer_gt vs. steer_random (same mentions, both conditions
applied to every one -- paired, not independent), and the grounded-replacement rate conditional
on a flip having occurred.

Usage: python analyze_attention_steering_multilayer.py
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
    """b, c: discordant pair counts. Exact two-sided binomial test on min(b,c) of (b+c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    target = binom_pmf(k, n)
    return sum(binom_pmf(i, n) for i in range(n + 1) if binom_pmf(i, n) <= target * (1 + 1e-9))


def main():
    r = json.load(open(ROOT / "results" / "attention_steering_multilayer_records.json"))
    n = len(r)
    gt_flips = sum(1 for x in r if x["steer_gt"]["flip"])
    rand_flips = sum(1 for x in r if x["steer_random"]["flip"])

    print(f"n={n} hallucinated mentions, steering layers L18-22 simultaneously")
    for name, k in [("steer_gt", gt_flips), ("steer_random", rand_flips)]:
        lo, hi = wilson_ci(k, n)
        print(f"  {name}: flip rate {k}/{n} = {k/n:.4f} [{lo:.4f}, {hi:.4f}]")

    both = sum(1 for x in r if x["steer_gt"]["flip"] and x["steer_random"]["flip"])
    gt_only = sum(1 for x in r if x["steer_gt"]["flip"] and not x["steer_random"]["flip"])
    rand_only = sum(1 for x in r if not x["steer_gt"]["flip"] and x["steer_random"]["flip"])
    neither = n - both - gt_only - rand_only
    p = mcnemar_exact_p(gt_only, rand_only)
    print(f"\nPaired 2x2: both={both} gt_only={gt_only} random_only={rand_only} neither={neither}")
    print(f"Exact McNemar p-value (gt vs random flip rate, paired): {p:.4f}")

    gt_grounded = sum(1 for x in r if x["steer_gt"]["grounded_replacement"])
    rand_grounded = sum(1 for x in r if x["steer_random"]["grounded_replacement"])
    print(f"\ngrounded-replacement count: steer_gt={gt_grounded}/{n}, steer_random={rand_grounded}/{n}")
    print(f"grounded-replacement RATE AMONG FLIPS: "
          f"steer_gt={gt_grounded}/{gt_flips}={gt_grounded/gt_flips:.4f}  "
          f"steer_random={rand_grounded}/{rand_flips}={rand_grounded/rand_flips:.4f}")

    print("\nFor comparison, single-layer (L20 only) results:")
    r1 = json.load(open(ROOT / "results" / "attention_steering_records.json"))
    n1 = len(r1)
    gt1 = sum(1 for x in r1 if x["steer_gt"]["flip"])
    rand1 = sum(1 for x in r1 if x["steer_random"]["flip"])
    print(f"  L20 only:  steer_gt={gt1}/{n1}={gt1/n1:.4f}  steer_random={rand1}/{n1}={rand1/n1:.4f}")
    print(f"  L18-22:    steer_gt={gt_flips}/{n}={gt_flips/n:.4f}  steer_random={rand_flips}/{n}={rand_flips/n:.4f}")


if __name__ == "__main__":
    main()
