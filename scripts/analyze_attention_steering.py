"""Analysis for run_attention_steering.py's output. Reports flip rates (Wilson 95% CI, matching
Table 2's existing convention) for attention steered onto ground-truth patches vs. a size-matched
random patch set, and compares against the existing input-masking intervention
(results/intervention_records_v2.json) for scale.

Usage: python analyze_attention_steering.py
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


def main():
    path = ROOT / "results" / "attention_steering_records_perturbation.json"
    if not path.exists():
        path = ROOT / "results" / "attention_steering_records.json"
    r = json.load(open(path))
    n = len(r)
    gt_flips = sum(1 for x in r if x["steer_gt"]["flip"])
    rand_flips = sum(1 for x in r if x["steer_random"]["flip"])
    gt_grounded = sum(1 for x in r if x["steer_gt"]["grounded_replacement"])
    rand_grounded = sum(1 for x in r if x["steer_random"]["grounded_replacement"])
    patches = [x["n_gt_patches"] for x in r]

    print(f"n={n} hallucinated mentions ({len({x['image_id'] for x in r})}-image subset of the first COCO batch)")
    print(f"mean target patch count: {sum(patches)/len(patches):.1f} / 576 "
          f"(min {min(patches)}, max {max(patches)})")

    for name, k in [("steer_gt (attention forced onto GT patches)", gt_flips),
                    ("steer_random (attention forced onto size-matched random patches)", rand_flips)]:
        lo, hi = wilson_ci(k, n)
        print(f"  {name}: flip rate {k}/{n} = {k/n:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n  steer_gt grounded-replacement count: {gt_grounded}/{n}")
    print(f"  steer_random grounded-replacement count: {rand_grounded}/{n}")

    if "hidden_delta_norm" in r[0]["steer_gt"]:
        for condition in ("steer_gt", "steer_random"):
            hidden = [x[condition]["hidden_delta_norm"] / x[condition]["hidden_base_norm"] for x in r]
            logits = [x[condition]["logit_delta_norm"] for x in r]
            print(f"  {condition} relative hidden perturbation: median={sorted(hidden)[n//2]:.3f}, "
                  f"range=[{min(hidden):.3f},{max(hidden):.3f}]; "
                  f"logit delta: median={sorted(logits)[n//2]:.1f}, range=[{min(logits):.1f},{max(logits):.1f}]")

    small = [x for x in r if x["n_gt_patches"] < 100]
    small_flips = sum(1 for x in small if x["steer_gt"]["flip"])
    lo, hi = wilson_ci(small_flips, len(small))
    print(f"\nSmall-target subset (<100/576 patches, a more constraining intervention): "
          f"n={len(small)}, steer_gt flips={small_flips} [{lo:.4f}, {hi:.4f}]")

    print("\nFor scale, the existing input-masking intervention (Table 2, hallucinated mentions):")
    masking = json.load(open(ROOT / "results" / "intervention_records_v2.json"))
    halluc = [x for x in masking if not x["grounded"]]
    restrict_flips = sum(1 for x in halluc if x["restrict_flip"])
    random_ablate_flips = sum(1 for x in halluc if x["random_ablate_flip"])
    lo1, hi1 = wilson_ci(restrict_flips, len(halluc))
    lo2, hi2 = wilson_ci(random_ablate_flips, len(halluc))
    print(f"  restrict-to-evidence: {restrict_flips}/{len(halluc)} = "
          f"{restrict_flips/len(halluc):.4f} [{lo1:.4f}, {hi1:.4f}]")
    print(f"  ablate-random-equal-size: {random_ablate_flips}/{len(halluc)} = "
          f"{random_ablate_flips/len(halluc):.4f} [{lo2:.4f}, {hi2:.4f}]")


if __name__ == "__main__":
    main()
