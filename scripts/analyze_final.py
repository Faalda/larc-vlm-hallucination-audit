"""Canonical, standardized-pipeline reanalysis for the paper's Table 1 and
cross-architecture Table 3 numbers. Supersedes run_pilot.py's analyze()
function, which never had StandardScaler and produces different (wrong,
relative to the paper's stated methodology) combined-feature AUCs.

Run this any time the underlying pilot_records_*.json files change, and
compare its output with the released expected values before trusting a change.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

LAYERS = [1, 4, 6, 7, 8, 9, 10, 12, 16, 18, 19, 20, 21, 22, 24, 28, 32]


def analyze(records, label, n_boot=2000, seed=0):
    y = np.array([1 if not r["grounded"] else 0 for r in records])
    n = len(y)
    print(f"\n=== {label}: n={n} mentions, {y.sum()} hallucinated "
          f"({y.sum()/n*100:.1f}%) ===")

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)

    def boot_ci(probs):
        boots = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            boots.append(roc_auc_score(y[idx], probs[idx]))
        return np.percentile(boots, [2.5, 97.5])

    def run(X, name):
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)
        lo, hi = boot_ci(probs)
        print(f"  {name:40s} AUC={auc:.4f}  CI=[{lo:.4f},{hi:.4f}]")
        return auc

    entropy = np.array([r["entropy"] for r in records])
    maxattn = np.array([r["max_attn"] for r in records])
    prior_gray = np.array([r["prior_delta_gray"] for r in records])
    prior_zero = np.array([r["prior_delta_zero"] for r in records])

    run(entropy.reshape(-1, 1), "entropy (baseline)")
    run(maxattn.reshape(-1, 1), "max attention weight (baseline)")

    print("  [layer sweep] GTC:")
    best_gtc, best_gtc_auc = None, -1
    for L in LAYERS:
        vals = np.array([r["by_layer"][str(L)]["gtc_any_object"] for r in records])
        auc = run(vals.reshape(-1, 1), f"    GTC L={L}")
        if auc > best_gtc_auc:
            best_gtc, best_gtc_auc = L, auc

    print("  [layer sweep] alignment:")
    best_align, best_align_auc = None, -1
    for L in LAYERS:
        vals = np.array([r["by_layer"][str(L)]["align_cos"] for r in records])
        auc = run(vals.reshape(-1, 1), f"    align L={L}")
        if auc > best_align_auc:
            best_align, best_align_auc = L, auc

    print(f"  best GTC layer: L{best_gtc} ({best_gtc_auc:.4f})")
    print(f"  best align layer: L{best_align} ({best_align_auc:.4f})")

    gtc_best = np.array([r["by_layer"][str(best_gtc)]["gtc_any_object"] for r in records])
    align_best = np.array([r["by_layer"][str(best_align)]["align_cos"] for r in records])

    run(np.stack([gtc_best, align_best], axis=1), "GTC + alignment (combined)")
    run(np.stack([gtc_best, align_best, prior_gray, prior_zero], axis=1),
        "GTC + alignment + both priors")
    run(prior_gray.reshape(-1, 1), "prior-override, gray null")
    run(prior_zero.reshape(-1, 1), "prior-override, zero-embed null")

    return best_gtc, best_align


if __name__ == "__main__":
    with open("results/pilot_records_v4_sensitivity.json") as f:
        orig = json.load(f)
    with open("results/pilot_records_extra.json") as f:
        extra = json.load(f)

    analyze(orig, "n=817 (original batch, matches Table 3 / OI-comparison basis)")
    analyze(orig + extra, "n=1649 (extended sample, matches Table 1 basis)")
