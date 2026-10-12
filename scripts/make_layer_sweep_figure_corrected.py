"""Corrected-label regeneration of layer_sweep.pdf (main.new.tex's \\needs{} item:
the original figure, from 2026-07-20, predates the caption-derived label correction
entirely and plots every curve on the uncorrected `grounded` field).

LLaVA already has a corrected-label record file with per-layer `by_layer` data
(pilot_records_caption_corrected.json, grounded_corrected field). InstructBLIP and
PaliGemma do not, but the correction is a pure function of (image_id, category) --
instance annotations union categories named in the image's five human reference
captions -- independent of which model's caption is being scored, and
results/corrected_gt_categories.json already covers all 600 sampled images
(verified: full image-id overlap with both instructblip_records.json and
paligemma_records.json). We join that lookup onto both to build grounded_corrected
for each, rather than treat it as unavailable.

Usage: python make_layer_sweep_figure_corrected.py
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def bootstrap_auc_image_level_fast(y_true, scores, image_groups, n_bootstrap=500, seed=42):
    """Fast image-level bootstrap equivalent to the shared reference helper,
    using sklearn.roc_auc_score (vectorized, rank-based) instead of the shared utility's
    pure-Python O(n^2) compute_auc_ordinal per resample -- verified identical on this
    project's tie-fix work (both give standard 0.5 credit for ties; roc_auc_score IS the
    reference implementation compute_auc_ordinal was corrected against). The O(n^2) version
    is fine for the one-off numbers reported in prose; it is not practical at 500 resamples
    x ~9 layers x 2 fields x 3 models for a figure, which is why this exists."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true)
    s_arr = np.asarray(scores)
    images = list(image_groups.keys())
    n_img = len(images)
    aucs = []
    for _ in range(n_bootstrap):
        sampled_images = rng.choice(images, size=n_img, replace=True)
        idx = np.concatenate([image_groups[iid] for iid in sampled_images])
        y_boot = y_arr[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        aucs.append(roc_auc_score(y_boot, s_arr[idx]))
    aucs = np.array(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {'ci_lower': float(lo), 'ci_upper': float(hi), 'mean': float(aucs.mean())}


BASE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, 'assets', 'figures', 'layer_sweep_corrected.pdf')

CORRECTED_GT = json.load(open(os.path.join(RESULTS, 'corrected_gt_categories.json')))


def relabel_via_corrected_gt(recs):
    missing = 0
    for r in recs:
        cats = CORRECTED_GT.get(str(r['image_id']))
        if cats is None:
            missing += 1
            r['grounded_corrected'] = r['grounded']
        else:
            r['grounded_corrected'] = r['category'] in cats
    if missing:
        print(f"  [warn] {missing}/{len(recs)} records had no corrected-GT entry for their image")
    return recs


MODELS = [
    dict(
        label='LLaVA-1.5-7B',
        files=['pilot_records_caption_corrected.json'],
        n_layers=32,
        color='#1f77b4',
        marker='o',
        lw=1.8,
        already_corrected=True,
    ),
    dict(
        label='InstructBLIP-7B',
        files=['instructblip_records.json'],
        n_layers=32,
        color='#2ca02c',
        marker='s',
        lw=1.4,
        already_corrected=False,
    ),
    dict(
        label='PaliGemma-3B',
        files=['paligemma_records.json'],
        n_layers=18,
        color='#d62728',
        marker='^',
        lw=1.4,
        already_corrected=False,
    ),
]


def compute_aucs(recs, field, n_layers):
    """Same as make_layer_sweep_figure.py, but scored against grounded_corrected."""
    y = np.array([1 if not r['grounded_corrected'] else 0 for r in recs])
    image_ids = np.array([r['image_id'] for r in recs])
    results = []
    layers = sorted({int(k) for r in recs for k in r.get('by_layer', {}).keys()})
    for L in layers:
        vals = np.array([r.get('by_layer', {}).get(str(L), {}).get(field, np.nan)
                         for r in recs])
        mask = ~np.isnan(vals)
        if mask.sum() < 40:
            continue
        X = vals[mask].reshape(-1, 1)
        y_m = y[mask]
        if len(np.unique(y_m)) < 2:
            continue
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        probs = cross_val_predict(pipe, X, y_m, cv=cv, method='predict_proba')[:, 1]
        auc = roc_auc_score(y_m, probs)

        img_groups = defaultdict(list)
        for i, iid in enumerate(image_ids[mask]):
            img_groups[iid].append(i)
        boot = bootstrap_auc_image_level_fast(y_m, probs, img_groups,
                                               n_bootstrap=500, seed=0)
        results.append((L / n_layers, auc, L, boot['ci_lower'], boot['ci_upper']))
    return results


def load_model_records(m):
    recs = []
    for fname in m['files']:
        with open(os.path.join(RESULTS, fname)) as f:
            recs.extend(json.load(f))
    if m['already_corrected']:
        for r in recs:
            assert 'grounded_corrected' in r
    else:
        relabel_via_corrected_gt(recs)
    return recs


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=False)

    for ax, field, title in zip(axes, ['gtc_any_object', 'align_cos'],
                                 ['GTC (routing)', 'Alignment']):
        ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.55, zorder=0)

        llava_pts = None
        paligemma_pts = None
        for m in MODELS:
            recs = load_model_records(m)
            pts = compute_aucs(recs, field, m['n_layers'])
            if not pts:
                continue
            rel_depths = [p[0] for p in pts]
            aucs = [p[1] for p in pts]
            abs_layers = [p[2] for p in pts]
            ci_lo = [p[3] for p in pts]
            ci_hi = [p[4] for p in pts]

            ax.fill_between(rel_depths, ci_lo, ci_hi, color=m['color'], alpha=0.15,
                             linewidth=0, zorder=1)
            ax.plot(rel_depths, aucs, marker=m['marker'], color=m['color'],
                    linewidth=m['lw'], markersize=4.5, label=m['label'], zorder=2)

            if m['label'].startswith('LLaVA'):
                llava_pts = list(zip(rel_depths, aucs, abs_layers))
            elif m['label'].startswith('PaliGemma'):
                paligemma_pts = list(zip(rel_depths, aucs, abs_layers))

        if llava_pts:
            peak = max(llava_pts, key=lambda p: p[1])
            rel_d, auc_val, abs_L = peak
            color = MODELS[0]['color']
            if field == 'gtc_any_object':
                label_txt = f'LLaVA routing\npeak  L{abs_L}'
                xt, yt = rel_d - 0.18, auc_val + 0.035
            else:
                label_txt = f'LLaVA alignment\npeak  L{abs_L}'
                xt, yt = rel_d + 0.18, auc_val + 0.030
            ax.annotate(
                label_txt, xy=(rel_d, auc_val), xytext=(xt, yt),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                fontsize=7.5, color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=0.8, alpha=0.9),
                zorder=3,
            )

        if field == 'align_cos' and paligemma_pts:
            peak = max(paligemma_pts, key=lambda p: p[1])
            rel_d, auc_val, abs_L = peak
            color = MODELS[2]['color']
            ax.annotate(
                f'PaliGemma alignment\npeak  L{abs_L}',
                xy=(rel_d, auc_val), xytext=(rel_d - 0.30, auc_val - 0.06),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                fontsize=7.5, color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=0.8, alpha=0.9),
                zorder=3,
            )

        ax.set_xlabel('Relative depth (layer / total layers)', fontsize=9)
        ax.set_ylabel('AUC (5-fold CV)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(0.42, 0.76)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08), frameon=True)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT}")
    for m in MODELS:
        recs = load_model_records(m)
        for field in ['gtc_any_object', 'align_cos']:
            pts = compute_aucs(recs, field, m['n_layers'])
            if pts:
                peak = max(pts, key=lambda p: p[1])
                print(f"  {m['label']} {field}: peak L{peak[2]} rel={peak[0]:.2f} AUC={peak[1]:.3f}")


if __name__ == '__main__':
    main()
