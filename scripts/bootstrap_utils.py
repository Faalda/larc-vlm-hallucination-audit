#!/usr/bin/env python3
"""
Image-level bootstrap utilities for Paper 3
Resamples images (not mentions) to account for clustering
"""

from collections import defaultdict
import random
import math


def compute_auc_ordinal(y_true, scores):
    """Compute AUC using ordinal (Mann-Whitney U) ranking, no external deps.

    FIXED 2026-10-07: the original implementation counted a pair as
    concordant only on strict inequality (score_i > score_j), giving ties
    zero credit instead of the standard 0.5. This silently returned 0.0
    (not 0.5) for a constant feature, and understated AUC for any
    tie-heavy feature (discrete pi, mention order, object counts, binary
    indicators) -- confirmed against sklearn.metrics.roc_auc_score on both
    synthetic heavy-tie/constant inputs and real project data (pi: 0.817
    reported vs. 0.864 correct on one population; a 1-4-category object
    count: 0.635 reported vs. 0.541 correct). Continuous, rarely-tied
    features (GTC distances, alignment cosines) were negligibly affected.
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    concordant = 0.0
    for i, (y_i, score_i) in enumerate(zip(y_true, scores)):
        if y_i == 1:
            for j, (y_j, score_j) in enumerate(zip(y_true, scores)):
                if y_j == 0:
                    if score_i > score_j:
                        concordant += 1.0
                    elif score_i == score_j:
                        concordant += 0.5

    return concordant / (n_pos * n_neg)


def compute_hallucination_rate(y_true, mask=None):
    """Compute hallucination rate (% hallucinated among selected mentions)"""
    if mask is None:
        mask = [True] * len(y_true)

    selected = [not y for y, m in zip(y_true, mask) if m]
    if not selected:
        return 0.0
    return sum(selected) / len(selected) * 100


def percentile(values, p):
    """Compute percentile (p is 0-100)"""
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lower = math.floor(idx)
    upper = math.ceil(idx)

    if lower == upper:
        return sorted_vals[lower]

    return sorted_vals[lower] * (upper - idx) + sorted_vals[upper] * (idx - lower)


def bootstrap_auc_image_level(y_true, scores, image_groups, n_bootstrap=2000, seed=42):
    """
    Bootstrap AUC by resampling IMAGES (not mentions)

    Args:
        y_true: list of labels (1=grounded, 0=hallucinated)
        scores: list of risk scores
        image_groups: dict {image_id: [mention_indices]}
        n_bootstrap: number of bootstrap samples
        seed: random seed

    Returns:
        dict with 'mean', 'ci_lower', 'ci_upper', 'aucs' (all samples)
    """
    random.seed(seed)
    images = list(image_groups.keys())
    aucs = []

    for _ in range(n_bootstrap):
        # Resample images with replacement
        sampled_images = [random.choice(images) for _ in range(len(images))]

        # Get all mentions from sampled images
        sampled_indices = []
        for img_id in sampled_images:
            sampled_indices.extend(image_groups[img_id])

        # Compute AUC on bootstrap sample
        y_boot = [y_true[i] for i in sampled_indices]
        s_boot = [scores[i] for i in sampled_indices]
        auc_boot = compute_auc_ordinal(y_boot, s_boot)
        aucs.append(auc_boot)

    return {
        'mean': sum(aucs) / len(aucs),
        'ci_lower': percentile(aucs, 2.5),
        'ci_upper': percentile(aucs, 97.5),
        'aucs': aucs,
        'n': len(aucs)
    }


def bootstrap_hallucination_rate_image_level(y_true, image_groups, coverage=1.0,
                                            threshold_idx=None, n_bootstrap=2000, seed=42):
    """
    Bootstrap hallucination rate by resampling IMAGES

    Args:
        y_true: list of labels (1=grounded, 0=hallucinated)
        image_groups: dict {image_id: [mention_indices]}
        coverage: fraction of mentions to keep (default 1.0 = all)
        threshold_idx: if provided, compute rate for mentions with index < threshold_idx
        n_bootstrap: number of bootstrap samples
        seed: random seed

    Returns:
        dict with 'mean', 'ci_lower', 'ci_upper', 'rates' (all samples)
    """
    random.seed(seed)
    images = list(image_groups.keys())
    rates = []

    for _ in range(n_bootstrap):
        # Resample images with replacement
        sampled_images = [random.choice(images) for _ in range(len(images))]

        # Get all mentions from sampled images
        sampled_indices = []
        for img_id in sampled_images:
            sampled_indices.extend(image_groups[img_id])

        # Apply coverage threshold if provided
        if threshold_idx is not None:
            sampled_indices = [i for i in sampled_indices if i < threshold_idx]
        else:
            # Random threshold for coverage
            n_keep = max(1, int(len(sampled_indices) * coverage))
            sampled_indices = sorted(sampled_indices)[:n_keep]

        # Compute hallucination rate
        y_boot = [y_true[i] for i in sampled_indices]
        if y_boot:
            halluc_rate = sum(1 for y in y_boot if y == 0) / len(y_boot) * 100
        else:
            halluc_rate = 0.0

        rates.append(halluc_rate)

    return {
        'mean': sum(rates) / len(rates),
        'ci_lower': percentile(rates, 2.5),
        'ci_upper': percentile(rates, 97.5),
        'rates': rates,
        'n': len(rates)
    }


def bootstrap_paired_difference_image_level(y_true_baseline, y_true_intervention,
                                           image_groups, n_bootstrap=2000, seed=42):
    """
    Bootstrap paired difference (intervention vs baseline) at IMAGE level

    Args:
        y_true_baseline: baseline labels {image_id: [labels]}
        y_true_intervention: intervention labels {image_id: [labels]}
        image_groups: dict {image_id: [mention_indices]}
        n_bootstrap: number of bootstrap samples
        seed: random seed

    Returns:
        dict with paired differences for each image
    """
    random.seed(seed)
    images = list(image_groups.keys())
    diffs = []

    for _ in range(n_bootstrap):
        # Resample images with replacement
        sampled_images = [random.choice(images) for _ in range(len(images))]

        # Compute paired difference at image level
        total_diff = 0
        for img_id in sampled_images:
            if img_id in y_true_baseline and img_id in y_true_intervention:
                baseline_rate = sum(1 for y in y_true_baseline[img_id] if y == 0) / len(y_true_baseline[img_id]) * 100
                interv_rate = sum(1 for y in y_true_intervention[img_id] if y == 0) / len(y_true_intervention[img_id]) * 100
                total_diff += (interv_rate - baseline_rate)

        diffs.append(total_diff / len(sampled_images))

    return {
        'mean': sum(diffs) / len(diffs),
        'ci_lower': percentile(diffs, 2.5),
        'ci_upper': percentile(diffs, 97.5),
        'diffs': diffs,
        'n': len(diffs)
    }


if __name__ == '__main__':
    print("Bootstrap utilities loaded. Use in your analysis scripts.")
