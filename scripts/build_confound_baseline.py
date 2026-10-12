"""Reproduces and extends LABEL_NOISE_AUDIT.md's "fourth check": do trivial image/caption
statistics (object count, ground-truth area fraction, category frequency, normalized caption
position) predict hallucination as well as or better than GTC/alignment? The audit found this
on ORIGINAL labels (confounds alone: AUC 0.771/0.776 original/corrected, beating GTC, alignment,
and their combination). This script rebuilds the same four features directly (not reusing any
cached confound file) on CORRECTED labels, for the full reframe.

Features:
  - n_boxes: count of GT instance annotations in the image
  - area_frac: total GT instance box area / image area
    - freq_score: leave-one-image-out category frequency (already computed in
        results/frequency_prior_raw_scored.json; label-independent and reusable)
  - p_c: normalized character position of the category's first mention in the caption

Usage: python build_confound_baseline.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

def load_image_meta():
    meta = {}
    for f in ["pilot_index.json", "pilot_index_extra.json"]:
        for r in json.load(open(os.path.join(DATA_DIR, f))):
            meta[r["image_id"]] = r
    return meta


def main():
    records = json.load(open(os.path.join(RESULTS_DIR, "pilot_records_caption_corrected.json")))
    freq_scored = json.load(open(os.path.join(RESULTS_DIR,
                                               "frequency_prior_raw_scored.json")))
    freq_lookup = {(r["image_id"], r["category"]): r["freq_score"] for r in freq_scored}
    meta = load_image_meta()

    rows = []
    for r in records:
        img = meta.get(r["image_id"])
        if img is None:
            continue
        img_area = img["width"] * img["height"]
        objs = img["objects"]
        n_boxes = len(objs)
        area_frac = sum(bb[2] * bb[3] for bb in (o["bbox"] for o in objs)) / img_area if img_area else 0.0

        caption = r.get("caption", "")
        word = r.get("word", r["category"])
        idx = caption.lower().find(word.lower())
        p_c = idx / len(caption) if idx != -1 and len(caption) > 0 else None
        if p_c is None:
            continue
        freq = freq_lookup.get((r["image_id"], r["category"]))
        if freq is None:
            continue

        rows.append(dict(
            image_id=r["image_id"],
            H_original=0 if r["grounded_original"] else 1,
            H_corrected=0 if r["grounded_corrected"] else 1,
            n_boxes=n_boxes, area_frac=area_frac, freq=freq, p_c=p_c,
            gtc=r["by_layer"]["20"]["gtc_any_object"],
            align=r["by_layer"]["8"]["align_cos"],
        ))

    print(f"n={len(rows)} usable mentions (out of {len(records)})")
    image_ids = np.array([r["image_id"] for r in rows])
    img_groups = defaultdict(list)
    for i, iid in enumerate(image_ids):
        img_groups[iid].append(i)

    X_confound = np.array([[r["n_boxes"], r["area_frac"], r["freq"], r["p_c"]] for r in rows])
    gtc = np.array([r["gtc"] for r in rows]).reshape(-1, 1)
    align = np.array([r["align"] for r in rows]).reshape(-1, 1)
    X_all = np.hstack([X_confound, gtc, align])

    def cv_auc(X, y):
        gkf = GroupKFold(n_splits=5)
        Xs = StandardScaler().fit_transform(X)
        oof = np.zeros(len(y))
        for tr, te in gkf.split(Xs, y, image_ids):
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
            clf.fit(Xs[tr], y[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return roc_auc_score(y, oof), oof

    for label_name, y_key in [("ORIGINAL labels", "H_original"), ("CORRECTED labels", "H_corrected")]:
        y = np.array([r[y_key] for r in rows])
        print(f"\n--- {label_name}: n_halluc={y.sum()} ({y.mean()*100:.1f}%) ---")
        auc_confound, oof_confound = cv_auc(X_confound, y)
        auc_gtc, _ = cv_auc(gtc, y)
        auc_align, _ = cv_auc(align, y)
        auc_combined, _ = cv_auc(np.hstack([gtc, align]), y)
        auc_confound_gtc, oof_cg = cv_auc(np.hstack([X_confound, gtc]), y)
        auc_confound_align, oof_ca = cv_auc(np.hstack([X_confound, align]), y)
        auc_confound_both, _ = cv_auc(X_all, y)
        print(f"  confounds alone (n_boxes+area+freq+position): {auc_confound:.4f}")
        print(f"  GTC alone: {auc_gtc:.4f}")
        print(f"  alignment alone: {auc_align:.4f}")
        print(f"  GTC+alignment combined: {auc_combined:.4f}")
        print(f"  confounds + GTC: {auc_confound_gtc:.4f}")
        print(f"  confounds + alignment: {auc_confound_align:.4f}")
        print(f"  confounds + GTC + alignment: {auc_confound_both:.4f}")

        # incremental value tests, paired bootstrap
        for name, oof_with in [("GTC", oof_cg), ("alignment", oof_ca)]:
            feat_base = oof_confound if auc_confound >= 0.5 else [1 - v for v in oof_confound]
            feat_with = oof_with if auc_confound_gtc >= 0.5 else [1 - v for v in oof_with]
            # paired bootstrap on the difference
            import random
            rng = random.Random(0)
            images = list(img_groups.keys())
            diffs = []
            for _ in range(2000):
                sampled = [rng.choice(images) for _ in range(len(images))]
                idxs = [i for img in sampled for i in img_groups[img]]
                y_b = y[idxs]
                base_b = np.array(feat_base)[idxs]
                with_b = np.array(feat_with)[idxs]
                a_base = roc_auc_score(y_b, base_b)
                a_with = roc_auc_score(y_b, with_b)
                diffs.append(max(a_with, 1 - a_with) - max(a_base, 1 - a_base))
            diffs = np.array(diffs)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            auc_with = roc_auc_score(y, feat_with)
            auc_base = roc_auc_score(y, feat_base)
            point = max(auc_with, 1 - auc_with) - max(auc_base, 1 - auc_base)
            print(f"  incremental value of {name} beyond confounds: {point:+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
