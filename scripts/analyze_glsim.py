"""Analysis for \\textsc{glsim-retest} (main.new.tex E2): does GLSim (Park &
Li, NeurIPS 2025), re-implemented on our own corrected-label population,
pass the same \\LARC{} checks GTC did not?

Joins results/glsim_records.json (from run_glsim.py) onto
results/pilot_records_caption_corrected.json by (image_id, category),
keeping only mentions where both agree on the category (drops the small
number of images where generation nondeterminism changed the mention set --
see run_glsim.py's own per-image mismatch warning).

Reports, matching Table 1 / the single-protocol check's own conventions:
  - GLSim alone, stratified mention-level CV (comparable to GTC 0.551,
    alignment 0.682/0.668)
  - GLSim alone, image-grouped CV (comparable to the single-protocol check's
    GTC 0.553, alignment 0.663, confounds 0.789)
  - confounds + GLSim vs. confounds alone, paired image-grouped bootstrap
    (comparable to the existing +0.010 GTC / +0.006 alignment increments)
  - the pre-specified prediction itself: does GLSim beat confounds on AUC,
    by less than 0.05 under the same protocol?

Usage: python analyze_glsim.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")


def load_image_meta():
    meta = {}
    for f in ["pilot_index.json", "pilot_index_extra.json"]:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            for r in json.load(open(path)):
                meta[r["image_id"]] = r
    return meta


def main():
    glsim = json.load(open(os.path.join(RESULTS_DIR, "glsim_records.json")))
    existing = json.load(open(os.path.join(RESULTS_DIR, "pilot_records_caption_corrected.json")))
    freq_scored_path = os.path.join(RESULTS_DIR, "frequency_prior_raw_scored.json")
    meta = load_image_meta()

    existing_lookup = {}
    for r in existing:
        existing_lookup.setdefault((r["image_id"], r["category"]), []).append(r)

    rows = []
    dropped = 0
    for g in glsim:
        key = (g["image_id"], g["category"])
        candidates = existing_lookup.get(key)
        if not candidates:
            dropped += 1
            continue
        e = candidates[0]
        img = meta.get(g["image_id"])
        if img is None:
            dropped += 1
            continue
        rows.append(dict(
            image_id=g["image_id"], category=g["category"],
            glsim_score=g["glsim_score"],
            gtc=e["by_layer"]["20"]["gtc_any_object"],
            align=e["by_layer"]["8"]["align_cos"],
            y=0 if e["grounded_corrected"] else 1,
            n_boxes=len(img["objects"]),
            area_frac=sum(bb[2] * bb[3] for bb in (o["bbox"] for o in img["objects"])) / (img["width"] * img["height"])
            if img["width"] * img["height"] else 0.0,
        ))

    print(f"n={len(rows)} joined mentions ({dropped} dropped: no matching "
          f"existing record or image metadata)")

    y = np.array([r["y"] for r in rows])
    s = np.array([r["glsim_score"] for r in rows])
    image_ids = np.array([r["image_id"] for r in rows])
    print(f"hallucinated: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

    # --- GLSim alone, stratified mention-level CV (Table 1 convention) ---
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof_strat = cross_val_predict(pipe, s.reshape(-1, 1), y, cv=cv, method="predict_proba")[:, 1]
    auc_strat = roc_auc_score(y, oof_strat)
    print(f"\nGLSim alone, stratified mention-level CV: AUC = {auc_strat:.4f}")
    print(f"  (compare: GTC 0.551, alignment per-head 0.682 / head-avg 0.668)")

    # --- GLSim alone, image-grouped CV (single-protocol check convention) ---
    def cv_auc_grouped(X, y):
        gkf = GroupKFold(n_splits=5)
        Xs = StandardScaler().fit_transform(X)
        oof = np.zeros(len(y))
        for tr, te in gkf.split(Xs, y, image_ids):
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
            clf.fit(Xs[tr], y[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return roc_auc_score(y, oof), oof

    auc_grouped, oof_glsim_grouped = cv_auc_grouped(s.reshape(-1, 1), y)
    print(f"\nGLSim alone, image-grouped CV: AUC = {auc_grouped:.4f}")
    print(f"  (compare: GTC 0.553, alignment head-avg 0.663, confounds 0.789, "
          f"all under the same protocol)")

    # --- confounds + GLSim vs confounds alone, paired image-grouped bootstrap ---
    if not os.path.exists(freq_scored_path):
        print(f"\n[skip] {freq_scored_path} not found -- cannot build the "
              f"confound baseline for the increment test")
        return
    freq_scored = json.load(open(freq_scored_path))
    freq_lookup = {(r["image_id"], r["category"]): r["freq_score"] for r in freq_scored}

    X_confound = []
    keep = []
    for i, r in enumerate(rows):
        caption = None
        for e in existing_lookup[(r["image_id"], r["category"])]:
            caption = e.get("caption", "")
            word = e.get("word", r["category"])
            break
        idx = caption.lower().find(word.lower()) if caption else -1
        p_c = idx / len(caption) if caption and idx != -1 and len(caption) > 0 else None
        freq = freq_lookup.get((r["image_id"], r["category"]))
        if p_c is None or freq is None:
            continue
        X_confound.append([r["n_boxes"], r["area_frac"], freq, p_c])
        keep.append(i)

    if len(keep) < len(rows):
        print(f"\n[note] {len(rows) - len(keep)} mentions dropped from the "
              f"increment test (missing caption position or frequency score)")

    X_confound = np.array(X_confound)
    y_k = y[keep]
    s_k = s[keep]
    image_ids_k = image_ids[keep]

    def cv_oof(X, y_, groups):
        gkf = GroupKFold(n_splits=5)
        Xs = StandardScaler().fit_transform(X)
        oof = np.zeros(len(y_))
        for tr, te in gkf.split(Xs, y_, groups):
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
            clf.fit(Xs[tr], y_[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return oof

    oof_confound = cv_oof(X_confound, y_k, image_ids_k)
    oof_confound_glsim = cv_oof(np.hstack([X_confound, s_k.reshape(-1, 1)]), y_k, image_ids_k)

    auc_confound = roc_auc_score(y_k, oof_confound)
    auc_confound_glsim = roc_auc_score(y_k, oof_confound_glsim)
    print(f"\nconfounds alone (this joined subset, n={len(y_k)}): AUC = {auc_confound:.4f}")
    print(f"confounds + GLSim: AUC = {auc_confound_glsim:.4f}")

    img_groups = defaultdict(list)
    for i, iid in enumerate(image_ids_k):
        img_groups[iid].append(i)
    images = list(img_groups.keys())
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        sampled = rng.choice(images, size=len(images), replace=True)
        idx = np.concatenate([img_groups[iid] for iid in sampled])
        yb = y_k[idx]
        if len(np.unique(yb)) < 2:
            continue
        a = roc_auc_score(yb, oof_confound[idx])
        b = roc_auc_score(yb, oof_confound_glsim[idx])
        diffs.append(b - a)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = auc_confound_glsim - auc_confound
    print(f"incremental value of GLSim beyond confounds: {point:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]")

    # --- GTC and alignment, recomputed on this exact same subset, for a
    # fully matched 5-row comparison table (not mixed with the main
    # population's Table 1 / single-protocol-check numbers) ---
    gtc_k = np.array([r["gtc"] for r in rows])[keep]
    align_k = np.array([r["align"] for r in rows])[keep]

    def report_signal(name, x_k):
        auc_g, _ = (lambda o: (roc_auc_score(y_k, o), o))(cv_oof(x_k.reshape(-1, 1), y_k, image_ids_k))
        oof_with = cv_oof(np.hstack([X_confound, x_k.reshape(-1, 1)]), y_k, image_ids_k)
        auc_with = roc_auc_score(y_k, oof_with)
        diffs_ = []
        for _ in range(2000):
            sampled = rng.choice(images, size=len(images), replace=True)
            idx = np.concatenate([img_groups[iid] for iid in sampled])
            yb = y_k[idx]
            if len(np.unique(yb)) < 2:
                continue
            diffs_.append(roc_auc_score(yb, oof_with[idx]) - roc_auc_score(yb, oof_confound[idx]))
        lo_, hi_ = np.percentile(diffs_, [2.5, 97.5])
        print(f"{name} (this joined subset, n={len(y_k)}): alone AUC = "
              f"{auc_g:.4f}, confounds+{name} = {auc_with:.4f}, increment = "
              f"{auc_with - auc_confound:+.4f} [{lo_:+.4f}, {hi_:+.4f}]")

    print(f"\n=== GTC / alignment recomputed on the exact GLSim-joined subset ===")
    report_signal("GTC", gtc_k)
    report_signal("alignment", align_k)

    print(f"\n=== Pre-specified prediction check ===")
    print(f"Prediction: GLSim beats confounds on AUC; increment < 0.05")
    beats = auc_grouped > 0.789
    increment = auc_grouped - 0.789
    print(f"GLSim alone (image-grouped) = {auc_grouped:.4f} vs. confounds "
          f"0.789: {'beats' if beats else 'does NOT beat'} confounds, "
          f"difference {increment:+.4f}")
    if beats and abs(increment) < 0.05:
        print("Prediction CONFIRMED as stated.")
    elif beats:
        print("Partially confirmed: beats confounds, but by more than the "
              "predicted <0.05 margin.")
    else:
        print("Prediction NOT confirmed: GLSim alone does not beat the "
              "confound baseline under this protocol.")


if __name__ == "__main__":
    main()
