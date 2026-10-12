"""Sample a deterministic 200-item multi-rater label audit.

The sample is drawn from the 275 mentions labeled hallucinated by the original
instances-only ground truth. Raters judge whether the target category is visible
without seeing the generated caption or model features.

Produces, for at least two independent human raters:
  - results/e4_label_audit/manifest_private.json   (full records, kept out of
    the rater sheets: image_id, category, original grounded label, caption,
    computed features -- used later to join rater judgments back for kappa
    and the FP-rate recompute, never shown to raters)
  - results/e4_label_audit/rater_sheet_A.csv        (blind: annotation_id,
    image_filename, flagged_category, judgment column left blank)
  - results/e4_label_audit/rater_sheet_B.csv        (identical content, same
    row order, separate file so raters fill independently)
  - results/e4_label_audit/rater_sheet_C.csv        (spare, in case a 3rd
    rater is available; Cohen's kappa only needs 2, Fleiss' kappa can use 3)

Rating standard:
"does an object matching the flagged category appear anywhere in the image a
reasonable viewer would recognize" -- not a COCO-bounding-box-quality
judgment. Each image should be judged by opening it directly
(data/images/<image_filename>), blind to the model's caption and to any
computed feature. Judgment column values: VISIBLE / NOT_VISIBLE / AMBIGUOUS.

Usage: python scripts/sample_e4_label_audit.py
"""
import csv
import json
import random
from pathlib import Path

HALLUC_DIR = Path(__file__).resolve().parents[1]
RESULTS = HALLUC_DIR / "results"
OUT_DIR = RESULTS / "e4_label_audit"
IMAGES_DIR = HALLUC_DIR / "data" / "images"

N_SAMPLE = 200
SEED = 20261008  # distinct from the original audit's seed=42, documented here


def load_population():
    mentions = []
    for fname in ["pilot_records_v4_sensitivity.json", "pilot_records_extra.json"]:
        for r in json.load(open(RESULTS / fname)):
            mentions.append(r)
    halluc = [r for r in mentions if not r["grounded"]]
    return halluc


def image_filename(image_id):
    return f"COCO_val2014_{image_id:012d}.jpg"


def main():
    halluc = load_population()
    print(f"Population: {len(halluc)} mentions labeled hallucinated under "
          f"original (instances-only) ground truth")
    assert len(halluc) == 275, (
        f"expected 275 original hallucinated mentions, got {len(halluc)} "
        "-- population source has changed, stop and check before sampling"
    )

    rng = random.Random(SEED)
    sample = rng.sample(halluc, N_SAMPLE)

    missing_images = 0
    manifest = []
    sheet_rows = []
    for i, r in enumerate(sample):
        fname = image_filename(r["image_id"])
        if not (IMAGES_DIR / fname).exists():
            missing_images += 1
        aid = f"e4_{i:04d}"
        manifest.append({
            "annotation_id": aid,
            "image_id": r["image_id"],
            "category": r["category"],
            "word": r.get("word"),
            "caption": r.get("caption"),
            "gtc_any_object": r.get("gtc_any_object"),
            "align_cos": r.get("align_cos"),
        })
        sheet_rows.append({
            "annotation_id": aid,
            "image_filename": fname,
            "flagged_category": r["category"],
            "judgment": "",
        })

    if missing_images:
        print(f"  [warn] {missing_images}/{N_SAMPLE} sampled images not found "
              f"under {IMAGES_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest_private.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for tag in ["A", "B", "C"]:
        path = OUT_DIR / f"rater_sheet_{tag}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["annotation_id", "image_filename",
                                               "flagged_category", "judgment"])
            w.writeheader()
            w.writerows(sheet_rows)

    print(f"Wrote {N_SAMPLE}-item manifest and 3 blind rater sheets to {OUT_DIR}")
    print("Have at least 2 people independently fill the 'judgment' column in "
          "rater_sheet_A.csv and rater_sheet_B.csv with VISIBLE / NOT_VISIBLE / "
          "AMBIGUOUS, judging data/images/<image_filename> against "
          "<flagged_category> only -- blind to caption and computed features.")


if __name__ == "__main__":
    main()
