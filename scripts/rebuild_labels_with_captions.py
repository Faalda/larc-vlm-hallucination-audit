"""Rebuild grounded/hallucinated labels using BOTH of standard CHAIR's ground-truth
sources -- MSCOCO instance annotations (already used) AND the five human reference
captions per image (missing from this pipeline until now, see LABEL_NOISE_AUDIT.md).

Reuses find_object_mentions (coco_categories.py) UNCHANGED on the reference captions --
the same synonym/plural matching already applied to model-generated captions, just
applied to a different text source.

Usage:
    python rebuild_labels_with_captions.py
"""
import json
import os

from larc_audit.coco_categories import find_object_mentions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_reference_captions():
    path = os.path.join(DATA_DIR, "annotations", "captions_val2014.json")
    with open(path) as f:
        data = json.load(f)
    by_image = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann["caption"])
    return by_image


def caption_derived_categories(captions):
    cats = set()
    for cap in captions:
        for category, _, _, _ in find_object_mentions(cap):
            cats.add(category)
    return cats


def main():
    ref_captions = load_reference_captions()
    print(f"loaded reference captions for {len(ref_captions)} images")

    index = {}
    for f in ["pilot_index.json", "pilot_index_extra.json"]:
        for r in json.load(open(os.path.join(DATA_DIR, f))):
            index[r["image_id"]] = r

    missing_refs = 0
    expanded_gt = {}  # image_id -> (instance_only_categories, instance_plus_caption_categories)
    for image_id, rec in index.items():
        instance_cats = {obj["category"] for obj in rec["objects"]}
        caps = ref_captions.get(image_id)
        if caps is None:
            missing_refs += 1
            expanded_gt[image_id] = (instance_cats, instance_cats)
            continue
        cap_cats = caption_derived_categories(caps)
        expanded_gt[image_id] = (instance_cats, instance_cats | cap_cats)

    print(f"{missing_refs}/{len(index)} images had NO reference captions found "
          f"(unexpected -- investigate if nonzero)")

    main_recs = json.load(open(os.path.join(RESULTS_DIR, "pilot_records_v4_sensitivity.json")))
    extra_recs = json.load(open(os.path.join(RESULTS_DIR, "pilot_records_extra.json")))
    recs = main_recs + extra_recs

    n_flipped = 0
    n_still_halluc = 0
    corrected = []
    for r in recs:
        rc = dict(r)
        instance_only, instance_plus_caption = expanded_gt.get(r["image_id"], (set(), set()))
        was_grounded = r["grounded"]
        now_grounded = r["category"] in instance_plus_caption
        rc["grounded_original"] = was_grounded
        rc["grounded_corrected"] = now_grounded
        if not was_grounded and now_grounded:
            n_flipped += 1
        elif not was_grounded and not now_grounded:
            n_still_halluc += 1
        corrected.append(rc)

    n_halluc_original = sum(1 for r in recs if not r["grounded"])
    print()
    print(f"original hallucinated count: {n_halluc_original}")
    print(f"flipped to grounded via reference-caption source: {n_flipped}")
    print(f"still hallucinated after caption-source correction: {n_still_halluc}")
    print(f"corrected hallucinated count: {n_still_halluc}  "
          f"({n_still_halluc / len(recs):.1%} of {len(recs)} mentions, "
          f"vs original {n_halluc_original / len(recs):.1%})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "pilot_records_caption_corrected.json")
    with open(out_path, "w") as f:
        json.dump(corrected, f, indent=2)
    print(f"\nwrote {len(corrected)} records (with both original and corrected labels) to {out_path}")


if __name__ == "__main__":
    main()
