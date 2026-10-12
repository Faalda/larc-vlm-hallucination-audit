"""Builds the per-image corrected ground-truth category set (instance annotations UNION
reference-caption-mentioned categories), matching rebuild_labels_with_captions.py's logic
exactly, for reuse anywhere a script needs to check "is this category really present in the
image" using the corrected (not instances-only) ground truth -- e.g. the is_grounded_replacement
checks in the causal-intervention scripts.

Usage: python build_corrected_gt_categories.py
"""
import json
import os

from larc_audit.coco_categories import find_object_mentions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_reference_captions():
    path = os.path.join(DATA_DIR, "annotations", "captions_val2014.json")
    data = json.load(open(path))
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
    index = {}
    for f in ["pilot_index.json", "pilot_index_extra.json"]:
        for r in json.load(open(os.path.join(DATA_DIR, f))):
            index[r["image_id"]] = r

    out = {}
    for image_id, rec in index.items():
        instance_cats = {obj["category"] for obj in rec["objects"]}
        caps = ref_captions.get(image_id)
        cap_cats = caption_derived_categories(caps) if caps else set()
        out[str(image_id)] = sorted(instance_cats | cap_cats)

    out_path = os.path.join(RESULTS_DIR, "corrected_gt_categories.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(out)} images to {out_path}")


if __name__ == "__main__":
    main()
