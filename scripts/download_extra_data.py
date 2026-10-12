"""Download 300 ADDITIONAL COCO val2014 images (disjoint from pilot_index.json)
to double the hallucinated-mention sample. Same selection criteria as the pilot,
different seed, explicit exclusion of already-used image ids."""
import json
import os

from pycocotools.coco import COCO

from download_pilot_data import ensure_annotations, select_images, download_images

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def main():
    with open(os.path.join(DATA_DIR, "pilot_index.json")) as f:
        used_ids = {rec["image_id"] for rec in json.load(f)}
    print(f"excluding {len(used_ids)} already-used images")

    ann_path = ensure_annotations()
    coco = COCO(ann_path)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # over-select, then drop overlaps with the original pilot set
    selected = select_images(coco, 400, min_cats=1, max_cats=4,
                             min_area_frac=0.01, seed=2)
    selected = [(info, anns) for info, anns in selected
                if info["id"] not in used_ids][:300]
    print(f"selected {len(selected)} new images")

    index = download_images(selected, seed=2)
    for rec in index:
        for obj in rec["objects"]:
            obj["category"] = cat_id_to_name[obj["category_id"]]

    out_path = os.path.join(DATA_DIR, "pilot_index_extra.json")
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {out_path} ({len(index)} images)")


if __name__ == "__main__":
    main()
