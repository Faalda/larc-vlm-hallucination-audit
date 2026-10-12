"""Download a small COCO val2014 subset (images + instance annotations) for the
grounding-geometry pilot.

Downloads the official annotations zip once (~250MB, cached), extracts only
instances_val2014.json, then fetches N individual images directly from
images.cocodataset.org (a few hundred KB each) instead of the full ~6GB
val2014 image archive.
"""
import argparse
import json
import os
import random
import zipfile

import requests
from pycocotools.coco import COCO

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
IMG_URL_TMPL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"


def ensure_annotations():
    ann_path = os.path.join(DATA_DIR, "annotations", "instances_val2014.json")
    if os.path.exists(ann_path):
        return ann_path
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "annotations_trainval2014.zip")
    if not os.path.exists(zip_path):
        print("downloading annotations zip (~250MB)...")
        r = requests.get(ANN_ZIP_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("extracting instances_val2014.json...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/instances_val2014.json", DATA_DIR)
    os.remove(zip_path)
    return ann_path


def select_images(coco, n_images, min_cats, max_cats, min_area_frac, seed):
    rng = random.Random(seed)
    img_ids = coco.getImgIds()
    rng.shuffle(img_ids)

    selected = []
    for img_id in img_ids:
        if len(selected) >= n_images:
            break
        info = coco.loadImgs(img_id)[0]
        img_area = info["width"] * info["height"]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        anns = [a for a in anns if a["area"] >= min_area_frac * img_area]
        cats = {coco.loadCats(a["category_id"])[0]["name"] for a in anns}
        if not (min_cats <= len(cats) <= max_cats):
            continue
        selected.append((info, anns))
    return selected


def download_images(selected, seed):
    img_dir = os.path.join(DATA_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    index = []
    for info, anns in selected:
        file_name = info["file_name"]
        local_path = os.path.join(img_dir, file_name)
        if not os.path.exists(local_path):
            url = IMG_URL_TMPL.format(info["id"])
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
        objects = []
        for a in anns:
            cat_name = None
            objects.append({
                "category_id": a["category_id"],
                "bbox": a["bbox"],  # [x, y, w, h] in original image pixels
            })
        index.append({
            "image_id": info["id"],
            "file_name": file_name,
            "local_path": local_path,
            "width": info["width"],
            "height": info["height"],
            "objects": objects,
        })
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=60)
    ap.add_argument("--min_cats", type=int, default=1)
    ap.add_argument("--max_cats", type=int, default=4)
    ap.add_argument("--min_area_frac", type=float, default=0.01,
                     help="drop GT boxes smaller than this fraction of image area")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann_path = ensure_annotations()
    coco = COCO(ann_path)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    selected = select_images(coco, args.n_images, args.min_cats, args.max_cats,
                              args.min_area_frac, args.seed)
    print(f"selected {len(selected)} images (requested {args.n_images})")

    index = download_images(selected, args.seed)
    for rec in index:
        for obj in rec["objects"]:
            obj["category"] = cat_id_to_name[obj["category_id"]]

    out_path = os.path.join(DATA_DIR, "pilot_index.json")
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {out_path} ({len(index)} images)")


if __name__ == "__main__":
    main()
