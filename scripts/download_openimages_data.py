"""Download a subset of Open Images V5 validation images with GT bounding boxes,
producing a pilot_index.json-compatible file so the existing GTC/alignment pipeline
can run on it without any changes.

Images come from open-images-dataset.s3.amazonaws.com (AWS mirror, no auth required).
Bbox annotations from storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv.
Class names from storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv.

We map Open Images /m/xxx MIDs to COCO-80 category names (those overlapping both
datasets) so coco_categories.py's CHAIR synonym list applies unchanged to the captions.
"""
import argparse
import csv
import io
import json
import os
import random

import requests
from PIL import Image as PILImage

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
IMG_URL_TMPL = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
BBOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
CLASS_URL = "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"

# Intersection of OI category names with COCO-80 category names that
# coco_categories.py's CHAIR synonym list already covers.
# Keyed by the lower-cased OI display name -> the canonical COCO category name
# used in the CHAIR synonym list (identical where they match).
OI_TO_COCO_CATEGORY = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "boat",
    "traffic light": "traffic light",
    "fire hydrant": "fire hydrant",
    "stop sign": "stop sign",
    "bench": "bench",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "handbag": "handbag",
    "tie": "tie",
    "suitcase": "suitcase",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "sports ball",
    "kite": "kite",
    "baseball bat": "baseball bat",
    "baseball glove": "baseball glove",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis racket",
    "bottle": "bottle",
    "wine glass": "wine glass",
    "cup": "cup",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "cake",
    "chair": "chair",
    "couch": "couch",
    "sofa": "couch",    # OI uses "sofa" for what COCO calls "couch"
    "potted plant": "potted plant",
    "bed": "bed",
    "dining table": "dining table",
    "toilet": "toilet",
    "tv": "tv",
    "television": "tv",
    "laptop": "laptop",
    "mouse": "mouse",
    "remote": "remote",
    "keyboard": "keyboard",
    "cell phone": "cell phone",
    "mobile phone": "cell phone",
    "microwave": "microwave",
    "oven": "oven",
    "sink": "sink",
    "refrigerator": "refrigerator",
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    "scissors": "scissors",
    "teddy bear": "teddy bear",
    "hair dryer": "hair drier",
    "toothbrush": "toothbrush",
    "airplane": "airplane",
    "aeroplane": "airplane",
    "sandwich": "sandwich",
    "skateboard": "skateboard",
    "toaster": "toaster",
}


def fetch_class_map():
    r = requests.get(CLASS_URL, timeout=30)
    r.raise_for_status()
    mid_to_name = {}
    for row in csv.reader(io.StringIO(r.text)):
        if len(row) >= 2:
            mid_to_name[row[0]] = row[1].lower()
    return mid_to_name


def fetch_bbox_annotations(mid_to_name, coco_mids):
    """Stream the bbox CSV and collect annotations for images that have >=1 COCO category box."""
    print("downloading bbox annotations (~25MB)...")
    r = requests.get(BBOX_URL, stream=True, timeout=60)
    r.raise_for_status()

    image_objects = {}  # image_id -> list of {category, bbox_norm, area_norm}
    buf = ""
    header = None
    for chunk in r.iter_content(chunk_size=1 << 20):
        buf += chunk.decode("utf-8", errors="replace")
        lines = buf.split("\n")
        buf = lines[-1]
        for line in lines[:-1]:
            if header is None:
                header = line.strip().split(",")
                continue
            row = line.strip().split(",")
            if len(row) < 8:
                continue
            image_id, _, label, conf = row[0], row[1], row[2], row[3]
            if label not in coco_mids or float(conf) < 1:
                continue
            is_group = int(row[10]) if len(row) > 10 else 0
            if is_group:
                continue
            x_min, x_max, y_min, y_max = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            oi_name = mid_to_name.get(label, "")
            coco_name = OI_TO_COCO_CATEGORY.get(oi_name)
            if coco_name is None:
                continue
            area_norm = (x_max - x_min) * (y_max - y_min)
            image_objects.setdefault(image_id, []).append({
                "category": coco_name,
                "bbox_norm": [x_min, y_min, x_max - x_min, y_max - y_min],  # [x,y,w,h] normalized
                "area_norm": area_norm,
            })
    return image_objects


def select_images(image_objects, n_images, min_cats, max_cats, min_area_norm, seed):
    rng = random.Random(seed)
    candidates = []
    for image_id, objs in image_objects.items():
        objs = [o for o in objs if o["area_norm"] >= min_area_norm]
        cats = {o["category"] for o in objs}
        if min_cats <= len(cats) <= max_cats:
            candidates.append((image_id, objs))
    rng.shuffle(candidates)
    return candidates[:n_images]


def download_images(selected, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    index = []
    for i, (image_id, objs) in enumerate(selected):
        local_path = os.path.join(img_dir, f"{image_id}.jpg")
        if not os.path.exists(local_path):
            url = IMG_URL_TMPL.format(image_id)
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"  [skip] {image_id}: HTTP {r.status_code}")
                continue
            with open(local_path, "wb") as f:
                f.write(r.content)
        try:
            img = PILImage.open(local_path)
            w, h = img.size
        except Exception as e:
            print(f"  [skip] {image_id}: bad image ({e})")
            os.remove(local_path)
            continue

        # Convert normalized bbox to pixel coords: OI gives [xmin,ymin,w_norm,h_norm]
        objects = []
        for o in objs:
            xn, yn, wn, hn = o["bbox_norm"]
            objects.append({
                "category": o["category"],
                "bbox": [xn * w, yn * h, wn * w, hn * h],  # [x,y,w,h] in pixels
            })

        index.append({
            "image_id": image_id,
            "file_name": f"{image_id}.jpg",
            "local_path": local_path,
            "width": w,
            "height": h,
            "objects": objects,
            "source": "openimages_v5_validation",
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(selected)} downloaded")
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=300)
    ap.add_argument("--min_cats", type=int, default=1)
    ap.add_argument("--max_cats", type=int, default=4)
    ap.add_argument("--min_area_norm", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)   # different seed from COCO pilot (seed=0)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "openimages_index.json"))
    args = ap.parse_args()

    print("fetching class descriptions...")
    mid_to_name = fetch_class_map()
    coco_mids = {mid for mid, name in mid_to_name.items()
                 if name in OI_TO_COCO_CATEGORY}
    print(f"  {len(coco_mids)} COCO-overlapping MIDs found in OI class list")

    image_objects = fetch_bbox_annotations(mid_to_name, coco_mids)
    print(f"  {len(image_objects)} images with >=1 COCO bbox annotation")

    selected = select_images(image_objects, args.n_images,
                              args.min_cats, args.max_cats, args.min_area_norm, args.seed)
    print(f"selected {len(selected)} images (requested {args.n_images})")

    img_dir = os.path.join(DATA_DIR, "openimages_images")
    index = download_images(selected, img_dir)
    print(f"downloaded {len(index)} images")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {args.out} ({len(index)} images)")


if __name__ == "__main__":
    main()
