"""Magnitude-preserving shuffled-attention control for GTC.

Complementary to the uniform-attention geometry-only control, this test asks
whether where real attention concentrates matters independently of its shape.

For each mention, takes the model's REAL per-patch attention distribution at L20 (same values,
same entropy, same peakedness), then randomly permutes which patch each weight is assigned to
(spatially scrambled, magnitude-preserving), recomputes GTC under K such permutations, and
averages. If shuffled-GTC scores nearly as well as real GTC, real GTC's power is largely a
function of the attention distribution's SHAPE (how concentrated it is), not genuine spatial
alignment with evidence. If shuffled-GTC collapses toward the uniform-attention control's AUC,
real GTC's power depends on genuinely knowing WHERE, not just how sharply, attention concentrates.

Teacher-forced on the saved captions (same protocol as run_selfgtc_paper1.py / run_alignment_
perhead.py), reusing the same 150-image subset for direct comparability.

Usage:
    python run_gtc_shuffled_attention.py --n-images 150 --n-shuffles 20
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import bbox_to_patch_indices, compute_gtc, token_char_spans, token_index_for_char

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
L_LATE = 20


def load_local_paths():
    paths = {}
    for f in ["pilot_index.json", "pilot_index_extra.json"]:
        for r in json.load(open(os.path.join(DATA_DIR, f))):
            paths[r["image_id"]] = r
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-images", type=int, default=150)
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "gtc_shuffled_attention_records.json"))
    args = ap.parse_args()

    records = json.load(open(os.path.join(RESULTS_DIR, "pilot_records_caption_corrected.json")))
    by_image = {}
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)
    image_ids = sorted(by_image.keys())[: args.n_images]
    meta = load_local_paths()

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    tokenizer = processor.tokenizer
    image_token_index = model.config.image_token_index
    patch_size = model.config.vision_config.patch_size
    shortest_edge = processor.image_processor.size.get("shortest_edge") or processor.image_processor.size.get("height")
    crop_size = processor.image_processor.crop_size
    grid_size = crop_size["height"] // patch_size

    rng = random.Random(0)
    out_records = []
    gtc_self_checks = []
    for i, image_id in enumerate(image_ids):
        if image_id not in meta:
            continue
        mentions = by_image[image_id]
        caption_text = mentions[0]["caption"]
        img_meta = meta[image_id]
        image = Image.open(img_meta["local_path"]).convert("RGB")
        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

        prompt_inputs = processor(images=image, text=prompt, return_tensors="pt")
        prompt_ids = prompt_inputs["input_ids"][0]
        pixel_values = prompt_inputs["pixel_values"].to(args.device, dtype)
        caption_ids = tokenizer(caption_text, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]
        full_ids = torch.cat([prompt_ids, caption_ids]).unsqueeze(0).to(args.device)
        prompt_len = prompt_ids.shape[0]

        image_token_positions = (full_ids[0] == image_token_index).nonzero(as_tuple=True)[0]
        if image_token_positions.numel() != grid_size * grid_size:
            print(f"  [warn] image {image_id}: unexpected image token count -- skipping")
            continue

        with torch.no_grad():
            out = model(input_ids=full_ids, pixel_values=pixel_values, output_attentions=True)

        print(f"[{i + 1}/{len(image_ids)}] image {image_id} ({len(mentions)} mentions)", flush=True)

        found_spans = find_object_mentions(caption_text)
        if len(found_spans) != len(mentions):
            print(f"  [warn] mention count mismatch on image {image_id} -- skipping image")
            continue
        spans, _ = token_char_spans(tokenizer, caption_ids)

        gt_by_category = {}
        all_gt_patches = set()
        for obj in img_meta["objects"]:
            idxs = bbox_to_patch_indices(obj["bbox"], img_meta["width"], img_meta["height"],
                                          shortest_edge, crop_size, patch_size)
            gt_by_category.setdefault(obj["category"], set()).update(idxs)
            all_gt_patches.update(idxs)
        all_gt_patches = sorted(all_gt_patches)
        n_img = image_token_positions.shape[0]

        mismatch = False
        for m, (fcat, fstart, fend, fword) in zip(mentions, found_spans):
            if fcat != m["category"] or fword.lower() != m["word"].lower():
                mismatch = True
                break
            tok_idx = token_index_for_char(spans, fend)
            if tok_idx is None:
                continue
            q_pos = prompt_len - 1 + tok_idx

            attn_row = out.attentions[L_LATE - 1][0][:, q_pos, :].mean(dim=0)  # [seq], head-avg
            attn_over_image = attn_row[image_token_positions]
            attn_norm = (attn_over_image / (attn_over_image.sum() + 1e-8)).cpu().numpy()

            real_gtc = compute_gtc(torch.from_numpy(attn_norm), all_gt_patches, grid_size)
            saved_gtc = m["by_layer"].get(str(L_LATE), {}).get("gtc_any_object")
            if saved_gtc is not None:
                gtc_self_checks.append(abs(real_gtc - saved_gtc))

            shuffled_vals = []
            perm_idx = list(range(n_img))
            for _ in range(args.n_shuffles):
                rng.shuffle(perm_idx)
                shuffled = np.asarray(attn_norm)[perm_idx]
                shuffled_gtc = compute_gtc(torch.from_numpy(shuffled), all_gt_patches, grid_size)
                shuffled_vals.append(shuffled_gtc)

            out_records.append(dict(
                image_id=image_id, category=m["category"], word=m["word"],
                grounded=m["grounded"],
                real_gtc=real_gtc,
                shuffled_gtc_mean=float(np.mean(shuffled_vals)),
                shuffled_gtc_all=shuffled_vals,
            ))
        if mismatch:
            print(f"  [warn] mention order mismatch on image {image_id} -- partial results kept")

        with open(args.out, "w") as f:
            json.dump(out_records, f, indent=2)

    if gtc_self_checks:
        print(f"\nSelf-check GTC L20: n={len(gtc_self_checks)}, "
              f"mean abs diff={np.mean(gtc_self_checks):.4f}, max={np.max(gtc_self_checks):.4f}")
    print(f"\nwrote {len(out_records)} records to {args.out}")


if __name__ == "__main__":
    main()
