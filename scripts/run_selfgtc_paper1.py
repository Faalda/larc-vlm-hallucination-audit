"""Compute box-free self-GTC on the released mention population.

The run is teacher-forced on captions in
results/pilot_records_caption_corrected.json rather than regenerating them.

Instead of ground-truth object patches, self-GTC uses the model's own estimate
of where the imminent concept is represented (top-K
image patches by cosine similarity to the pre-generation hidden state at an early layer),
requiring no ground truth at inference time.

The fixed design uses L8 for retrieval, L20 for routing, and TOPK=10.

Usage:
    python scripts/run_selfgtc_paper1.py --n-images 592
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import (
    bbox_to_patch_indices,
    compute_gtc,
    token_char_spans,
    token_index_for_char,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

L_EARLY = 8
L_LATE = 20
TOPK = 10


def self_region_patches(image_hidden, h_prev, k):
    sims = F.cosine_similarity(image_hidden.float(), h_prev.float().unsqueeze(0), dim=-1)
    k = min(k, sims.shape[0])
    return sims.topk(k).indices.cpu().tolist()


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
    ap.add_argument("--n-images", type=int, default=592)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "selfgtc_paper1_records.json"))
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

    out_records = []
    gtc_self_checks, align_self_checks = [], []
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
            print(f"  [warn] image {image_id}: expected {grid_size*grid_size} image tokens, "
                  f"found {image_token_positions.numel()} -- skipping")
            continue

        with torch.no_grad():
            out = model(input_ids=full_ids, pixel_values=pixel_values,
                        output_attentions=True, output_hidden_states=True)

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

        # out.hidden_states is a flat per-layer tuple from a plain forward pass (not the
        # step-wise generate() structure get_image_hidden expects), so index directly.
        image_hidden_early = out.hidden_states[L_EARLY][0][image_token_positions]

        mismatch = False
        for m, (fcat, fstart, fend, fword) in zip(mentions, found_spans):
            if fcat != m["category"] or fword.lower() != m["word"].lower():
                mismatch = True
                break
            tok_idx = token_index_for_char(spans, fend)
            if tok_idx is None:
                continue
            q_pos = prompt_len - 1 + tok_idx

            # Oracle GTC at L20, recomputed here as a self-check against the saved value.
            # Plain-forward-pass layout: attentions[layer-1][0] is [heads, seq, seq] over the
            # WHOLE sequence (not a step-wise single-query slice), so index q_pos explicitly.
            attn_row_20 = out.attentions[L_LATE - 1][0][:, q_pos, :].mean(dim=0)  # [seq], head-avg
            attn_over_image_20 = attn_row_20[image_token_positions]
            attn_norm_20 = attn_over_image_20 / (attn_over_image_20.sum() + 1e-8)
            oracle_gtc_recomputed = compute_gtc(attn_norm_20, all_gt_patches, grid_size)
            saved_gtc = m["by_layer"].get(str(L_LATE), {}).get("gtc_any_object")
            if saved_gtc is not None:
                gtc_self_checks.append(abs(oracle_gtc_recomputed - saved_gtc))

            # Alignment at L8, recomputed as a self-check.
            h_prev_8 = out.hidden_states[L_EARLY][0][q_pos]
            attn_row_8 = out.attentions[L_EARLY - 1][0][:, q_pos, :].mean(dim=0)
            attn_over_image_8 = attn_row_8[image_token_positions]
            attn_norm_8 = attn_over_image_8 / (attn_over_image_8.sum() + 1e-8)
            v_t_8 = (attn_norm_8.unsqueeze(-1).to(image_hidden_early.dtype) * image_hidden_early).sum(dim=0)
            align_recomputed = F.cosine_similarity(
                h_prev_8.float().unsqueeze(0), v_t_8.float().unsqueeze(0)).item()
            saved_align = m["by_layer"].get(str(L_EARLY), {}).get("align_cos")
            if saved_align is not None:
                align_self_checks.append(abs(align_recomputed - saved_align))

            # Self-GTC: box-free, using the model's own top-K patches at L_early as the
            # "evidence" set instead of ground-truth boxes.
            self_patches = self_region_patches(image_hidden_early, h_prev_8, TOPK)
            self_gtc = compute_gtc(attn_norm_20, self_patches, grid_size)

            out_records.append(dict(
                image_id=image_id, category=m["category"], word=m["word"],
                grounded=m["grounded"],
                oracle_gtc_recomputed=oracle_gtc_recomputed,
                align_cos_recomputed=align_recomputed,
                self_gtc=self_gtc,
            ))
        if mismatch:
            print(f"  [warn] mention order mismatch on image {image_id} -- partial results kept")

        with open(args.out, "w") as f:
            json.dump(out_records, f, indent=2)

    if gtc_self_checks:
        import numpy as np
        print(f"\nSelf-check GTC L20: n={len(gtc_self_checks)}, "
              f"mean abs diff={np.mean(gtc_self_checks):.4f}, max={np.max(gtc_self_checks):.4f}")
    if align_self_checks:
        import numpy as np
        print(f"Self-check align L8: n={len(align_self_checks)}, "
              f"mean abs diff={np.mean(align_self_checks):.4f}, max={np.max(align_self_checks):.4f}")
    print(f"\nwrote {len(out_records)} records to {args.out}")


if __name__ == "__main__":
    main()
