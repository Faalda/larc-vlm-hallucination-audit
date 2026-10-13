"""Cache LLaVA's per-patch L20 attention for every mention (teacher-forced on the saved captions).

No re-generation: the captions are fixed, so this only re-extracts the attention rows that were
previously reduced to the scalar GTC. Once cached, any ground-truth perturbation can be scored offline.

Usage: python scripts/cache_l20_attention.py --image-dir /path/to/COCO_val2014 [--device cuda:0]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import bbox_to_patch_indices, compute_gtc, token_char_spans, token_index_for_char

ROOT = Path(__file__).resolve().parents[1]
L_LATE = 20

try:  # torch>=2.14 routes some ops through triton, which needs a C compiler this host lacks
    from torch._native import registry as _native
    _native.deregister_op_overrides(disable_dsl_names=["triton"])
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "l20_attention_cache.npz"))
    args = ap.parse_args()

    records = json.load(open(ROOT / "results" / "pilot_records_caption_corrected.json"))
    by_image = defaultdict(list)
    for i, r in enumerate(records):
        by_image[r["image_id"]].append(i)
    meta = {m["image_id"]: m for f in ["pilot_index.json", "pilot_index_extra.json"]
            for m in json.load(open(ROOT / "data" / f))}

    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager").to(args.device).eval()
    tok = processor.tokenizer
    img_tok = model.config.image_token_index
    patch = model.config.vision_config.patch_size
    shortest = processor.image_processor.size.get("shortest_edge") or processor.image_processor.size.get("height")
    crop = processor.image_processor.crop_size
    grid = crop["height"] // patch

    attn = np.full((len(records), grid * grid), np.nan, dtype=np.float32)
    diffs = []
    for n, (iid, idxs) in enumerate(by_image.items()):
        caption = records[idxs[0]]["caption"]
        image = Image.open(Path(args.image_dir) / meta[iid]["file_name"]).convert("RGB")
        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        pin = processor(images=image, text=prompt, return_tensors="pt")
        cap_ids = tok(caption, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([pin["input_ids"][0], cap_ids]).unsqueeze(0).to(args.device)
        pos = (ids[0] == img_tok).nonzero(as_tuple=True)[0]
        if pos.numel() != grid * grid:
            continue
        with torch.no_grad():
            out = model(input_ids=ids, pixel_values=pin["pixel_values"].to(args.device, torch.float16),
                        output_attentions=True)
        spans, _ = token_char_spans(tok, cap_ids)
        found = find_object_mentions(caption)
        if len(found) != len(idxs):
            continue
        all_gt = sorted({p for o in meta[iid]["objects"]
                         for p in bbox_to_patch_indices(o["bbox"], meta[iid]["width"], meta[iid]["height"],
                                                        shortest, crop, patch)})
        for i, (cat, _, end, word) in zip(idxs, found):
            if cat != records[i]["category"] or word.lower() != records[i]["word"].lower():
                break
            t = token_index_for_char(spans, end)
            if t is None:
                continue
            q = pin["input_ids"].shape[1] - 1 + t
            row = out.attentions[L_LATE - 1][0][:, q, :].mean(dim=0)[pos]
            a = (row / (row.sum() + 1e-8)).float().cpu()
            attn[i] = a.numpy()
            saved = records[i]["by_layer"][str(L_LATE)]["gtc_any_object"]
            diffs.append(abs(compute_gtc(a, all_gt, grid) - saved))
        if n % 25 == 0:
            print(f"[{n + 1}/{len(by_image)}] cached={int((~np.isnan(attn[:, 0])).sum())} "
                  f"self-check mean|d|={np.mean(diffs):.4f} max={np.max(diffs):.4f}", flush=True)
    np.savez_compressed(args.out, attn=attn)
    print(f"saved {int((~np.isnan(attn[:, 0])).sum())}/{len(records)} rows; "
          f"self-check mean|d|={np.mean(diffs):.5f} max={np.max(diffs):.5f}")


if __name__ == "__main__":
    main()
