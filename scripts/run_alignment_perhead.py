"""Compute alignment from per-head value projections and output projection.

Current align_cos (llava_geometry / run_pilot.py) uses v_t = sum_j p_j * x_j, where
p_j is HEAD-AVERAGED attention and x_j is the OUTPUT residual-stream state of layer L
for image token j -- an approximation, not what layer L's attention mechanism itself
computes and adds to the stream.

This script reconstructs the real quantity: for layer L, per head h,
    V_j^h = v_proj_L(input_layernorm_L(x_j^{L-1}))[head h]      (x^{L-1} = INPUT to layer L)
    retrieved^h = sum_j p_j^h * V_j^h                            (p^h = layer L's own,
                                                                    per-head attention,
                                                                    renormalized over
                                                                    image patches)
    retrieved = o_proj_L(concat_h retrieved^h)
    align_perhead = cos(h_prev, retrieved)
where h_prev is unchanged (output of layer L at the query position, matching the
existing align_cos convention exactly, so only the "what was retrieved" side changes).

Teacher-forced (not generated): reuses the EXACT captions/mentions/labels already in
results/pilot_records_caption_corrected.json (Table 1's population) via a single
forward pass per image (prompt + saved caption, output_attentions/output_hidden_states
=True) -- cheaper than autoregressive generation and guarantees the same mentions are
scored, no regeneration drift. Self-checks the recomputed OLD align_cos against the
saved value before trusting the new one, per this project's standing discipline.

Scoped to a subset of images (--n-images) for tractability; not the full 1649-mention
population.

Usage:
    python run_alignment_perhead.py --n-images 150 --layer 8
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import token_char_spans, token_index_for_char

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_local_paths():
    paths = {}
    for f in ["pilot_index.json", "pilot_index_extra.json"]:
        for r in json.load(open(os.path.join(DATA_DIR, f))):
            paths[r["image_id"]] = r["local_path"]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n-images", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "alignment_perhead_records.json"))
    args = ap.parse_args()

    records = json.load(open(os.path.join(RESULTS_DIR, "pilot_records_caption_corrected.json")))
    by_image = {}
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)
    image_ids = sorted(by_image.keys())[: args.n_images]
    local_paths = load_local_paths()

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    tokenizer = processor.tokenizer
    image_token_index = model.config.image_token_index
    lm = model.model.language_model
    layer_module = lm.layers[args.layer - 1]  # 1-indexed L -> 0-indexed module list
    v_proj, o_proj = layer_module.self_attn.v_proj, layer_module.self_attn.o_proj
    input_ln = layer_module.input_layernorm
    n_heads = lm.config.num_attention_heads
    head_dim = layer_module.self_attn.head_dim

    out_records = []
    self_check_diffs = []
    for i, image_id in enumerate(image_ids):
        if image_id not in local_paths:
            continue
        mentions = by_image[image_id]
        caption_text = mentions[0]["caption"]
        image = Image.open(local_paths[image_id]).convert("RGB")
        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

        prompt_inputs = processor(images=image, text=prompt, return_tensors="pt")
        prompt_ids = prompt_inputs["input_ids"][0]
        pixel_values = prompt_inputs["pixel_values"].to(args.device, dtype)
        caption_ids = tokenizer(caption_text, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]
        full_ids = torch.cat([prompt_ids, caption_ids]).unsqueeze(0).to(args.device)
        prompt_len = prompt_ids.shape[0]

        image_token_positions = (full_ids[0] == image_token_index).nonzero(as_tuple=True)[0]
        if image_token_positions.numel() == 0:
            print(f"  [warn] no image tokens found for image {image_id}, skipping")
            continue

        with torch.no_grad():
            out = model(input_ids=full_ids, pixel_values=pixel_values,
                        output_attentions=True, output_hidden_states=True)

        print(f"[{i + 1}/{len(image_ids)}] image {image_id} "
              f"({len(mentions)} mentions, seq_len={full_ids.shape[1]})", flush=True)

        attn_L = out.attentions[args.layer - 1][0]        # [heads, seq, seq]
        hidden_Lminus1 = out.hidden_states[args.layer - 1][0]  # input to layer L, [seq, hidden]
        hidden_L = out.hidden_states[args.layer][0]            # output of layer L, [seq, hidden]

        x_img = hidden_Lminus1[image_token_positions]          # [n_img, hidden]
        with torch.no_grad():
            x_img_normed = input_ln(x_img)
            V_img = v_proj(x_img_normed)                       # [n_img, hidden]
        n_img = image_token_positions.shape[0]
        V_img_heads = V_img.view(n_img, n_heads, head_dim)      # [n_img, heads, head_dim]

        # Ordered mention spans, re-extracted the same way the saved records were built
        # (NOT str.find(word), which collapses repeated words in one caption onto the
        # same, wrong, first-occurrence position -- caught via a duplicate-position audit
        # on an earlier, buggy run of this script; verified count/order match the saved
        # `mentions` list exactly on all 150 images before relying on this).
        found_spans = find_object_mentions(caption_text)
        if len(found_spans) != len(mentions):
            print(f"  [warn] mention count mismatch on image {image_id} "
                  f"(saved={len(mentions)}, re-extracted={len(found_spans)}) -- skipping image")
            continue
        spans, _ = token_char_spans(tokenizer, caption_ids)

        for m, (fcat, fstart, fend, fword) in zip(mentions, found_spans):
            if fcat != m["category"] or fword.lower() != m["word"].lower():
                print(f"  [warn] mention order mismatch on image {image_id} -- skipping image")
                break
            tok_idx = token_index_for_char(spans, fend)
            if tok_idx is None:
                continue
            q_pos = prompt_len - 1 + tok_idx  # position whose logits predict token tok_idx

            attn_row = attn_L[:, q_pos, :]                      # [heads, seq]
            attn_over_image = attn_row[:, image_token_positions]  # [heads, n_img]
            p_head = attn_over_image / (attn_over_image.sum(dim=-1, keepdim=True) + 1e-8)

            # OLD-style (head-averaged) recompute, for a self-check against the saved value
            p_avg = p_head.mean(dim=0)                          # [n_img]
            v_t_old = (p_avg.unsqueeze(-1).to(hidden_L.dtype) * hidden_L[image_token_positions]).sum(dim=0)
            h_prev = hidden_L[q_pos]
            align_old_recomputed = F.cosine_similarity(
                h_prev.float().unsqueeze(0), v_t_old.float().unsqueeze(0)).item()

            # NEW per-head value/output-projection alignment
            with torch.no_grad():
                retrieved_heads = torch.einsum("hn,nhd->hd", p_head.to(V_img_heads.dtype), V_img_heads)
                retrieved = o_proj(retrieved_heads.reshape(-1))
            align_perhead = F.cosine_similarity(
                h_prev.float().unsqueeze(0), retrieved.float().unsqueeze(0)).item()

            saved_align = m["by_layer"].get(str(args.layer), {}).get("align_cos")
            if saved_align is not None:
                self_check_diffs.append(abs(align_old_recomputed - saved_align))

            out_records.append(dict(
                image_id=image_id, category=m["category"], word=m["word"],
                grounded=m["grounded"], layer=args.layer,
                align_cos_saved=saved_align,
                align_cos_recomputed=align_old_recomputed,
                align_perhead=align_perhead,
            ))

        with open(args.out, "w") as f:
            json.dump(out_records, f, indent=2)

    if self_check_diffs:
        import numpy as np
        print(f"\nSelf-check: recomputed vs saved align_cos, n={len(self_check_diffs)}, "
              f"mean abs diff={np.mean(self_check_diffs):.4f}, "
              f"max={np.max(self_check_diffs):.4f}")
    print(f"\nwrote {len(out_records)} records to {args.out}")


if __name__ == "__main__":
    main()
