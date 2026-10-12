"""Second-VLM replication: does GTC / alignment survive on an architecture
with genuine Q-Former cross-attention (InstructBLIP-Vicuna-7B) rather than
LLaVA's fused self-attention?

Reuses the generic LLM-side helpers from llava_geometry.py (hidden_states/
attentions tuple conventions are identical -- same Vicuna-7B backbone, 32
layers). New pieces, in instructblip_geometry.py: composing the LLM's
attention over the 32 query tokens with each query's own Q-Former
cross-attention distribution over image patches (to get a per-word patch
distribution for GTC, since the 32 query tokens have no fixed spatial
position of their own), and the stretch-resize (no-crop) bbox transform
InstructBLIP's image processor uses instead of CLIP's shortest-edge+crop.

Usage:
    python run_instructblip.py --limit 3
    python run_instructblip.py
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from larc_audit.coco_categories import find_object_mentions
from larc_audit.instructblip_geometry import (
    bbox_to_patch_indices_stretch,
    compose_patch_attention,
    get_qformer_patch_distributions,
)
from larc_audit.llava_geometry import (
    compute_gtc,
    get_attn_row,
    get_hidden_prev,
    get_image_hidden,
    token_char_spans,
    token_index_for_char,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
LAYER_SWEEP = [1, 4, 8, 12, 16, 20, 24, 28, 32]
RESIZE_W = RESIZE_H = 224
PATCH_SIZE = 14
GRID_SIZE = RESIZE_W // PATCH_SIZE  # 16


def get_null_logits(model, full_ids, qformer_input_ids, qformer_attention_mask, device):
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(full_ids)
        attn_mask = torch.ones(full_ids.shape, dtype=torch.long, device=device)
        placeholder_mask = model.get_placeholder_mask(full_ids, inputs_embeds=text_embeds)
        null_query_embeds = torch.zeros(1, model.config.num_query_tokens,
                                         text_embeds.shape[-1], dtype=text_embeds.dtype, device=device)
        embeds_null = text_embeds.masked_scatter(placeholder_mask, null_query_embeds)
        null_out = model.language_model(inputs_embeds=embeds_null, attention_mask=attn_mask)
    return null_out[0][0]


def run_one_image(model, processor, device, dtype, image_token_id, record, max_new_tokens):
    image = Image.open(record["local_path"]).convert("RGB")
    prompt = "Describe this image in detail."
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    if image_token_positions.numel() != model.config.num_query_tokens:
        print(f"  [warn] expected {model.config.num_query_tokens} query tokens, "
              f"found {image_token_positions.numel()} -- skipping image {record['image_id']}")
        return []

    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_attentions=True, output_hidden_states=True, output_scores=True,
            return_dict_in_generate=True,
        )
    generated_ids = gen_out.sequences[0, prompt_len:]
    tokenizer = processor.tokenizer
    spans, caption = token_char_spans(tokenizer, generated_ids)
    mentions = find_object_mentions(caption)
    if not mentions:
        return []

    # Self-check #1: teacher-forced forward with the real image reproduces
    # the same greedy tokens as generation.
    with torch.no_grad():
        real_check = model(input_ids=gen_out.sequences, pixel_values=inputs["pixel_values"],
                            qformer_input_ids=inputs["qformer_input_ids"],
                            qformer_attention_mask=inputs.get("qformer_attention_mask"))
    check_logits = real_check.logits[0, prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate = (check_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate < 0.95:
        print(f"  [warn] indexing self-check match rate {match_rate:.2f} on image "
              f"{record['image_id']} -- skipping")
        return []

    # Self-check #2: the null-logits construction, when NOT zeroing the query
    # embeddings (i.e. using the real ones), should reproduce the same logits.
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(gen_out.sequences)
        real_feats = model.get_image_features(
            pixel_values=inputs["pixel_values"], qformer_input_ids=inputs["qformer_input_ids"],
            qformer_attention_mask=inputs.get("qformer_attention_mask"), output_attentions=True,
            return_dict=True,
        )
        placeholder_mask = model.get_placeholder_mask(gen_out.sequences, inputs_embeds=text_embeds)
        embeds_real = text_embeds.masked_scatter(
            placeholder_mask, real_feats.pooler_output[0].to(text_embeds.dtype))
        attn_mask_full = torch.ones(gen_out.sequences.shape, dtype=torch.long, device=device)
        real_check2 = model.language_model(inputs_embeds=embeds_real, attention_mask=attn_mask_full)
    check2_logits = real_check2[0][0, prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate2 = (check2_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate2 < 0.95:
        print(f"  [warn] reconstruction self-check match rate {match_rate2:.2f} on image "
              f"{record['image_id']} -- skipping")
        return []

    null_logits = get_null_logits(model, gen_out.sequences, inputs["qformer_input_ids"],
                                   inputs.get("qformer_attention_mask"), device)

    qformer_patch_dist = get_qformer_patch_distributions(real_feats.qformer_outputs.cross_attentions)
    image_hidden_by_layer = {
        L: get_image_hidden(gen_out.hidden_states, L, image_token_positions) for L in LAYER_SWEEP
    }

    gt_by_category = {}
    all_gt_patches = set()
    for obj in record["objects"]:
        idxs = bbox_to_patch_indices_stretch(obj["bbox"], record["width"], record["height"],
                                              RESIZE_W, RESIZE_H, PATCH_SIZE)
        gt_by_category.setdefault(obj["category"], set()).update(idxs)
        all_gt_patches.update(idxs)
    all_gt_patches = sorted(all_gt_patches)

    out_records = []
    for category, char_start, char_end, word in mentions:
        step = token_index_for_char(spans, char_end)
        if step is None or step >= len(generated_ids):
            continue
        grounded = category in gt_by_category

        by_layer = {}
        for L in LAYER_SWEEP:
            attn_row = get_attn_row(gen_out.attentions, step, L - 1)
            attn_over_queries = attn_row[image_token_positions]
            attn_norm = attn_over_queries / (attn_over_queries.sum() + 1e-8)

            composite_patch_attn = compose_patch_attention(attn_norm, qformer_patch_dist)
            gtc_any_object = compute_gtc(composite_patch_attn, all_gt_patches, GRID_SIZE)

            h_prev = get_hidden_prev(gen_out.hidden_states, step, L)
            v_t = (attn_norm.unsqueeze(-1).to(image_hidden_by_layer[L].dtype)
                   * image_hidden_by_layer[L]).sum(dim=0)
            align_cos = F.cosine_similarity(h_prev.float().unsqueeze(0), v_t.float().unsqueeze(0)).item()
            by_layer[L] = dict(gtc_any_object=gtc_any_object, align_cos=align_cos)

            if L == LAYER_SWEEP[-1]:
                image_mass_total = attn_over_queries.sum().item()
                max_attn = attn_over_queries.max().item()
                entropy = -(attn_norm * torch.log(attn_norm + 1e-8)).sum().item()

        token_id = generated_ids[step].item()
        logp_with = F.log_softmax(gen_out.scores[step][0].float(), dim=-1)[token_id].item()
        pos_in_seq = prompt_len + step - 1
        logp_null = F.log_softmax(null_logits[pos_in_seq].float(), dim=-1)[token_id].item()

        out_records.append(dict(
            image_id=record["image_id"], category=category, word=word, grounded=bool(grounded),
            image_mass_total=image_mass_total, max_attn=max_attn, entropy=entropy,
            prior_delta=logp_with - logp_null, caption=caption,
            by_layer={str(L): v for L, v in by_layer.items()},
        ))
    return out_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(DATA_DIR, "pilot_index.json"))
    ap.add_argument("--model", default="Salesforce/instructblip-vicuna-7b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "instructblip_records.json"))
    args = ap.parse_args()

    with open(args.index) as f:
        index = json.load(f)
    if args.limit:
        index = index[: args.limit]

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ...")
    processor = InstructBlipProcessor.from_pretrained(args.model)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    image_token_id = model.config.image_token_id
    print(f"grid_size={GRID_SIZE}x{GRID_SIZE}, patch_size={PATCH_SIZE}, "
          f"num_query_tokens={model.config.num_query_tokens}, image_token_id={image_token_id}")

    all_records = []
    for i, record in enumerate(index):
        print(f"[{i + 1}/{len(index)}] image {record['image_id']}")
        try:
            recs = run_one_image(model, processor, args.device, dtype, image_token_id,
                                  record, args.max_new_tokens)
            all_records.extend(recs)
        except Exception as e:
            print(f"  [error] {e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nwrote {len(all_records)} records to {args.out}")


if __name__ == "__main__":
    main()
