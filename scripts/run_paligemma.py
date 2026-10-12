"""Third-VLM replication: PaliGemma-3B (SigLIP vision encoder + direct patch
fusion into a Gemma-2B language model), a different point from both LLaVA
(fused self-attention, Vicuna-7B/CLIP) and InstructBLIP (Q-Former
cross-attention, Vicuna-7B/CLIP).

NOT an architecture-isolated comparison the way LLaVA-vs-InstructBLIP was:
PaliGemma differs in vision encoder (SigLIP vs CLIP), LLM family and depth
(Gemma-2B/18 layers vs Vicuna-7B/32 layers), and training recipe, all at
once. A different result here cannot be cleanly attributed to the
vision-fusion mechanism alone -- this is a broader robustness check (does
the finding hold on a wholly different model), not a second controlled
manipulation of one variable. Reported as such in the paper.

Mechanistically, PaliGemma is closer to LLaVA than to InstructBLIP: 256
patch tokens (16x16 grid, 224px, patch size 14, no CLS token) are injected
directly into the LLM's own self-attention, so GTC/alignment reuse
llava_geometry.py's helpers unchanged -- no query-to-patch composition
needed (that was specific to InstructBLIP's compressed 32-query Q-Former).
The image processor does a direct stretch-resize with no center crop
(same convention as InstructBLIP's), so bbox_to_patch_indices_stretch is
reused from instructblip_geometry.py rather than duplicated.

Usage:
    python run_paligemma.py --limit 3
    python run_paligemma.py
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from larc_audit.coco_categories import find_object_mentions
from larc_audit.instructblip_geometry import bbox_to_patch_indices_stretch
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
LAYER_SWEEP = [2, 4, 6, 8, 10, 12, 14, 16, 18]  # Gemma-2B has 18 layers, vs 32 for Vicuna-7B
RESIZE_W = RESIZE_H = 224
PATCH_SIZE = 14
GRID_SIZE = RESIZE_W // PATCH_SIZE  # 16


def embed_input_ids(model, input_ids):
    """PaliGemma's image_token_id is intentionally out-of-vocab (avoids a
    real embedding-table collision before the image features overwrite
    those positions via masked_scatter); the reference forward() zeroes
    those ids before embedding lookup, replicated here for our own
    from-scratch embeds construction.
    """
    ids = input_ids.clone()
    if model.config.image_token_id >= model.model.vocab_size:
        ids[input_ids == model.config.image_token_id] = 0
    return model.get_input_embeddings()(ids)


def full_seq_token_type_ids(full_ids, prompt_len, device):
    """PaliGemma's prefix-LM mask needs token_type_ids: 0 for the
    bidirectional prefix (image + prompt), 1 for the causal suffix
    (generated continuation). Built here since we assemble the full
    prompt+generated sequence ourselves rather than letting generate()
    extend it incrementally.
    """
    seq_len = full_ids.shape[1]
    tt = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    tt[0, prompt_len:] = 1
    return tt


def get_null_logits(model, full_ids, prompt_len, device):
    with torch.no_grad():
        text_embeds = embed_input_ids(model, full_ids)
        attn_mask = torch.ones(full_ids.shape, dtype=torch.long, device=device)
        token_type_ids = full_seq_token_type_ids(full_ids, prompt_len, device)
        n_img = (full_ids[0] == model.config.image_token_id).sum().item()
        null_features = torch.zeros(1, n_img, text_embeds.shape[-1],
                                     dtype=text_embeds.dtype, device=device)
        placeholder_mask = model.model.get_placeholder_mask(
            full_ids, inputs_embeds=text_embeds, image_features=null_features)
        embeds_null = text_embeds.masked_scatter(placeholder_mask, null_features)
        null_out = model(inputs_embeds=embeds_null, attention_mask=attn_mask,
                          token_type_ids=token_type_ids, pixel_values=None)
    return null_out.logits[0]


def run_one_image(model, processor, device, dtype, image_token_id, record, max_new_tokens):
    image = Image.open(record["local_path"]).convert("RGB")
    prompt = "Describe this image in detail."
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    if image_token_positions.numel() != GRID_SIZE * GRID_SIZE:
        print(f"  [warn] expected {GRID_SIZE * GRID_SIZE} image tokens, "
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

    full_token_type_ids = full_seq_token_type_ids(gen_out.sequences, prompt_len, device)

    # Self-check #1: teacher-forced forward with the real image reproduces
    # the same greedy tokens as generation.
    with torch.no_grad():
        real_check = model(input_ids=gen_out.sequences, pixel_values=inputs["pixel_values"],
                            token_type_ids=full_token_type_ids)
    check_logits = real_check.logits[0, prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate = (check_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate < 0.95:
        print(f"  [warn] indexing self-check match rate {match_rate:.2f} on image "
              f"{record['image_id']} -- skipping")
        return []

    # Self-check #2: manually reconstructed embeds_real (built the same way
    # as the null, but with real image features) should match the normal
    # forward path's logits.
    with torch.no_grad():
        text_embeds = embed_input_ids(model, gen_out.sequences)
        real_feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = real_feats.pooler_output if hasattr(real_feats, "pooler_output") else real_feats
        placeholder_mask = model.model.get_placeholder_mask(
            gen_out.sequences, inputs_embeds=text_embeds, image_features=image_features)
        embeds_real = text_embeds.masked_scatter(placeholder_mask, image_features.to(text_embeds.dtype))
        attn_mask_full = torch.ones(gen_out.sequences.shape, dtype=torch.long, device=device)
        real_check2 = model(inputs_embeds=embeds_real, attention_mask=attn_mask_full,
                             token_type_ids=full_token_type_ids, pixel_values=None)
    check2_logits = real_check2.logits[0, prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate2 = (check2_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate2 < 0.95:
        print(f"  [warn] reconstruction self-check match rate {match_rate2:.2f} on image "
              f"{record['image_id']} -- skipping")
        return []

    null_logits = get_null_logits(model, gen_out.sequences, prompt_len, device)

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
            attn_over_image = attn_row[image_token_positions]
            attn_norm = attn_over_image / (attn_over_image.sum() + 1e-8)

            gtc_any_object = compute_gtc(attn_norm, all_gt_patches, GRID_SIZE)
            h_prev = get_hidden_prev(gen_out.hidden_states, step, L)
            v_t = (attn_norm.unsqueeze(-1).to(image_hidden_by_layer[L].dtype)
                   * image_hidden_by_layer[L]).sum(dim=0)
            align_cos = F.cosine_similarity(h_prev.float().unsqueeze(0), v_t.float().unsqueeze(0)).item()
            by_layer[L] = dict(gtc_any_object=gtc_any_object, align_cos=align_cos)

            if L == LAYER_SWEEP[-1]:
                image_mass_total = attn_over_image.sum().item()
                max_attn = attn_over_image.max().item()
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
    ap.add_argument("--model", default="google/paligemma-3b-mix-224")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "paligemma_records.json"))
    args = ap.parse_args()

    with open(args.index) as f:
        index = json.load(f)
    if args.limit:
        index = index[: args.limit]

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ...")
    processor = PaliGemmaProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    image_token_id = model.config.image_token_id
    print(f"grid_size={GRID_SIZE}x{GRID_SIZE}, patch_size={PATCH_SIZE}, "
          f"image_token_id={image_token_id}, layers={LAYER_SWEEP}")

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
