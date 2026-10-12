"""Matched-backbone ablation: Intel/llava-gemma-2b (CLIP-336 + Gemma-2B/18L).

Same CLIP vision tower as LLaVA-1.5-7B (24L, patch_size=14, 336px -> 576 patches)
but Gemma-2B/18L as the LLM, matching PaliGemma's LLM depth/size. This lets us
ask whether GTC's weakness on PaliGemma (AUC 0.573) is driven by the SigLIP vision
encoder or by the Gemma-2B LLM -- if CLIP+Gemma-2B also shows weak GTC, the LLM is
the more likely explanation; if it holds, SigLIP is.

AutoProcessor for this checkpoint doesn't auto-expand the <image> token correctly
(patch_size not set in its preprocessor_config), so we tokenize with the Gemma
tokenizer's chat template, then manually expand the single <image> id to 576 copies.
The rest of the pipeline (bbox_to_patch_indices, compute_gtc, get_attn_row,
get_hidden_prev, get_image_hidden) is reused from llava_geometry.py unchanged --
same CLIP tower means same coordinate geometry.

Usage:
    python run_llava_gemma.py --limit 3       # smoke test (3 images)
    python run_llava_gemma.py                 # full 300-image run
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import (
    bbox_to_patch_indices,
    compute_gtc,
    get_attn_row,
    get_hidden_prev,
    get_image_hidden,
    token_char_spans,
    token_index_for_char,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
CHECKPOINT = "Intel/llava-gemma-2b"

# Same sweep as run_pilot.py so results are directly comparable.
LAYER_SWEEP = sorted(set([1, 4, 8, 12, 16, 18] + [6, 7, 9, 10] + [14, 15, 17]))


def expand_image_tokens(base_ids, image_token_index, n_img_tokens, device):
    """Replace the single <image> token in base_ids with n_img_tokens copies."""
    img_positions = (base_ids == image_token_index).nonzero(as_tuple=True)[0]
    assert img_positions.numel() == 1, f"expected 1 <image> token, found {img_positions.numel()}"
    pos = img_positions[0].item()
    expanded = torch.cat([
        base_ids[:pos],
        torch.full((n_img_tokens,), image_token_index, dtype=torch.long),
        base_ids[pos + 1:],
    ]).unsqueeze(0).to(device)
    return expanded


def build_gray_null_logits(model, full_seq, pixel_values, ip, image, device, dtype):
    null_image = Image.new("RGB", image.size, (127, 127, 127))
    null_pv = ip(images=null_image, return_tensors="pt")["pixel_values"].to(device, dtype)
    attn_mask = torch.ones_like(full_seq)
    with torch.no_grad():
        out = model(input_ids=full_seq, attention_mask=attn_mask, pixel_values=null_pv)
    return out.logits[0]


def build_zero_embed_null(model, full_seq, pixel_values, device):
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(full_seq)
        image_features = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            return_dict=True,
        ).pooler_output
        image_features = torch.cat(list(image_features), dim=0).to(text_embeds.dtype)
        placeholder_mask = model.model.get_placeholder_mask(
            full_seq, inputs_embeds=text_embeds, image_features=image_features)
        embeds_real = text_embeds.masked_scatter(placeholder_mask, image_features)
        embeds_null = text_embeds.masked_scatter(placeholder_mask, torch.zeros_like(image_features))
        attn_mask = torch.ones(full_seq.shape, dtype=torch.long, device=device)
        real_check = model(inputs_embeds=embeds_real, attention_mask=attn_mask)
        null_out = model(inputs_embeds=embeds_null, attention_mask=attn_mask)
    return real_check.logits[0], null_out.logits[0]


def run_one_image(model, tok, ip, device, dtype, image_token_index,
                   grid_size, shortest_edge, crop_size, patch_size,
                   base_prompt_ids, record, max_new_tokens):
    image = Image.open(record["local_path"]).convert("RGB")
    pixel_values = ip(images=image, return_tensors="pt")["pixel_values"].to(device, dtype)

    n_img = grid_size * grid_size
    expanded = expand_image_tokens(base_prompt_ids, image_token_index, n_img, device)
    attn_mask = torch.ones_like(expanded)
    prompt_len = expanded.shape[1]

    image_token_positions = (expanded[0] == image_token_index).nonzero(as_tuple=True)[0]
    if image_token_positions.numel() != n_img:
        print(f"  [warn] expected {n_img} image tokens, found {image_token_positions.numel()} "
              f"-- skipping image {record['image_id']}")
        return []

    with torch.no_grad():
        gen_out = model.generate(
            input_ids=expanded, attention_mask=attn_mask, pixel_values=pixel_values,
            max_new_tokens=max_new_tokens, do_sample=False,
            output_attentions=True, output_hidden_states=True, output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = gen_out.sequences[0, prompt_len:]
    spans, caption = token_char_spans(tok, generated_ids)
    mentions = find_object_mentions(caption)
    if not mentions:
        return []

    # Self-check #1: teacher-forced forward with real image should reproduce argmax tokens.
    full_seq = gen_out.sequences
    with torch.no_grad():
        real_check = model(input_ids=full_seq,
                           attention_mask=torch.ones_like(full_seq),
                           pixel_values=pixel_values)
    check_logits = real_check.logits[0, prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate = (check_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate < 0.95:
        print(f"  [warn] indexing self-check match rate {match_rate:.2f} on image "
              f"{record['image_id']} -- skipping")
        return []

    null_logits_gray = build_gray_null_logits(model, full_seq, pixel_values,
                                               ip, image, device, dtype)

    # Self-check #2: zero-embed reconstruction should match real forward pass.
    real_check2_logits, null_logits_zero = build_zero_embed_null(
        model, full_seq, pixel_values, device)
    check2_logits = real_check2_logits[prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate2 = (check2_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate2 < 0.95:
        print(f"  [warn] zero_embed reconstruction match rate {match_rate2:.2f} on image "
              f"{record['image_id']} -- zero_embed null numbers suspect")

    image_hidden_by_layer = {
        L: get_image_hidden(gen_out.hidden_states, L, image_token_positions) for L in LAYER_SWEEP
    }

    gt_by_category = {}
    all_gt_patches = set()
    for obj in record["objects"]:
        idxs = bbox_to_patch_indices(obj["bbox"], record["width"], record["height"],
                                      shortest_edge, crop_size, patch_size)
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

            gtc_any_object = compute_gtc(attn_norm, all_gt_patches, grid_size)
            h_prev = get_hidden_prev(gen_out.hidden_states, step, L)
            v_t = (attn_norm.unsqueeze(-1).to(image_hidden_by_layer[L].dtype)
                   * image_hidden_by_layer[L]).sum(dim=0)
            align_cos = F.cosine_similarity(
                h_prev.float().unsqueeze(0), v_t.float().unsqueeze(0)).item()
            by_layer[L] = dict(gtc_any_object=gtc_any_object, align_cos=align_cos)

            if L == LAYER_SWEEP[-1]:
                image_mass_total = attn_over_image.sum().item()
                max_attn = attn_over_image.max().item()
                entropy = -(attn_norm * torch.log(attn_norm + 1e-8)).sum().item()
                gtc_same_cat = (compute_gtc(attn_norm,
                                             sorted(gt_by_category.get(category, [])),
                                             grid_size) if grounded else None)

        token_id = generated_ids[step].item()
        logp_with = F.log_softmax(gen_out.scores[step][0].float(), dim=-1)[token_id].item()
        pos_in_seq = prompt_len + step - 1
        logp_null_gray = F.log_softmax(null_logits_gray[pos_in_seq].float(), dim=-1)[token_id].item()
        logp_null_zero = F.log_softmax(null_logits_zero[pos_in_seq].float(), dim=-1)[token_id].item()

        out_records.append(dict(
            image_id=record["image_id"], category=category, word=word,
            grounded=bool(grounded), gtc_same_category=gtc_same_cat,
            gtc_any_object=by_layer[LAYER_SWEEP[-1]]["gtc_any_object"],
            align_cos=by_layer[LAYER_SWEEP[-1]]["align_cos"],
            prior_delta_gray=logp_with - logp_null_gray,
            prior_delta_zero=logp_with - logp_null_zero,
            image_mass_total=image_mass_total, max_attn=max_attn, entropy=entropy,
            logp_with_image=logp_with, caption=caption,
            by_layer={str(L): v for L, v in by_layer.items()},
        ))
    return out_records


def analyze(records):
    print(f"\n{len(records)} object mentions "
          f"({sum(r['grounded'] for r in records)} grounded, "
          f"{sum(not r['grounded'] for r in records)} hallucinated)")

    mass = np.array([r["image_mass_total"] for r in records])
    print(f"\n[sink check] image-attention mass: mean={mass.mean():.3f} "
          f"median={np.median(mass):.3f}")

    y = np.array([1 if r["grounded"] else 0 for r in records])
    n_g = y.sum(); n_h = len(y) - n_g
    if n_g < 5 or n_h < 5:
        print(f"[stop] too few examples (grounded={n_g}, hallucinated={n_h})")
        return
    cv = StratifiedKFold(n_splits=min(5, int(n_g), int(n_h)),
                          shuffle=True, random_state=0)
    rng = np.random.RandomState(0)

    def bootstrap_ci(y_true, probs, n_boot=2000):
        aucs = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], probs[idx]))
        return np.percentile(aucs, [2.5, 97.5])

    def auc_for(arr, name, with_ci=False):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # StandardScaler required for multi-feature combinations -- see
        # run_pilot.py's auc_for() and the paper's "Three corrections"
        # Discussion paragraph.
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        probs = cross_val_predict(pipe, arr, y, cv=cv,
                                   method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)
        if with_ci:
            lo, hi = bootstrap_ci(y, probs)
            print(f"  AUC[{name}] = {auc:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
        else:
            print(f"  AUC[{name}] = {auc:.3f}")
        return auc, probs

    print()
    auc_for([r["entropy"] for r in records], "entropy", with_ci=True)
    auc_for([r["max_attn"] for r in records], "max_attn", with_ci=True)
    auc_gtc, _ = auc_for([r["gtc_any_object"] for r in records], "gtc_any_object", with_ci=True)
    auc_align, p_align = auc_for([r["align_cos"] for r in records], "align_cos", with_ci=True)
    _, p_gtc = auc_for([r["gtc_any_object"] for r in records], "gtc_any_object")
    combined = np.stack([p_gtc, p_align], axis=1).mean(axis=1)
    auc_comb = roc_auc_score(y, combined)
    lo, hi = bootstrap_ci(y, combined)
    print(f"  AUC[combined] = {auc_comb:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
    auc_for([r["prior_delta_gray"] for r in records], "prior_delta_gray", with_ci=True)
    auc_for([r["prior_delta_zero"] for r in records], "prior_delta_zero", with_ci=True)

    print("\nLayer sweep (GTC | align):")
    for L in LAYER_SWEEP:
        gtcs = [r["by_layer"][str(L)]["gtc_any_object"] for r in records if str(L) in r["by_layer"]]
        alns = [r["by_layer"][str(L)]["align_cos"] for r in records if str(L) in r["by_layer"]]
        if not gtcs:
            continue
        auc_g, _ = auc_for(gtcs, f"gtc_L{L}")
        auc_a, _ = auc_for(alns, f"aln_L{L}")
        print(f"    L{L:2d}: GTC={auc_g:.3f}  align={auc_a:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(DATA_DIR, "pilot_index.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "llava_gemma_records.json"))
    args = ap.parse_args()

    with open(args.index) as f:
        index = json.load(f)
    if args.limit:
        index = index[:args.limit]

    dtype = torch.float16
    print(f"loading {CHECKPOINT} ...")
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    ip = CLIPImageProcessor.from_pretrained(CHECKPOINT)
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device); model.eval()

    image_token_index = model.config.image_token_index
    patch_size = model.config.vision_config.patch_size        # 14
    crop_size = ip.crop_size                                   # {height:336, width:336}
    shortest_edge = ip.size.get("shortest_edge") or ip.size.get("height")  # 336
    grid_size = crop_size["height"] // patch_size              # 24

    # Build the base prompt once (single <image> token) to expand per image.
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": "Describe this image in detail.<image>"}],
        tokenize=False, add_generation_prompt=True,
    )
    base_prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"][0]
    assert (base_prompt_ids == image_token_index).sum().item() == 1, \
        "prompt must contain exactly one <image> token"
    print(f"grid={grid_size}, n_img_tokens={grid_size**2}, "
          f"base_prompt_len={len(base_prompt_ids)}")

    all_records = []
    for i, record in enumerate(index):
        print(f"[{i + 1}/{len(index)}] image {record['image_id']}")
        try:
            recs = run_one_image(
                model, tok, ip, args.device, dtype, image_token_index,
                grid_size, shortest_edge, crop_size, patch_size,
                base_prompt_ids, record, args.max_new_tokens,
            )
            all_records.extend(recs)
        except Exception as e:
            print(f"  [error] {e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nwrote {len(all_records)} records to {args.out}")

    if all_records:
        analyze(all_records)


if __name__ == "__main__":
    main()
