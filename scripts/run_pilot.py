"""Grounding-geometry pilot: does routing (GTC) / alignment (attended-content
vs. pre-generation hidden state cosine) / prior-override (logp with vs.
without image) separate grounded from hallucinated object mentions in a real
VLM's captions, before committing to the full diagnostic-then-intervention
research program?

v2 adds two follow-ups to the first pilot's result (GTC alone: AUC 0.69;
align_cos and prior_delta individually weak, ~0.52-0.55):
  1. Layer sweep for gtc_any_object / align_cos -- v1 fixed both to the last
     layer; hidden_states/attentions already contain every layer per
     generation step, so sweeping costs zero extra forward passes.
  2. An embedding-level null ("zero_embed"): instead of routing a blank gray
     image through the vision tower (v1's null, which still produces some
     arbitrary nonzero, out-of-distribution features), directly zero out the
     projected image-feature vectors before they're spliced into the LLM's
     input embeddings. This is the literal "decoder receives no visual
     information" condition. Computed alongside the v1 gray-image null on
     the same mentions for a direct comparison, not a replacement.

Usage:
    python run_pilot.py --limit 3          # smoke test
    python run_pilot.py                    # full pilot over data/pilot_index.json
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
from transformers import AutoProcessor, LlavaForConditionalGeneration

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
# Coarse 9-point sweep plus fine-grained neighbors around the two claimed
# peaks (L8 for alignment, L20 for GTC) for a real sensitivity check --
# free to add since hidden_states/attentions already contain every layer
# per generation step, no extra forward passes.
LAYER_SWEEP = sorted(set([1, 4, 8, 12, 16, 20, 24, 28, 32] + [6, 7, 9, 10] + [18, 19, 21, 22]))


def build_gray_null_logits(model, gen_out, processor, image, device, dtype):
    null_image = Image.new("RGB", image.size, (127, 127, 127))
    null_pixel_values = processor.image_processor(null_image, return_tensors="pt")["pixel_values"]
    null_pixel_values = null_pixel_values.to(device, dtype)
    with torch.no_grad():
        null_out = model(input_ids=gen_out.sequences, pixel_values=null_pixel_values)
    return null_out.logits[0]


def build_zero_embed_null(model, gen_out, real_pixel_values, device):
    """Zero the projected image-feature vectors before they're spliced into the
    LLM's input embeddings (bypasses the vision tower entirely for the null
    branch). Also returns the reconstructed 'real' embeds so callers can
    self-check this manual path against the normal input_ids+pixel_values path.
    """
    full_ids = gen_out.sequences
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(full_ids)
        image_features = model.get_image_features(
            pixel_values=real_pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            return_dict=True,
        ).pooler_output
        image_features = torch.cat(list(image_features), dim=0).to(text_embeds.dtype)
        placeholder_mask = model.model.get_placeholder_mask(
            full_ids, inputs_embeds=text_embeds, image_features=image_features)

        embeds_real = text_embeds.masked_scatter(placeholder_mask, image_features)
        embeds_null = text_embeds.masked_scatter(placeholder_mask, torch.zeros_like(image_features))
        attn_mask = torch.ones(full_ids.shape, dtype=torch.long, device=device)

        real_check_out = model(inputs_embeds=embeds_real, attention_mask=attn_mask)
        null_out = model(inputs_embeds=embeds_null, attention_mask=attn_mask)
    return real_check_out.logits[0], null_out.logits[0]


def run_one_image(model, processor, device, dtype, image_token_index,
                   grid_size, shortest_edge, crop_size, patch_size, record, max_new_tokens):
    image = Image.open(record["local_path"]).convert("RGB")
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_positions = (inputs["input_ids"][0] == image_token_index).nonzero(as_tuple=True)[0]
    if image_token_positions.numel() != grid_size * grid_size:
        print(f"  [warn] expected {grid_size * grid_size} image tokens, "
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

    # Self-check #1: a teacher-forced forward pass with the REAL image over the
    # full sequence should reproduce the same greedy argmax tokens at every
    # position -- validates the step/position indexing used below.
    with torch.no_grad():
        real_check = model(input_ids=gen_out.sequences, pixel_values=inputs["pixel_values"])
    check_logits = real_check.logits[0, prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate = (check_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate < 0.95:
        print(f"  [warn] indexing self-check match rate {match_rate:.2f} on image "
              f"{record['image_id']} -- treat this image's alignment/prior numbers with caution")

    null_logits_gray = build_gray_null_logits(model, gen_out, processor, image, device, dtype)

    # Self-check #2: the manually-reconstructed embeds_real forward pass should
    # match the normal input_ids+pixel_values forward pass -- validates that
    # get_image_features/get_placeholder_mask are being called with the right
    # config values before trusting the zero_embed null built the same way.
    real_check2_logits, null_logits_zero = build_zero_embed_null(
        model, gen_out, inputs["pixel_values"], device)
    check2_logits = real_check2_logits[prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate2 = (check2_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate2 < 0.95:
        print(f"  [warn] zero_embed reconstruction match rate {match_rate2:.2f} on image "
              f"{record['image_id']} -- zero_embed null numbers for this image are suspect")

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
            align_cos = F.cosine_similarity(h_prev.float().unsqueeze(0), v_t.float().unsqueeze(0)).item()
            by_layer[L] = dict(gtc_any_object=gtc_any_object, align_cos=align_cos)

            if L == LAYER_SWEEP[-1]:
                image_mass_total = attn_over_image.sum().item()
                max_attn = attn_over_image.max().item()
                entropy = -(attn_norm * torch.log(attn_norm + 1e-8)).sum().item()
                gtc_same_cat = compute_gtc(attn_norm, sorted(gt_by_category.get(category, [])), grid_size) \
                    if grounded else None

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
    print(f"\n{len(records)} object mentions extracted "
          f"({sum(r['grounded'] for r in records)} grounded, "
          f"{sum(not r['grounded'] for r in records)} hallucinated)")

    mass = np.array([r["image_mass_total"] for r in records])
    print(f"\n[sink check] attention mass on image tokens: "
          f"mean={mass.mean():.3f} median={np.median(mass):.3f} "
          f"(near 0 => attention barely reaches image tokens at all -- "
          f"stop and investigate before trusting anything below)")

    n_grounded = sum(r["grounded"] for r in records)
    n_halluc = sum(not r["grounded"] for r in records)
    if n_grounded < 5 or n_halluc < 5:
        print(f"\n[stop] too few examples in one class (grounded={n_grounded}, "
              f"hallucinated={n_halluc}) for a meaningful AUC -- collect more images "
              f"before trusting the numbers below")
        return
    y = np.array([1 if r["grounded"] else 0 for r in records])
    n_splits = min(5, n_grounded, n_halluc)
    if n_splits < 2:
        print("\n[stop] cannot cross-validate with this few examples per class")
        return
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    rng = np.random.RandomState(0)

    def bootstrap_ci(y_true, probs, n_boot=2000):
        n = len(y_true)
        boot_aucs = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_true[idx], probs[idx]))
        lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
        return lo, hi

    def auc_for(arr, name, with_ci=False):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # StandardScaler is required here: without it, features with larger
        # natural scales (e.g. GTC) dominate smaller-scale ones (e.g. max
        # attention weight) under LogisticRegression's default L2 penalty,
        # silently deflating real signal in multi-feature combinations. See
        # the paper's Discussion ("Three corrections") and analyze_final.py.
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        probs = cross_val_predict(pipe, arr, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)
        if with_ci:
            lo, hi = bootstrap_ci(y, probs)
            print(f"  AUC[{name}] = {auc:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
        else:
            print(f"  AUC[{name}] = {auc:.3f}")
        return auc

    print("\n[baselines: single trivial signal]")
    auc_for([r["max_attn"] for r in records], "max_attn alone", with_ci=True)
    auc_for([r["entropy"] for r in records], "entropy alone", with_ci=True)

    print("\n[layer sweep] gtc_any_object per layer")
    best_gtc = max(LAYER_SWEEP, key=lambda L: auc_for(
        [r["by_layer"][str(L)]["gtc_any_object"] for r in records], f"gtc L={L}"))
    auc_for([r["by_layer"][str(best_gtc)]["gtc_any_object"] for r in records],
            f"gtc L={best_gtc} (best layer, with CI)", with_ci=True)
    print("\n[layer sweep] align_cos per layer")
    best_align = max(LAYER_SWEEP, key=lambda L: auc_for(
        [r["by_layer"][str(L)]["align_cos"] for r in records], f"align_cos L={L}"))

    print("\n[prior-override null condition comparison]")
    auc_for([r["prior_delta_gray"] for r in records], "prior_delta (gray-image null, v1)", with_ci=True)
    auc_for([r["prior_delta_zero"] for r in records], "prior_delta (zero-embed null, v2)", with_ci=True)

    print(f"\n[best combination found: gtc@L{best_gtc}, align_cos@L{best_align}, best prior null]")
    gtc_best = np.array([r["by_layer"][str(best_gtc)]["gtc_any_object"] for r in records])
    align_best = np.array([r["by_layer"][str(best_align)]["align_cos"] for r in records])
    prior_gray = np.array([r["prior_delta_gray"] for r in records])
    prior_zero = np.array([r["prior_delta_zero"] for r in records])
    X = np.stack([gtc_best, align_best, prior_gray, prior_zero], axis=1)
    print("  feature correlation matrix [gtc_best, align_best, prior_gray, prior_zero]")
    print(np.corrcoef(X.T).round(2))
    auc_for(X, "gtc_best + align_best + both priors (combined)", with_ci=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(DATA_DIR, "pilot_index.json"))
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="only process first N images (smoke test)")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "pilot_records_v4_sensitivity.json"))
    args = ap.parse_args()

    with open(args.index) as f:
        index = json.load(f)
    if args.limit:
        index = index[: args.limit]

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    ip = processor.image_processor
    shortest_edge = ip.size.get("shortest_edge") or ip.size.get("height")
    crop_size = ip.crop_size
    patch_size = model.config.vision_config.patch_size
    grid_size = crop_size["height"] // patch_size
    image_token_index = model.config.image_token_index
    print(f"grid_size={grid_size}x{grid_size}, patch_size={patch_size}, "
          f"shortest_edge={shortest_edge}, image_token_index={image_token_index}")

    all_records = []
    for i, record in enumerate(index):
        print(f"[{i + 1}/{len(index)}] image {record['image_id']}")
        try:
            recs = run_one_image(model, processor, args.device, dtype, image_token_index,
                                  grid_size, shortest_edge, crop_size, patch_size,
                                  record, args.max_new_tokens)
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
