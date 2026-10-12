"""Causal grounding-intervention experiment.

The pilot (run_pilot.py) is purely diagnostic: GTC and alignment correlate
with hallucination. This asks a causal question instead: if we control
exactly which image patches the model can see (via the same embedding-level
masking technique already validated for the zero-embed null in run_pilot.py,
generalized from "zero everything" to "zero everything except a chosen
patch subset"), does the model's predicted word change?

Six conditions, all reusing the same masked teacher-forced forward pass:
  1. grounded mention,     restrict to real-evidence patches only -> expect LOW flip rate (control: doesn't break correct generation)
  2. grounded mention,     ablate exactly the real-evidence patches -> expect HIGH flip rate (necessity: was that evidence actually used?)
  3. hallucinated mention, restrict to real-evidence patches only -> expect HIGH flip rate (does forcing real evidence correct the hallucination?)
  4. hallucinated mention, ablate all real-evidence patches       -> expect LOW flip rate (is the hallucination evidence-independent / prior-driven?)
  5. grounded mention,     ablate a random equal-size patch set   -> baseline for (2): is the flip caused by losing THAT evidence, or by any comparable-size disruption?
  6. hallucinated mention, ablate a random equal-size patch set   -> baseline for (4): does an equally sized but content-blind ablation flip the hallucination just as rarely?
Conditions 5/6 isolate whether (2)/(4) reflect the identity of the ablated
region or just its size -- the "gist stabilization" question raised in
Discussion is only a candidate explanation without this control.

"Flip" = the argmax token at that position changes from the original
greedy-decoded token once the image features are masked.

Usage:
    python run_intervention.py --limit 3           # smoke test
    python run_intervention.py                     # full run over data/pilot_index.json
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import bbox_to_patch_indices, token_char_spans, token_index_for_char

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def masked_forward_logits(model, full_ids, real_pixel_values, device, keep_patch_idxs):
    """Teacher-forced forward pass with only `keep_patch_idxs` image patches
    real; all others are zeroed before splicing into the LLM's input
    embeddings (vision tower still runs once on the real image; masking
    happens at the projected-feature level, same mechanism as the zero-embed
    null in run_pilot.py, generalized from all-zero to a partial mask).
    """
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(full_ids)
        image_features = model.get_image_features(
            pixel_values=real_pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            return_dict=True,
        ).pooler_output
        image_features = torch.cat(list(image_features), dim=0).to(text_embeds.dtype)

        mask = torch.zeros(image_features.shape[0], dtype=torch.bool, device=device)
        if keep_patch_idxs:
            mask[list(keep_patch_idxs)] = True
        masked_features = image_features * mask.unsqueeze(-1)

        placeholder_mask = model.model.get_placeholder_mask(
            full_ids, inputs_embeds=text_embeds, image_features=masked_features)
        embeds = text_embeds.masked_scatter(placeholder_mask, masked_features)
        attn_mask = torch.ones(full_ids.shape, dtype=torch.long, device=device)
        out = model(inputs_embeds=embeds, attention_mask=attn_mask)
    return out.logits[0]


def run_one_image(model, processor, device, dtype, image_token_index,
                   grid_size, shortest_edge, crop_size, patch_size, record, max_new_tokens, rng,
                   corrected_gt_categories=None):
    image = Image.open(record["local_path"]).convert("RGB")
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_positions = (inputs["input_ids"][0] == image_token_index).nonzero(as_tuple=True)[0]
    if image_token_positions.numel() != grid_size * grid_size:
        return []

    with torch.no_grad():
        gen_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  return_dict_in_generate=True)
    generated_ids = gen_out.sequences[0, prompt_len:]
    tokenizer = processor.tokenizer
    spans, caption = token_char_spans(tokenizer, generated_ids)
    mentions = find_object_mentions(caption)
    if not mentions:
        return []

    # Self-check: teacher-forced forward with the REAL (fully-kept) image
    # should reproduce the same greedy tokens as generation.
    all_patch_idxs = set(range(grid_size * grid_size))
    full_logits = masked_forward_logits(model, gen_out.sequences, inputs["pixel_values"],
                                         device, all_patch_idxs)
    check_logits = full_logits[prompt_len - 1: prompt_len - 1 + len(generated_ids)]
    match_rate = (check_logits.argmax(dim=-1).cpu() == generated_ids.cpu()).float().mean().item()
    if match_rate < 0.95:
        print(f"  [warn] intervention self-check match rate {match_rate:.2f} on image "
              f"{record['image_id']} -- results for this image are suspect")
        return []

    gt_by_category = {}
    all_gt_patches = set()
    for obj in record["objects"]:
        idxs = bbox_to_patch_indices(obj["bbox"], record["width"], record["height"],
                                      shortest_edge, crop_size, patch_size)
        gt_by_category.setdefault(obj["category"], set()).update(idxs)
        all_gt_patches.update(idxs)

    out_records = []
    for category, char_start, char_end, word in mentions:
        step = token_index_for_char(spans, char_end)
        if step is None or step >= len(generated_ids):
            continue
        pos = prompt_len + step - 1
        original_token = generated_ids[step].item()
        if corrected_gt_categories is not None:
            grounded = category in corrected_gt_categories.get(str(record["image_id"]), [])
        else:
            grounded = category in gt_by_category
        if grounded and category not in gt_by_category:
            # Corrected-label grounded (named in a reference caption) but no COCO
            # instance box exists for this category -- there is no region to
            # selectively ablate/restrict for the necessity test, independent of
            # the label fix. Out of scope for this box-ablation experiment.
            continue

        rec = dict(image_id=record["image_id"], category=category, word=word, grounded=grounded)

        # Condition: restrict to real-evidence patches only (all ground-truth
        # objects in the image, not just this mention's category -- same
        # "any_object" convention as GTC, since hallucinated mentions have no
        # matching category to restrict to).
        if all_gt_patches:
            restrict_logits = masked_forward_logits(model, gen_out.sequences, inputs["pixel_values"],
                                                      device, all_gt_patches)
            restrict_token = restrict_logits[pos].argmax().item()
            rec["restrict_flip"] = restrict_token != original_token
            rec["restrict_new_word"] = tokenizer.decode([restrict_token]).strip()

        # Condition: ablate exactly this mention's ground-truth patches
        # (only defined for grounded mentions); complement condition for
        # hallucinated mentions ablates ALL ground-truth patches instead
        # (there's no mention-specific region to remove).
        if grounded:
            ablate_idxs = all_patch_idxs - gt_by_category[category]
        else:
            ablate_idxs = all_patch_idxs - all_gt_patches
        ablate_logits = masked_forward_logits(model, gen_out.sequences, inputs["pixel_values"],
                                                device, ablate_idxs)
        ablate_token = ablate_logits[pos].argmax().item()
        rec["ablate_flip"] = ablate_token != original_token
        rec["ablate_new_word"] = tokenizer.decode([ablate_token]).strip()
        rec["original_word"] = tokenizer.decode([original_token]).strip()

        # Random-patch ablation baseline: same number of patches removed as
        # the evidence-specific ablation above, but chosen without regard to
        # content, so a flip caused by sheer loss of image information (not
        # by removing that particular evidence) shows up here too.
        n_ablated = len(all_patch_idxs) - len(ablate_idxs)
        random_ablate_idxs = set(rng.sample(sorted(all_patch_idxs), n_ablated))
        random_keep_idxs = all_patch_idxs - random_ablate_idxs
        random_logits = masked_forward_logits(model, gen_out.sequences, inputs["pixel_values"],
                                                device, random_keep_idxs)
        random_token = random_logits[pos].argmax().item()
        rec["random_ablate_flip"] = random_token != original_token
        rec["random_ablate_new_word"] = tokenizer.decode([random_token]).strip()

        out_records.append(rec)
    return out_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(DATA_DIR, "pilot_index.json"))
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "intervention_records_v2.json"))
    args = ap.parse_args()

    with open(args.index) as f:
        index = json.load(f)
    if args.limit:
        index = index[: args.limit]

    corrected_gt_categories = None
    corrected_path = os.path.join(RESULTS_DIR, "corrected_gt_categories.json")
    if os.path.exists(corrected_path):
        corrected_gt_categories = json.load(open(corrected_path))

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

    rng = random.Random(42)
    all_records = []
    for i, record in enumerate(index):
        print(f"[{i + 1}/{len(index)}] image {record['image_id']}")
        try:
            recs = run_one_image(model, processor, args.device, dtype, image_token_index,
                                  grid_size, shortest_edge, crop_size, patch_size,
                                  record, args.max_new_tokens, rng,
                                  corrected_gt_categories=corrected_gt_categories)
            all_records.extend(recs)
        except Exception as e:
            print(f"  [error] {e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nwrote {len(all_records)} records to {args.out}")

    grounded = [r for r in all_records if r["grounded"]]
    halluc = [r for r in all_records if not r["grounded"]]

    def rate(recs, key):
        vals = [r[key] for r in recs if key in r]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    r1, n1 = rate(grounded, "restrict_flip")
    r2, n2 = rate(grounded, "ablate_flip")
    r3, n3 = rate(halluc, "restrict_flip")
    r4, n4 = rate(halluc, "ablate_flip")
    r5, n5 = rate(grounded, "random_ablate_flip")
    r6, n6 = rate(halluc, "random_ablate_flip")
    print(f"\n[1] grounded + restrict-to-evidence flip rate:  {r1} (n={n1})  -- expect LOW (control)")
    print(f"[2] grounded + ablate-evidence flip rate:        {r2} (n={n2})  -- expect HIGH (necessity)")
    print(f"[3] hallucinated + restrict-to-evidence flip rate: {r3} (n={n3})  -- expect HIGH (correctable?)")
    print(f"[4] hallucinated + ablate-all-evidence flip rate:  {r4} (n={n4})  -- expect LOW (prior-driven?)")
    print(f"[5] grounded + random-equal-size-ablation flip rate:      {r5} (n={n5})  -- baseline for [2]")
    print(f"[6] hallucinated + random-equal-size-ablation flip rate:  {r6} (n={n6})  -- baseline for [4]")


if __name__ == "__main__":
    main()
