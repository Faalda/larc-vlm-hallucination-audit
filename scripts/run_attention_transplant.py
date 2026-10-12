"""Attention-map transplantation with a grounded within-image donor.

Uniform steering (run_attention_steering.py) forces attention onto a target patch set
with a flat, unrealistic distribution. This asks a sharper question: does transplanting a REAL,
naturalistic attention pattern -- the model's own attention when it successfully grounds a
DIFFERENT, actually-present object in the SAME image -- correct a hallucinated mention better
than uniform steering does?

Design: for each hallucinated mention in a "mixed" image (has at least one grounded mention too,
the majority case per the error-analysis finding in main.tex), pick the first grounded mention in
that same image as the donor. In a single teacher-forced forward pass (both mentions' positions
are in the same caption), extract the donor's real per-patch attention distribution at the
steering layers (same L18-22 plateau as the multi-layer steering result) and, via the same
forward-hook mechanism, force the hallucinated mention's query position to use the DONOR's
attention distribution instead of its own -- not uniform, not random, an actual working pattern
from the same image and generation.

Conditions, all on the same mentions for a fair paired comparison:
  - baseline: no intervention (already implicit; flip is relative to the observed word)
  - transplant_donor: the grounded donor's real attention distribution, same image
  - transplant_shuffled: the SAME donor distribution, spatially permuted (magnitude-preserving
    control, mirroring run_gtc_shuffled_attention.py's logic, so we can tell whether any benefit
    comes from the donor's specific spatial pattern or just its concentration statistics)

Usage:
    python run_attention_transplant.py --n-images 592 --layers 18,19,20,21,22
"""
import argparse
import json
import os
import random

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import token_char_spans, token_index_for_char

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

STATE = dict(active=False, target_position=None, target_dist_by_layer=None,
             image_token_positions=None, n_heads=None, head_dim=None)


def make_steering_hook(layer_idx):
    def steering_hook(module, args, kwargs, output):
        if not STATE["active"]:
            return None
        hidden_states = kwargs["hidden_states"]
        attn_output, attn_weights = output
        q_pos = STATE["target_position"]
        image_token_positions = STATE["image_token_positions"]
        p = STATE["target_dist_by_layer"][layer_idx]  # this layer's OWN donor distribution

        x_img = hidden_states[0, image_token_positions, :]
        with torch.no_grad():
            V_img = module.v_proj(x_img)
        n_img = image_token_positions.shape[0]
        V_img_heads = V_img.view(n_img, STATE["n_heads"], STATE["head_dim"])

        retrieved = torch.einsum("n,nhd->hd", p.to(V_img.dtype), V_img_heads)
        with torch.no_grad():
            steered = module.o_proj(retrieved.reshape(-1))

        new_attn_output = attn_output.clone()
        new_attn_output[0, q_pos, :] = steered.to(new_attn_output.dtype)
        return (new_attn_output, attn_weights)
    return steering_hook


def load_local_paths():
    paths = {}
    for f in ["pilot_index.json", "pilot_index_extra.json"]:
        for r in json.load(open(os.path.join(DATA_DIR, f))):
            paths[r["image_id"]] = r
    return paths


def is_grounded_replacement(new_token_text, image_categories):
    for c, _, _, _ in find_object_mentions(new_token_text):
        if c in image_categories:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-images", type=int, default=592)
    ap.add_argument("--layers", default="18,19,20,21,22")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "attention_transplant_records.json"))
    args = ap.parse_args()
    steer_layers = [int(x) for x in args.layers.split(",")]

    # CORRECTED-LABEL population, full 592-image pool (see run_attention_steering.py).
    records = json.load(open(os.path.join(RESULTS_DIR, "pilot_records_caption_corrected.json")))
    for r in records:
        r["grounded"] = r["grounded_corrected"]
    corrected_gt_categories = json.load(open(os.path.join(RESULTS_DIR, "corrected_gt_categories.json")))
    by_image = {}
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)
    mixed_image_ids = sorted(
        iid for iid, ms in by_image.items()
        if any(not m["grounded"] for m in ms) and any(m["grounded"] for m in ms)
    )[: args.n_images]
    meta = load_local_paths()

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ... {len(mixed_image_ids)} mixed images, "
          f"steering layers {steer_layers}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    tokenizer = processor.tokenizer
    image_token_index = model.config.image_token_index

    STATE["n_heads"] = model.model.language_model.config.num_attention_heads
    handles = []
    for L in steer_layers:
        layer_module = model.model.language_model.layers[L - 1]
        STATE["head_dim"] = layer_module.self_attn.head_dim
        handles.append(layer_module.self_attn.register_forward_hook(
            make_steering_hook(L), with_kwargs=True))

    rng = random.Random(0)
    out_records = []
    match_count, match_total = 0, 0

    for i, image_id in enumerate(mixed_image_ids):
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
        n_img = image_token_positions.shape[0]

        found_spans = find_object_mentions(caption_text)
        if len(found_spans) != len(mentions):
            print(f"  [warn] mention count mismatch on image {image_id} -- skipping image")
            continue
        # Positional match only (never search-by-name: repeated words/categories in one caption
        # would all resolve to the same, wrong occurrence otherwise -- the exact bug caught and
        # fixed in run_alignment_perhead.py/run_selfgtc_paper1.py; verify order/category agree).
        mismatch = False
        for m, (fcat, fstart, fend, fword) in zip(mentions, found_spans):
            if fcat != m["category"] or fword.lower() != m["word"].lower():
                mismatch = True
                break
        if mismatch:
            print(f"  [warn] mention order mismatch on image {image_id} -- skipping image")
            continue
        spans, _ = token_char_spans(tokenizer, caption_ids)
        image_categories = set(corrected_gt_categories.get(str(image_id), []))

        def q_pos_for_index(idx):
            fend = found_spans[idx][2]
            tok_idx = token_index_for_char(spans, fend)
            if tok_idx is None:
                return None
            return prompt_len - 1 + tok_idx

        donor_idx = next((k for k, m in enumerate(mentions) if m["grounded"]), None)
        if donor_idx is None:
            continue
        donor = mentions[donor_idx]
        donor_q_pos = q_pos_for_index(donor_idx)
        if donor_q_pos is None:
            continue

        STATE["image_token_positions"] = image_token_positions
        print(f"[{i + 1}/{len(mixed_image_ids)}] image {image_id} "
              f"({len(mentions)} mentions, donor={donor['category']})", flush=True)

        # get donor's REAL attention distribution, PER LAYER (head-averaged, matching all other
        # scripts), from an un-hooked pass -- disable the hook for this read
        STATE["active"] = False
        with torch.no_grad():
            out_clean = model(input_ids=full_ids, pixel_values=pixel_values, output_attentions=True)
        donor_dist_by_layer = {}
        for L in steer_layers:
            row = out_clean.attentions[L - 1][0][:, donor_q_pos, :].mean(dim=0)
            over_img = row[image_token_positions]
            donor_dist_by_layer[L] = over_img / (over_img.sum() + 1e-8)
        perm = list(range(n_img))
        rng.shuffle(perm)
        donor_dist_by_layer_shuffled = {L: d[perm] for L, d in donor_dist_by_layer.items()}

        for idx, m in enumerate(mentions):
            if m["grounded"]:
                continue
            q_pos = q_pos_for_index(idx)
            if q_pos is None or q_pos == donor_q_pos:
                continue
            original_token_id = int(caption_ids[q_pos - (prompt_len - 1)])

            STATE["active"] = False
            with torch.no_grad():
                out_base = model(input_ids=full_ids, pixel_values=pixel_values)
            base_argmax = out_base.logits[0, q_pos].argmax().item()
            match_total += 1
            if base_argmax == original_token_id:
                match_count += 1

            results_this_mention = {}
            for cond_name, dist_by_layer in [("transplant_donor", donor_dist_by_layer),
                                              ("transplant_shuffled", donor_dist_by_layer_shuffled)]:
                STATE["active"] = True
                STATE["target_position"] = q_pos
                STATE["target_dist_by_layer"] = dist_by_layer
                with torch.no_grad():
                    out_steer = model(input_ids=full_ids, pixel_values=pixel_values)
                STATE["active"] = False
                new_argmax = out_steer.logits[0, q_pos].argmax().item()
                flip = new_argmax != original_token_id
                new_word_text = tokenizer.decode([new_argmax], skip_special_tokens=True)
                grounded_repl = is_grounded_replacement(new_word_text, image_categories) if flip else False
                results_this_mention[cond_name] = dict(
                    flip=flip, new_word=new_word_text, grounded_replacement=grounded_repl)

            out_records.append(dict(
                image_id=image_id, category=m["category"], word=m["word"],
                donor_category=donor["category"],
                transplant_donor=results_this_mention["transplant_donor"],
                transplant_shuffled=results_this_mention["transplant_shuffled"],
            ))

        with open(args.out, "w") as f:
            json.dump(out_records, f, indent=2)

    for h in handles:
        h.remove()
    print(f"\nBaseline argmax match rate: {match_count}/{match_total} = "
          f"{match_count/match_total*100 if match_total else 0:.1f}%")
    print(f"wrote {len(out_records)} records to {args.out}")


if __name__ == "__main__":
    main()
