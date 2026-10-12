"""Multi-layer extension of run_attention_steering.py. Single-layer steering at L20 (GTC's own
peak layer) gave a near-null flip rate (4.1%, indistinguishable from steering-to-noise at 3.3%).
Before concluding "routing is not causally load-bearing," check whether that null was just
DILUTION -- one layer's contribution overridden by 11 further untouched layers -- by steering
attention simultaneously at every layer in GTC's own established plateau, L18-22
(Section~discussion: "GTC's AUC is stable across sampled layers (0.601-0.635)... L18-22 form a
flat plateau"). If steering the whole plateau still doesn't flip much, the null is not a dilution
artifact of picking only one layer.

Same intervention mechanics as run_attention_steering.py (validated there): a forward hook per
layer, each independently reconstructing that layer's own attention output under a forced,
uniform-over-target-patches distribution, via that layer's own v_proj/o_proj. All layers share
the same target position and patch set for a given forward pass, so hooks on L18..L22 compound
naturally through the residual stream -- no additional plumbing needed beyond registering the
same hook function on multiple modules.

Usage:
    python run_attention_steering_multilayer.py --n-images 100 --layers 18,19,20,21,22
"""
import argparse
import json
import os
import random

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import bbox_to_patch_indices, token_char_spans, token_index_for_char

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

STATE = dict(active=False, target_position=None, target_patches=None,
             image_token_positions=None, n_heads=None, head_dim=None)


def steering_hook(module, args, kwargs, output):
    if not STATE["active"]:
        return None
    hidden_states = kwargs["hidden_states"]
    attn_output, attn_weights = output
    q_pos = STATE["target_position"]
    image_token_positions = STATE["image_token_positions"]
    target_local = STATE["target_patches"]

    x_img = hidden_states[0, image_token_positions, :]
    with torch.no_grad():
        V_img = module.v_proj(x_img)
    n_img = image_token_positions.shape[0]
    V_img_heads = V_img.view(n_img, STATE["n_heads"], STATE["head_dim"])

    p = torch.zeros(n_img, device=x_img.device, dtype=V_img.dtype)
    p[target_local] = 1.0 / len(target_local)
    retrieved = torch.einsum("n,nhd->hd", p, V_img_heads)
    with torch.no_grad():
        steered = module.o_proj(retrieved.reshape(-1))

    new_attn_output = attn_output.clone()
    new_attn_output[0, q_pos, :] = steered.to(new_attn_output.dtype)
    return (new_attn_output, attn_weights)


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
    ap.add_argument("--n-images", type=int, default=600)
    ap.add_argument("--layers", default="18,19,20,21,22")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "attention_steering_multilayer_records.json"))
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
    halluc_image_ids = sorted({r["image_id"] for r in records if not r["grounded"]})[: args.n_images]
    meta = load_local_paths()

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ... steering layers {steer_layers}")
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

    STATE["n_heads"] = model.model.language_model.config.num_attention_heads
    handles = []
    for L in steer_layers:
        layer_module = model.model.language_model.layers[L - 1]
        STATE["head_dim"] = layer_module.self_attn.head_dim  # same for every layer, harmless reassignment
        handles.append(layer_module.self_attn.register_forward_hook(steering_hook, with_kwargs=True))

    rng = random.Random(0)
    out_records = []
    match_count, match_total = 0, 0

    for i, image_id in enumerate(halluc_image_ids):
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

        found_spans = find_object_mentions(caption_text)
        if len(found_spans) != len(mentions):
            print(f"  [warn] mention count mismatch on image {image_id} -- skipping image")
            continue
        spans, _ = token_char_spans(tokenizer, caption_ids)

        gt_by_category = {}
        all_gt_patches_set = set()
        for obj in img_meta["objects"]:
            idxs = bbox_to_patch_indices(obj["bbox"], img_meta["width"], img_meta["height"],
                                          shortest_edge, crop_size, patch_size)
            gt_by_category.setdefault(obj["category"], set()).update(idxs)
            all_gt_patches_set.update(idxs)
        image_categories = set(corrected_gt_categories.get(str(image_id), []))
        n_img = image_token_positions.shape[0]
        all_gt_local = sorted(all_gt_patches_set)
        non_gt_local = [j for j in range(n_img) if j not in all_gt_patches_set]
        if not all_gt_local or not non_gt_local:
            continue

        STATE["image_token_positions"] = image_token_positions

        print(f"[{i + 1}/{len(halluc_image_ids)}] image {image_id} "
              f"({len(mentions)} mentions, {sum(1 for m in mentions if not m['grounded'])} halluc)",
              flush=True)

        for m in mentions:
            if m["grounded"]:
                continue
            match = None
            for fcat, fstart, fend, fword in found_spans:
                if fcat == m["category"] and fword.lower() == m["word"].lower():
                    match = (fstart, fend)
            if match is None:
                continue
            tok_idx = token_index_for_char(spans, match[1])
            if tok_idx is None:
                continue
            q_pos = prompt_len - 1 + tok_idx
            original_token_id = int(caption_ids[tok_idx])

            STATE["active"] = False
            with torch.no_grad():
                out_base = model(input_ids=full_ids, pixel_values=pixel_values)
            base_argmax = out_base.logits[0, q_pos].argmax().item()
            match_total += 1
            if base_argmax == original_token_id:
                match_count += 1

            k = min(len(all_gt_local), n_img - 1)
            random_local = rng.sample(non_gt_local, min(k, len(non_gt_local)))

            results_this_mention = {}
            for cond_name, target_local in [("steer_gt", all_gt_local), ("steer_random", random_local)]:
                STATE["active"] = True
                STATE["target_position"] = q_pos
                STATE["target_patches"] = torch.tensor(target_local, device=args.device)
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
                n_gt_patches=len(all_gt_local),
                steer_gt=results_this_mention["steer_gt"],
                steer_random=results_this_mention["steer_random"],
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
