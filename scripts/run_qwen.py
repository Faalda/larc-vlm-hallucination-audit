"""Qwen2.5-VL replication of the grounding-geometry pipeline.

Qwen2.5-VL inserts a 16x16 grid of visual tokens directly into its 28-layer
decoder. The extraction protocol therefore follows the LLaVA runner, while
using Qwen's chat template and direct stretch-resize preprocessing.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.masking_utils import eager_mask

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
LAYER_SWEEP = list(range(2, 29, 2))
RESOLUTION = 448
GRID_SIZE = 16
PATCH_SIZE = RESOLUTION // GRID_SIZE
PROMPT = "Describe this image in detail."


def fp32_eager_attention(module, query, key, value, attention_mask, scaling,
                         dropout=0.0, **kwargs):
    """Avoid Qwen's bf16 eager-attention generation drift.

    This is the validated Qwen attention interface used in the prior project:
    only score computation is promoted to fp32; outputs retain model dtype.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, :key.shape[-2]].float()
    weights = F.softmax(weights, dim=-1)
    output = torch.matmul(weights, value.float()).to(query.dtype).transpose(1, 2).contiguous()
    return output, weights.to(query.dtype)


def make_inputs(processor, image, device, dtype):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image.resize((RESOLUTION, RESOLUTION))],
                       return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs


def run_one_image(model, processor, device, dtype, record, max_new_tokens):
    image = Image.open(record["local_path"]).convert("RGB")
    inputs = make_inputs(processor, image, device, dtype)
    prompt_len = inputs["input_ids"].shape[1]
    image_positions = (inputs["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=True)[0]
    if image_positions.numel() != GRID_SIZE * GRID_SIZE:
        print(f"  [warn] expected 256 image tokens, found {image_positions.numel()}; skipping")
        return []

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_attentions=True, output_hidden_states=True, output_scores=True,
            return_dict_in_generate=True,
        )
    generated_ids = generated.sequences[0, prompt_len:]
    spans, caption = token_char_spans(processor.tokenizer, generated_ids)
    mentions = find_object_mentions(caption)
    if not mentions:
        return []

    # Qwen's cached-generation logits are not reproduced by a full-sequence
    # teacher-forced forward in this Transformers version, even with the
    # validated fp32 attention interface. We therefore retain only quantities
    # read directly from the generation pass (GTC and alignment), and do not
    # report a Qwen prior-override/null result.
    score_tokens = torch.stack([score[0].argmax() for score in generated.scores])
    if not torch.equal(score_tokens.cpu(), generated_ids.cpu()):
        print(f"  [warn] generation score mismatch; skipping image {record['image_id']}")
        return []

    image_hidden = {L: get_image_hidden(generated.hidden_states, L, image_positions)
                    for L in LAYER_SWEEP}
    gt_by_category, all_gt = {}, set()
    for obj in record["objects"]:
        patches = bbox_to_patch_indices_stretch(
            obj["bbox"], record["width"], record["height"], RESOLUTION, RESOLUTION, PATCH_SIZE)
        gt_by_category.setdefault(obj["category"], set()).update(patches)
        all_gt.update(patches)
    all_gt = sorted(all_gt)

    records = []
    for category, _, char_end, word in mentions:
        step = token_index_for_char(spans, char_end)
        if step is None or step >= len(generated_ids):
            continue
        by_layer = {}
        for L in LAYER_SWEEP:
            attention = get_attn_row(generated.attentions, step, L - 1)
            image_attention = attention[image_positions]
            normalized = image_attention / (image_attention.sum() + 1e-8)
            content = (normalized.unsqueeze(-1).to(image_hidden[L].dtype) * image_hidden[L]).sum(dim=0)
            by_layer[str(L)] = {
                "gtc_any_object": compute_gtc(normalized, all_gt, GRID_SIZE),
                "align_cos": F.cosine_similarity(
                    get_hidden_prev(generated.hidden_states, step, L).float().unsqueeze(0),
                    content.float().unsqueeze(0)).item(),
            }
            if L == LAYER_SWEEP[-1]:
                image_mass = image_attention.sum().item()
                max_attention = image_attention.max().item()
                entropy = -(normalized * torch.log(normalized + 1e-8)).sum().item()
        token_id = generated_ids[step].item()
        records.append(dict(
            image_id=record["image_id"], category=category, word=word,
            grounded=category in gt_by_category, caption=caption,
            image_mass_total=image_mass, max_attn=max_attention, entropy=entropy,
            by_layer=by_layer,
        ))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=os.path.join(DATA_DIR, "pilot_index.json"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "qwen_records.json"))
    args = parser.parse_args()
    with open(args.index) as f:
        index = json.load(f)
    index = index[args.offset:]
    if args.limit:
        index = index[:args.limit]
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=RESOLUTION ** 2,
                                              max_pixels=RESOLUTION ** 2)
    AttentionInterface.register("qwen_fp32_eager", fp32_eager_attention)
    AttentionMaskInterface.register("qwen_fp32_eager", eager_mask)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype,
        attn_implementation={"text_config": "qwen_fp32_eager", "vision_config": "sdpa", "": "sdpa"},
    ).to(args.device).eval()
    all_records = []
    for i, record in enumerate(index):
        print(f"[{i + 1}/{len(index)}] image {record['image_id']}", flush=True)
        try:
            all_records.extend(run_one_image(
                model, processor, args.device, dtype, record, args.max_new_tokens))
        except Exception as exc:
            print(f"  [error] {exc}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"wrote {len(all_records)} records to {args.out}")


if __name__ == "__main__":
    main()
