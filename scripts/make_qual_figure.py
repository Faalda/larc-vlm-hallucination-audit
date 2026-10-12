"""Generate the qualitative attention-heatmap figure replacing tab:qual:
one hallucinated and one grounded mention, each shown as the actual
preprocessed image the model saw, with the real attention distribution
(layer 20, same layer as the headline GTC number) overlaid as a heatmap
and the ground-truth box(es) drawn on top.

Reuses the exact resize+center-crop transform already validated for GTC
(llava_geometry.bbox_to_patch_indices/bbox_to_pixel_coords) so the box
and the heatmap are guaranteed to be in the same coordinate frame.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import (
    bbox_to_pixel_coords,
    get_attn_row,
    token_char_spans,
    token_index_for_char,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LAYER = 20

TARGETS = [
    dict(image_id=279387, category="traffic light", label="hallucinated"),
    dict(image_id=368291, category="carrot", label="grounded"),
]


def unnormalize(pixel_values, image_mean, image_std):
    arr = pixel_values[0].float().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(image_std) + np.array(image_mean)
    return np.clip(arr, 0, 1)


def main():
    with open(os.path.join(DATA_DIR, "pilot_index.json")) as f:
        index = {r["image_id"]: r for r in json.load(f)}

    device = "cuda:0"
    dtype = torch.float16
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager")
    model.to(device)
    model.eval()

    ip = processor.image_processor
    shortest_edge = ip.size.get("shortest_edge") or ip.size.get("height")
    crop_size = ip.crop_size
    patch_size = model.config.vision_config.patch_size
    grid_size = crop_size["height"] // patch_size
    image_token_index = model.config.image_token_index

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))

    for ax, target in zip(axes, TARGETS):
        record = index[target["image_id"]]
        image = Image.open(record["local_path"]).convert("RGB")
        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        prompt_len = inputs["input_ids"].shape[1]
        image_token_positions = (inputs["input_ids"][0] == image_token_index).nonzero(as_tuple=True)[0]

        with torch.no_grad():
            gen_out = model.generate(
                **inputs, max_new_tokens=40, do_sample=False,
                output_attentions=True, return_dict_in_generate=True,
            )
        generated_ids = gen_out.sequences[0, prompt_len:]
        tokenizer = processor.tokenizer
        spans, caption = token_char_spans(tokenizer, generated_ids)
        mentions = find_object_mentions(caption)
        match = next((m for m in mentions if m[0] == target["category"]), None)
        if match is None:
            print(f"  [warn] category {target['category']!r} not found in caption for "
                  f"image {target['image_id']}: {caption!r}")
            continue
        category, char_start, char_end, word = match
        step = token_index_for_char(spans, char_end)

        attn_row = get_attn_row(gen_out.attentions, step, LAYER - 1)
        attn_over_image = attn_row[image_token_positions]
        attn_norm = (attn_over_image / (attn_over_image.sum() + 1e-8)).float().cpu().numpy()
        heat = attn_norm.reshape(grid_size, grid_size)
        print(f"  heat stats: min={heat.min():.5f} max={heat.max():.5f} "
              f"mean={heat.mean():.5f} std={heat.std():.5f}")

        # Attention over 576 patches is inherently diffuse (max is rarely
        # more than a few x the mean) -- percentile-based color limits make
        # the real concentration visible instead of washing out under a
        # plain min/max linear scale.
        vmin, vmax = np.percentile(heat, 40), np.percentile(heat, 99)

        img_arr = unnormalize(inputs["pixel_values"], ip.image_mean, ip.image_std)
        ax.imshow(img_arr)
        im = ax.imshow(heat, cmap="inferno", alpha=0.6, vmin=vmin, vmax=vmax,
                        extent=(0, crop_size["width"], crop_size["height"], 0),
                        interpolation="bilinear")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

        for obj in record["objects"]:
            x0, y0, x1, y1 = bbox_to_pixel_coords(obj["bbox"], record["width"], record["height"],
                                                    shortest_edge, crop_size)
            if x1 <= x0 or y1 <= y0:
                continue
            edge_color = "#2ecc71" if obj["category"] == target["category"] else "#3498db"
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.3,
                                      edgecolor=edge_color, facecolor="none")
            ax.add_patch(rect)

        ax.set_title(f"“{word}” ({target['label']})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        print(f"image {target['image_id']}: word={word!r} step={step} caption={caption[:80]!r}")

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "figures", "qual_heatmap.pdf")
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
