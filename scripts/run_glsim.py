"""E2 (main.new.tex, tab:prereg, \\textsc{glsim-retest}): re-implement GLSim
(Park & Li, NeurIPS 2025, arXiv:2508.19972) on our own population, scored
against our own corrected labels, so it can go through the same \\LARC{}
checks (C1 label-fragility already N/A -- GLSim's own util/chair.py already
includes the caption-derived source; C2 confound comparison is the point)
as GTC and alignment.

Method, read directly from the released code
(github.com/deeplearning-wisc/glsim, lvlm/LLaVA.py + evaluate.py), not
re-derived from the paper's prose alone, because the two are not guaranteed
to match:
  - For LLaVA-1.5-7B: image_layer=32 (last layer), text_layer=31,
    k=32 (patches), w=0.6.
  - global_score  = cosine(question_hidden[layer=31], token_hidden[layer=32])
    where "question_hidden" is the LAST PROMPT TOKEN's hidden state (the
    prefill pass's last position) and "token_hidden" is the object token's
    own hidden state.
  - local_score   = mean over the top-k image patches (by Visual-Logit-Lens
    probability of the target token at layer=32) of
    cosine(token_hidden[layer=31], patch_hidden[layer=32])
  - final score   = 0.6 * global_score + 0.4 * local_score
  - "token_hidden" at generation step s is get_hidden_prev(hidden_states, s,
    layer) -- i.e. the hidden state that PRODUCED token s's logits (the
    model's pre-commitment state), not a post-hoc embedding of token s. This
    matches GLSim's own LLaVA.py indexing (hidden_states[toke_idx]) exactly;
    it is their deliberate choice, not something we are second-guessing.
  - Multi-token objects: GLSim's paper says "the first token is used" --
    we therefore take the FIRST generated token whose decoded span overlaps
    the word, not the last (our own GTC/align_cos pipeline's
    token_index_for_char utility finds the LAST subtoken that completes a
    word, which is the wrong convention for GLSim specifically; see
    first_token_step() below, a small local helper, not a change to the
    shared, already-validated llava_geometry.py).

This re-runs generation (greedy, deterministic, same prompt as run_pilot.py)
rather than reusing cached captions, because raw hidden-state tensors were
never saved to disk (only scalar GTC/align_cos features were). Generation
is deterministic (do_sample=False), so this reproduces the exact same
captions/mentions as pilot_records_caption_corrected.json -- verified
per-image below rather than assumed.

Usage:
    python run_glsim.py --limit 3          # smoke test
    python run_glsim.py                    # full population
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from larc_audit.coco_categories import find_object_mentions
from larc_audit.llava_geometry import get_hidden_prev, get_image_hidden, token_char_spans

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

IMAGE_LAYER = 32
TEXT_LAYER = 31
TOP_K = 32
W_GLOBAL = 0.6


def first_token_step(spans, char_start):
    """First generated-token index whose decoded span overlaps the word
    starting at char_start -- GLSim's "first token" convention for
    multi-token objects, distinct from token_index_for_char's "last
    subtoken that completes the word" convention used elsewhere in this
    project for GTC/align_cos."""
    for k, (_s, e) in enumerate(spans):
        if e > char_start:
            return k
    return None


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    return model.language_model.lm_head


def glsim_score(hidden_states, step, lm_head, image_token_positions, target_token_id):
    question_hidden = get_hidden_prev(hidden_states, 0, TEXT_LAYER)
    token_hidden_32 = get_hidden_prev(hidden_states, step, IMAGE_LAYER)
    token_hidden_31 = get_hidden_prev(hidden_states, step, TEXT_LAYER)

    global_score = F.cosine_similarity(
        question_hidden.float().unsqueeze(0), token_hidden_32.float().unsqueeze(0)
    ).item()

    patch_hidden_32 = get_image_hidden(hidden_states, IMAGE_LAYER, image_token_positions)
    with torch.no_grad():
        logits = lm_head(patch_hidden_32)
        probs = torch.softmax(logits.float(), dim=-1)[:, target_token_id]
    k = min(TOP_K, probs.shape[0])
    top_k_idx = torch.topk(probs, k).indices
    selected = F.normalize(patch_hidden_32[top_k_idx].float(), p=2, dim=-1)
    token_31_norm = F.normalize(token_hidden_31.float().unsqueeze(0), p=2, dim=-1)
    local_score = (selected @ token_31_norm.squeeze(0)).mean().item()

    return W_GLOBAL * global_score + (1 - W_GLOBAL) * local_score, global_score, local_score


def run_one_image(model, processor, device, dtype, image_token_index,
                   record, max_new_tokens, expected_categories):
    image = Image.open(record["local_path"]).convert("RGB")
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    prompt_len = inputs["input_ids"].shape[1]

    image_token_positions = (inputs["input_ids"][0] == image_token_index).nonzero(as_tuple=True)[0]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
        )

    generated_ids = gen_out.sequences[0, prompt_len:]
    tokenizer = processor.tokenizer
    spans, caption = token_char_spans(tokenizer, generated_ids)
    mentions = find_object_mentions(caption)
    if not mentions:
        return [], caption, set()

    found_categories = {m[0] for m in mentions}
    if expected_categories is not None and found_categories != expected_categories:
        print(f"  [warn] image {record['image_id']}: mention categories differ from "
              f"existing records -- expected {expected_categories}, got "
              f"{found_categories}. Generation may not be reproducing the "
              f"original deterministic run exactly; treat this image's "
              f"GLSim scores with caution.")

    lm_head = get_lm_head(model)
    out = []
    for category, char_start, char_end, word in mentions:
        step = first_token_step(spans, char_start)
        if step is None or step >= len(generated_ids):
            continue
        target_token_id = generated_ids[step].item()
        try:
            score, g, l = glsim_score(gen_out.hidden_states, step, lm_head,
                                       image_token_positions, target_token_id)
        except Exception as e:
            print(f"  [error] {record['image_id']}/{category}: {e}")
            continue
        out.append(dict(image_id=record["image_id"], category=category, word=word,
                         glsim_score=score, glsim_global=g, glsim_local=l))
    return out, caption, found_categories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(DATA_DIR, "pilot_index.json"))
    ap.add_argument("--index_extra", default=os.path.join(DATA_DIR, "pilot_index_extra.json"))
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "glsim_records.json"))
    ap.add_argument("--existing", default=os.path.join(RESULTS_DIR, "pilot_records_caption_corrected.json"),
                     help="used only to cross-check that generation reproduces the same "
                          "mention categories per image, not to skip generation")
    args = ap.parse_args()

    index = json.load(open(args.index))
    if os.path.exists(args.index_extra):
        index = index + json.load(open(args.index_extra))
    if args.limit:
        index = index[: args.limit]

    existing_by_image = {}
    if os.path.exists(args.existing):
        for r in json.load(open(args.existing)):
            existing_by_image.setdefault(r["image_id"], set()).add(r["category"])

    dtype = torch.float16
    print(f"loading {args.model} on {args.device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()
    image_token_index = model.config.image_token_index
    print(f"image_token_index={image_token_index}, lm_head={get_lm_head(model)}")

    all_records = []
    n_mismatch = 0
    for i, record in enumerate(index):
        print(f"[{i + 1}/{len(index)}] image {record['image_id']}")
        expected = existing_by_image.get(record["image_id"])
        try:
            recs, caption, found = run_one_image(
                model, processor, args.device, dtype, image_token_index,
                record, args.max_new_tokens, expected)
            all_records.extend(recs)
            if expected is not None and found != expected:
                n_mismatch += 1
        except Exception as e:
            print(f"  [error] {e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nwrote {len(all_records)} records to {args.out}")
    print(f"images with a mention-category mismatch vs existing records: "
          f"{n_mismatch}/{len(index)}")


if __name__ == "__main__":
    main()
