"""Core extraction primitives for the grounding-geometry pilot.

Index bookkeeping this file depends on (verified against transformers'
LlavaForConditionalGeneration.generate() output with use_cache=True):

- outputs.attentions is a tuple of length T (one per generated token). Step 0
  covers the full prefill forward pass over the whole prompt (image + text
  tokens); its attention tensors have shape [batch, heads, full_seq, full_seq].
  Steps k>=1 cover only the newly generated query position, attending back
  over the growing kv-cache; shape [batch, heads, 1, prompt_len + k].
- outputs.hidden_states follows the same per-step shape pattern.
- outputs.scores[k] are the final-layer logits used to pick generated token k
  (pre-argmax), independent of caching, so log p(token_k | image) reads
  directly off outputs.scores[k].
- Image token positions are wherever input_ids == model.config.image_token_index;
  the LlavaProcessor already expands the <image> placeholder to num_image_tokens
  copies before this point, so no manual span arithmetic is needed to find them.
"""
import numpy as np
import torch


def get_attn_row(attentions, step, layer):
    t = attentions[step][layer][0]  # [heads, q, kv]
    row = t[:, -1, :] if step == 0 else t[:, 0, :]
    return row.mean(dim=0)  # average heads -> [kv]


def get_hidden_prev(hidden_states, step, layer):
    t = hidden_states[step][layer][0]  # [seq_or_1, hidden]
    return t[-1] if step == 0 else t[0]


def get_image_hidden(hidden_states, layer, image_token_positions):
    t = hidden_states[0][layer][0]  # prefill pass, [full_seq, hidden]
    return t[image_token_positions]  # [n_img_tokens, hidden]


def token_char_spans(tokenizer, generated_ids):
    """Char span (start, end) in the decoded caption contributed by each
    generated token id, built by incremental re-decoding (robust to BPE
    merge quirks that a single re-encode-from-scratch pass could mismatch).
    """
    ids_list = generated_ids.tolist() if torch.is_tensor(generated_ids) else list(generated_ids)
    spans = []
    prev_text = ""
    for k in range(len(ids_list)):
        text_so_far = tokenizer.decode(ids_list[: k + 1], skip_special_tokens=True)
        spans.append((len(prev_text), len(text_so_far)))
        prev_text = text_so_far
    return spans, prev_text


def token_index_for_char(spans, char_end):
    """First generated-token index whose cumulative decoded span reaches char_end
    (i.e. the token at which the word is fully spelled out, for multi-subtoken words).
    """
    for k, (_, e) in enumerate(spans):
        if e >= char_end:
            return k
    return None


def bbox_to_patch_indices(bbox, orig_w, orig_h, shortest_edge, crop_size, patch_size):
    """Map a COCO [x, y, w, h] pixel bbox through CLIP's resize+center-crop
    preprocessing into a set of flattened patch indices on the resulting
    square patch grid. Returns [] if the box is fully cropped away.
    """
    x, y, w, h = bbox
    scale = shortest_edge / min(orig_w, orig_h)
    resized_w, resized_h = orig_w * scale, orig_h * scale
    crop_w, crop_h = crop_size["width"], crop_size["height"]
    crop_x0 = (resized_w - crop_w) / 2.0
    crop_y0 = (resized_h - crop_h) / 2.0

    x0 = x * scale - crop_x0
    y0 = y * scale - crop_y0
    x1 = (x + w) * scale - crop_x0
    y1 = (y + h) * scale - crop_y0
    x0, x1 = max(0.0, x0), min(crop_w, x1)
    y0, y1 = max(0.0, y0), min(crop_h, y1)
    if x1 <= x0 or y1 <= y0:
        return []

    grid_size = crop_h // patch_size
    patch_w, patch_h = crop_w / grid_size, crop_h / grid_size
    col0, col1 = int(x0 // patch_w), min(grid_size, int(np.ceil(x1 / patch_w)))
    row0, row1 = int(y0 // patch_h), min(grid_size, int(np.ceil(y1 / patch_h)))
    return [r * grid_size + c for r in range(row0, row1) for c in range(col0, col1)]


def bbox_to_pixel_coords(bbox, orig_w, orig_h, shortest_edge, crop_size):
    """Same resize+center-crop transform as bbox_to_patch_indices, but
    returning pixel-space (x0, y0, x1, y1) in the cropped frame instead of
    patch indices -- for drawing the box directly on the preprocessed image.
    """
    x, y, w, h = bbox
    scale = shortest_edge / min(orig_w, orig_h)
    resized_w, resized_h = orig_w * scale, orig_h * scale
    crop_w, crop_h = crop_size["width"], crop_size["height"]
    crop_x0 = (resized_w - crop_w) / 2.0
    crop_y0 = (resized_h - crop_h) / 2.0

    x0 = x * scale - crop_x0
    y0 = y * scale - crop_y0
    x1 = (x + w) * scale - crop_x0
    y1 = (y + h) * scale - crop_y0
    x0, x1 = max(0.0, x0), min(crop_w, x1)
    y0, y1 = max(0.0, y0), min(crop_h, y1)
    return (x0, y0, x1, y1)


def compute_gtc(attn_over_image, target_patch_idxs, grid_size):
    """Grounding Transport Cost: attention-weighted mean distance (in patch
    units) from every image patch to its nearest target patch.
    """
    if len(target_patch_idxs) == 0:
        return None
    coords = np.array([(i // grid_size, i % grid_size) for i in range(grid_size * grid_size)],
                       dtype=np.float32)
    target_coords = coords[target_patch_idxs]
    dists = np.min(np.linalg.norm(coords[:, None, :] - target_coords[None, :, :], axis=-1), axis=1)
    attn = attn_over_image.detach().float().cpu().numpy()
    return float(np.sum(attn * dists))
