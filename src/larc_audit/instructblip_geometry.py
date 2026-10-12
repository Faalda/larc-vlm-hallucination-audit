"""InstructBLIP-specific extraction primitives.

InstructBLIP's LLM side (Vicuna-7B, 32 layers) has the exact same
hidden_states/attentions tuple conventions as LLaVA's, so get_attn_row,
get_hidden_prev, get_image_hidden, token_char_spans, token_index_for_char,
and compute_gtc from llava_geometry.py are reused unchanged.

Two things are genuinely different and need new code:

1. There is no fixed image-token-to-patch-grid correspondence on the LLM
   side. InstructBLIP compresses the image into 32 LEARNED query tokens via
   a Q-Former; each query is its own weighted combination of image patches,
   determined by the Q-Former's own cross-attention, not a spatial index.
   To compute GTC (which needs a patch-grid distance), we compose two
   attention stages: the LLM's attention from the generated word back to
   the 32 query positions, with each query's own Q-Former cross-attention
   distribution over the 256 image patches. This gives an effective
   per-word distribution over patches, exactly like LLaVA's direct
   per-word distribution over its 576 image tokens.

2. InstructBLIP's image processor does a direct independent-axis resize to
   224x224 (do_center_crop=False) rather than LLaVA/CLIP's
   shortest-edge-then-center-crop. The bbox-to-patch-grid transform is
   correspondingly simpler (no crop offset, but the aspect ratio is not
   preserved).
"""
import numpy as np
import torch


def get_qformer_patch_distributions(qformer_cross_attentions, layer_idx=-1):
    """Average over heads, drop the CLS token, renormalize -> [n_query, n_patch]
    matrix: row i is query i's own distribution over image patches.
    """
    attn = qformer_cross_attentions[layer_idx][0]  # [heads, n_query, 1+n_patch]
    attn = attn.mean(dim=0)  # [n_query, 1+n_patch]
    attn = attn[:, 1:]  # drop CLS token
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    return attn  # [n_query, n_patch]


def compose_patch_attention(llm_attn_over_queries, qformer_patch_dist):
    """llm_attn_over_queries: [n_query] attention from the generated word to
    each query position. qformer_patch_dist: [n_query, n_patch]. Returns the
    composite [n_patch] distribution: how much of the word's attention,
    transitively, lands on each image patch.
    """
    composite = llm_attn_over_queries.unsqueeze(0) @ qformer_patch_dist.to(llm_attn_over_queries.dtype)
    composite = composite.squeeze(0)
    return composite / (composite.sum() + 1e-8)


def bbox_to_patch_indices_stretch(bbox, orig_w, orig_h, resize_w, resize_h, patch_size):
    """Direct independent-axis resize (no crop): map a COCO [x,y,w,h] pixel
    bbox straight through to the patch grid. Returns [] if degenerate.
    """
    x, y, w, h = bbox
    scale_x, scale_y = resize_w / orig_w, resize_h / orig_h
    x0, y0 = x * scale_x, y * scale_y
    x1, y1 = (x + w) * scale_x, (y + h) * scale_y
    x0, x1 = max(0.0, x0), min(resize_w, x1)
    y0, y1 = max(0.0, y0), min(resize_h, y1)
    if x1 <= x0 or y1 <= y0:
        return []

    grid_w, grid_h = resize_w // patch_size, resize_h // patch_size
    patch_w, patch_h = resize_w / grid_w, resize_h / grid_h
    col0, col1 = int(x0 // patch_w), min(grid_w, int(np.ceil(x1 / patch_w)))
    row0, row1 = int(y0 // patch_h), min(grid_h, int(np.ceil(y1 / patch_h)))
    return [r * grid_w + c for r in range(row0, row1) for c in range(col0, col1)]
