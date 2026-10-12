# Released Results

These are derived per-mention records used by the CPU analysis scripts.

## Canonical population

- `pilot_records_v4_sensitivity.json`: first source batch.
- `pilot_records_extra.json`: second source batch.
- `pilot_records_caption_corrected.json`: canonical corrected 1,649-mention
  population over 592 contributing images.
- `corrected_gt_categories.json`: caption-plus-instance category sets.

## Audit and baselines

- `e4_label_audit/audit_labels.jsonl`: sanitized 200-item human audit.
- `frequency_prior_raw_scored.json`: leave-one-image-out category frequency.
- `glsim_records.json`: GLSim scores on the matched population.
- `selfgtc_paper1_records.json`: box-free self-GTC.
- `gtc_shuffled_attention_records.json`: magnitude-preserving shuffled control.

## Architectures

`instructblip_records.json`, `paligemma_records.json`,
`llava_gemma_records.json`, `qwen_records*.json`, and
`openimages_records.json` support cross-architecture/distribution analyses.

## Interventions

Original and corrected records are kept as explicit pairs for masking,
single-layer steering, multi-layer steering, and transplantation. This permits
scope-matched comparisons without rerunning model inference.

All files are derived records. Raw third-party images are not distributed.
