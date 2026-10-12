# Reproducibility Guide

## Levels of reproduction

### 1. Released-record verification

This is the fastest path and does not require a GPU or raw images.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/build_corrected_table1.py
python scripts/build_confound_baseline.py
python scripts/analyze_glsim.py
python scripts/analyze_e4_label_audit.py
python scripts/analyze_gtc_shuffled.py
```

### 2. Label reconstruction

Download COCO images and annotations, then rebuild the caption-augmented ground
truth and corrected labels:

```bash
python scripts/download_pilot_data.py
python scripts/download_extra_data.py
python scripts/build_corrected_gt_categories.py
python scripts/rebuild_labels_with_captions.py
```

The released corrected records remain the canonical artifact. Rebuilding should
preserve the reported population after excluding documented compound-mention
collisions.

### 3. Model regeneration

Model runs require a CUDA GPU and access to the corresponding Hugging Face model
weights. Start with `python scripts/<runner>.py --help`. The primary run is:

```bash
python scripts/run_pilot.py
```

Cross-architecture and intervention runners are listed in the root README.
Generation can vary across GPU kernels and Transformers versions; every runner
performs local token/logit consistency checks before writing records.

## Expected headline outputs

| Check | Expected result |
|---|---:|
| GTC, original labels | 0.635 AUC |
| GTC, corrected labels | 0.551 AUC |
| Per-head alignment, original labels | 0.696 AUC |
| Per-head alignment, corrected labels | 0.682 AUC |
| Model-free baseline | 0.789 AUC |
| GLSim increment beyond baseline | +0.041 AUC |
| E4 Cohen's kappa | 0.505 |
| E4 final labels | 134 / 61 / 5 |
| Corrected real vs. shuffled GTC | 0.624 / 0.572 AUC |

Small differences in bootstrap interval endpoints can occur across numerical
library versions. Point estimates and sample counts should agree.

## Released-record provenance

- `results/pilot_records_v4_sensitivity.json` and
  `results/pilot_records_extra.json` are the two source batches.
- `results/pilot_records_caption_corrected.json` is their corrected,
  canonical 1,649-mention population.
- Original/corrected intervention files are retained as explicit pairs.
- `results/e4_label_audit/audit_labels.jsonl` is the sanitized public audit;
  private annotation screens and rater identities are not released.

## Data licenses

COCO and Open Images are downloaded from their maintainers and remain under
their respective terms. This repository contains derived metadata and result
records, not third-party images.
