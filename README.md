# LARC: Auditing Internal Hallucination Probes in VLMs

Code, released records, human-audit labels, and reproducibility assets for **"Your
Hallucination Detector May Be Measuring Your Labels: A Four-Check Audit for
Internal Probes in Vision--Language Models."**

LARC tests four failure modes:

1. **Label fragility:** perturb or repair the ground truth and measure
   $\Delta$AUC.
2. **Alternative baseline:** compare against model-free scene and caption
   statistics under image-grouped cross-validation.
3. **Model-removed control:** test how much signal remains without a model
   forward pass.
4. **Causal scope matching:** compare interventions on the same population
   before attributing changes to relabeling.

## Repository layout

```text
src/larc_audit/       Shared extraction and geometry package
scripts/              Analysis, model-run, intervention, and figure workflows
data/                 Sanitized image indexes (raw images excluded)
results/              Canonical released per-mention and audit records
figures/              Standalone editable figure sources
assets/figures/       Generated figure assets
docs/                 Detailed reproducibility documentation
tests/                Lightweight artifact-integrity tests
```

The manuscript source is intentionally maintained separately and is not part of
this public code/data artifact.

## Quick verification

The released per-mention records allow the main analyses to run without a GPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/build_corrected_table1.py
python scripts/build_confound_baseline.py
python scripts/analyze_glsim.py
python scripts/analyze_e4_label_audit.py
python scripts/analyze_gtc_shuffled.py
python scripts/analyze_attention_steering.py
python scripts/analyze_attention_steering_multilayer.py
python scripts/analyze_attention_transplant.py
```

Key checks reproduced by these commands include:

- GTC: $0.635 \rightarrow 0.551$ after label correction.
- Per-head alignment: $0.696 \rightarrow 0.682$.
- Model-free baseline: AUC $0.789$.
- GLSim increment beyond that baseline: $+0.041$.
- Multi-rater audit: $\kappa=0.505$; final labels 134 visible, 61 not
  visible, and 5 ambiguous.
- Corrected shuffled-GTC comparison: $0.624$ versus $0.572$,
  $\Delta=0.052$.

Bootstrap estimates are deterministic for the seeds encoded in each script but
can differ in the last decimal across NumPy/scikit-learn versions.

## Artifact map

| Component | Code | Released records |
|---|---|---|
| C1 label correction | `scripts/rebuild_labels_with_captions.py`, `scripts/build_corrected_gt_categories.py`, `scripts/build_corrected_table1.py` | `results/pilot_records_caption_corrected.json`, `results/corrected_gt_categories.json` |
| Multi-rater audit | `scripts/sample_e4_label_audit.py`, `scripts/annotate_e4.py`, `scripts/analyze_e4_label_audit.py` | `results/e4_label_audit/audit_labels.jsonl` |
| C2 baseline | `scripts/build_confound_baseline.py` | `results/frequency_prior_raw_scored.json` |
| C3 controls | `scripts/verify_gtc_robustness_corrected.py`, `scripts/run_selfgtc_paper1.py`, `scripts/run_gtc_shuffled_attention.py` | `results/selfgtc_paper1_records.json`, `results/gtc_shuffled_attention_records.json` |
| GLSim retest | `scripts/run_glsim.py`, `scripts/analyze_glsim.py` | `results/glsim_records.json` |
| Cross-architecture | model runners in `scripts/` | model-specific JSON files in `results/` |
| C4 interventions | intervention runners and matching analyzers in `scripts/` | original/corrected record pairs in `results/` |
| Figures | figure generators in `scripts/`, `figures/label_fragility.tex` | `assets/figures/` |

`results/pilot_records_v4_sensitivity.json` and
`results/pilot_records_extra.json` are retained as source records for rebuilding
the corrected population. `results/pilot_records_v3_chair.json` is retained for
the held-out layer-selection analysis.

## Data preparation

Raw images and full COCO annotations are not redistributed. Download the indexed
subsets with:

```bash
python scripts/download_pilot_data.py
python scripts/download_extra_data.py
python scripts/download_openimages_data.py
```

The index files use repository-relative paths. The COCO scripts also download
`instances_val2014.json` and `captions_val2014.json`, which are required to
rebuild the corrected CHAIR-style ground truth.

## GPU reruns

The primary model is LLaVA-1.5-7B. Cross-architecture scripts cover
InstructBLIP-Vicuna-7B, PaliGemma-3B, CLIP+Gemma-2B, and Qwen2.5-VL. Runs were
performed on a single RTX 6000 Ada 48 GB GPU. Typical entry points are:

```bash
python scripts/run_pilot.py
python scripts/rebuild_labels_with_captions.py
python scripts/run_instructblip.py
python scripts/run_paligemma.py
python scripts/run_llava_gemma.py
python scripts/run_qwen.py
python scripts/run_glsim.py
```

The scripts expose their available options through `--help`. Full regeneration
is expensive; the released records are the canonical inputs to the CPU analyses
above.

## Human-audit release

`results/e4_label_audit/audit_labels.jsonl` contains anonymous item IDs, COCO
image IDs, target mentions, two independent labels, disagreement adjudication,
and final labels. It omits captions, model features, timestamps, annotator
identities, and local paths. The `rater_a_protocol` field records the disclosed
protocol variation for judgments imported from an earlier caption-visible
interface.

## Figures

Figure sources and generated assets are independent of the manuscript:

```bash
python scripts/make_layer_sweep_figure_corrected.py
python scripts/make_qual_figure.py
tectonic figures/label_fragility.tex --outdir assets/figures
```

## License

Code is released under the MIT License. COCO and Open Images retain their own
licenses. This repository distributes index metadata and derived result records,
not third-party images.
