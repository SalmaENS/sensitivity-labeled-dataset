# Sensitivity-Labeled Dataset for Sensitivity-Aware Deepfake Detection

A dataset of **58,442 real, fake, and tampered images** spanning content from everyday scenes to highly sensitive, high-stakes imagery, each labeled with a **Content Sensitivity Score (CSS, 1–4)** via a Mixture-of-Agents VLM pipeline. Built to study how content sensitivity affects general-purpose VLMs' ability to distinguish authentic, fully generated, and partially manipulated images.

Master 1 internship project — ENS Paris-Saclay (SIEN Department), carried out at [ALCOR Lab](https://alcorlab.diag.uniroma1.it/), Sapienza University of Rome, under the supervision of Prof. Irene Amerini and Simone Teglia.

## Overview

The project has three phases:

1. **Image collection** — sample real/fake/tampered images from four source datasets (DGM4, SID-Set, RRDataset, Sens-VisualNews) to cover both authenticity and content sensitivity.
2. **Content Sensitivity Scoring (CSS)** — score every image 1–4 using a Mixture-of-Agents pipeline: three VLM proposers (Pixtral-12B, Qwen2.5-VL-7B, InternVL3-8B) independently score each image, and a text-only aggregator (Qwen2.5-7B-Instruct) synthesizes a final label.
3. **VLM benchmarking** — evaluate general-purpose VLMs (Qwen2.5-VL-7B, InternVL3-8B) on real/fake/tampered classification, broken down by CSS level, using a new Content-Aware Accuracy (caAcc) metric.

## Repository structure

```
.
├── collect_samples.py         # Phase 1 — image sampling from the 4 source datasets
├── proposer_pixtral.py        # Phase 2 — CSS proposer: Pixtral-12B
├── proposer_qwen.py           # Phase 2 — CSS proposer: Qwen2.5-VL-7B-Instruct
├── proposer_internvl.py       # Phase 2 — CSS proposer: InternVL3-8B
├── aggregator.py              # Phase 2 — CSS aggregator: Qwen2.5-7B-Instruct (text-only)
├── sample_uniform_dataset.py  # Phase 2 — draws a balanced 30k subset for benchmarking
├── benchmark_small.py         # Phase 3 — VLM benchmarking (Qwen2.5-VL-7B, InternVL3-8B)
├── merge_model_results.py     # Phase 3 — merges per-model/per-shard benchmark CSVs
└── results_merged.csv         # Final merged benchmark results
```

## Requirements

- Python 3.10+
- CUDA GPU(s) — the CSS-scoring and benchmarking stages run VLM inference and need a GPU per model (see each script's `--gpu` flag).
- A [Hugging Face](https://huggingface.co/) account/token (`HF_TOKEN`) for streaming DGM4, RRDataset, and SID-Set from the Hub.
- A local copy of VisualNews's `origin.tar` (~91 GB, from the [VisualNews project page](https://www.cs.rice.edu/~vo9/visualnews/)) if you want to include Sens-VisualNews.

Install dependencies:

```bash
pip install torch torchvision transformers accelerate datasets huggingface_hub \
            pillow tqdm qwen-vl-utils
# Pixtral proposer only:
pip install mistral_inference mistral_common
# optional, faster attention (InternVL3 / Qwen2.5-VL):
pip install flash-attn --no-build-isolation
```

Set your Hugging Face token once per session:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

> All scripts below default to author-specific paths (e.g. `/mnt/ssd1/bairat/...`). Every path is overridable via CLI flags — the commands below use `./data/...` as a portable example; substitute your own paths.

---

## Phase 1 — Image collection

Samples real/fake/tampered images from DGM4, SID-Set, RRDataset, and Sens-VisualNews into a JPEG folder plus an incrementally-written `manifest.json`. Fully resumable — safe to interrupt and rerun; already-collected images are skipped.

The counts below reproduce the corpus composition used in this project (Table 1.1 — 58,442 images total: 20,000 real / 18,442 fake / 20,000 tampered):

```bash
python collect_samples.py \
    --output_dir ./data/dataset/raw \
    --metadata_path ./data/metadata/manifest.json \
    --hf_token $HF_TOKEN \
    --seed 42 \
    --n_dgm4_real 500 \
    --n_dgm4_tampered 19000 \
    --n_sidset_fake 10000 \
    --n_sidset_tampered 1000 \
    --n_rrdataset_real 10000 \
    --n_rrdataset_fake 8442 \
    --n_sensvisualnews_real 9500 \
    --sens_visualnews_tar /path/to/visualnews/origin.tar
```

Useful flags: `--skip_rrdataset` / `--skip_dgm4` / `--skip_sidset` / `--skip_sensvisualnews` to skip a source entirely (e.g. if you don't have the VisualNews tar, add `--skip_sensvisualnews` and drop `--n_sensvisualnews_real`).

---

## Phase 2 — Content Sensitivity Scoring (Mixture-of-Agents)

### Step 1 — run the three proposers

Each proposer depends on different (sometimes conflicting) library versions, so they're designed to run independently — in separate environments/containers, and ideally on separate GPUs in parallel. All support `--resume` to only (re)score images that are missing or failed previously.

```bash
python proposer_pixtral.py \
    --manifest ./data/metadata/manifest.json \
    --images_dir ./data/dataset/raw \
    --output ./data/css_scores/pixtral.json \
    --gpu 0 \
    --resume

python proposer_qwen.py \
    --manifest ./data/metadata/manifest.json \
    --images_dir ./data/dataset/raw \
    --output ./data/css_scores/qwen.json \
    --gpu 1 \
    --resume

python proposer_internvl.py \
    --manifest ./data/metadata/manifest.json \
    --images_dir ./data/dataset/raw \
    --output ./data/css_scores/internvl.json \
    --gpu 2 \
    --resume
```

Progress and per-image logs are written to `<output>.log` (e.g. `pixtral.log`), not the console — tail that file to watch progress. Approximate runtime for the full 58,442-image corpus: Qwen2.5-VL-7B ≈ 2 days, InternVL3-8B ≈ 2.5 days, Pixtral-12B ≈ 3.5 days (single GPU each; Pixtral uses 2 GPUs).

### Step 2 — aggregate into a final CSS label

```bash
python aggregator.py \
    --manifest ./data/metadata/manifest.json \
    --pixtral_scores ./data/css_scores/pixtral.json \
    --qwen_scores ./data/css_scores/qwen.json \
    --internvl_scores ./data/css_scores/internvl.json \
    --output ./data/metadata/manifest_css.json \
    --gpu 0 \
    --resume
```

This produces `manifest_css.json`: the original manifest with each entry's `css_proposer_responses`, `css_aggregated_response`, and `final_css_score` filled in — the ground-truth sensitivity label used from here on.

### Step 3 — draw a balanced subset for benchmarking

`manifest_css.json`'s raw CSS distribution is neither balanced across labels nor CSS levels. `sample_uniform_dataset.py` draws a subset with equal real/fake/tampered totals, balanced across CSS levels as evenly as each label's availability allows (30,000 images used in this project: 10,000 per label):

```bash
python sample_uniform_dataset.py \
    --manifest ./data/metadata/manifest_css.json \
    --images-dir ./data/dataset/raw \
    --output-images-dir ./data/dataset/uniform \
    --output-manifest ./data/metadata/manifest_css_uniform.json \
    --target-total 30000 \
    --seed 42
```

---

## Phase 3 — VLM benchmarking

Evaluates VLMs on real/fake/tampered classification across CSS levels and reports three tables (real, fake, tampered), each with per-CSS-level accuracy plus two Content-Aware Accuracy summaries (`caAcc_linear`, `caAcc_fib`):

```bash
python benchmark_small.py \
    --manifest ./data/metadata/manifest_css_uniform.json \
    --images-dir ./data/dataset/uniform \
    --models Qwen-2.5VL-7B,InternVL3-8B \
    --gpu 0 \
    --output-csv ./results/benchmark_results.csv
```

This writes three files: `benchmark_results_real.csv`, `benchmark_results_fake.csv`, `benchmark_results_tampered.csv`.

**To run each model on its own GPU in parallel instead** (recommended for two GPUs):

```bash
python benchmark_small.py --manifest ./data/metadata/manifest_css_uniform.json \
    --images-dir ./data/dataset/uniform --models Qwen-2.5VL-7B \
    --gpu 0 --output-csv ./results/qwen.csv &

python benchmark_small.py --manifest ./data/metadata/manifest_css_uniform.json \
    --images-dir ./data/dataset/uniform --models InternVL3-8B \
    --gpu 1 --output-csv ./results/internvl.csv &

wait
```

Then merge the per-model shard CSVs into one combined results file (the repository's `results_merged.csv` was produced this way):

```bash
python merge_model_results.py \
    results/qwen_real.csv results/qwen_fake.csv results/qwen_tampered.csv \
    results/internvl_real.csv results/internvl_fake.csv results/internvl_tampered.csv \
    --output results/results_merged.csv
```

---

## Results

Accuracy (%) on real/fake/tampered classification, by CSS level, from `results_merged.csv`:

**Real images**

| Model | CSS 1 | CSS 2 | CSS 3 | CSS 4 | caAcc_linear | caAcc_fib |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 98.2 | 96.3 | 98.2 | 96.2 | 97.0 | 96.9 |
| InternVL3-8B | 78.6 | 74.5 | 76.3 | 73.9 | 75.2 | 75.1 |

**Fake images (fully synthetic)**

| Model | CSS 1 | CSS 2 | CSS 3 | CSS 4 | caAcc_linear | caAcc_fib |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 2.7 | 7.6 | 3.3 | 17.0 | 9.6 | 10.3 |
| InternVL3-8B | 25.0 | 30.9 | 14.6 | 42.6 | 30.1 | 31.2 |

**Tampered images**

| Model | CSS 1 | CSS 2 | CSS 3 | CSS 4 | caAcc_linear | caAcc_fib |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 1.0 | 1.8 | 3.3 | 2.1 | 2.3 | 2.3 |
| InternVL3-8B | 22.7 | 39.3 | 42.6 | 56.2 | 45.4 | 46.4 |

Both models are biased toward predicting *real*, to very different degrees; InternVL3-8B trades real-image accuracy for far better manipulation detection, making it the more trustworthy model overall. Content sensitivity has an opposite-direction effect on the two failure modes: real-image accuracy declines slightly as CSS increases, while fake/tampered detection accuracy rises — most dramatically for InternVL3-8B on tampered images (22.7% → 56.2%). See the full report for discussion.

## Dataset composition

| Source Dataset | Real | Fake | Tampered | Total | % of Total |
|---|---|---|---|---|---|
| DGM4 | 500 | — | 19,000 | 19,500 | 33.4% |
| SID-Set | — | 10,000 | 1,000 | 11,000 | 18.8% |
| RRDataset | 10,000 | 8,442 | — | 18,442 | 31.6% |
| Sens-VisualNews | 9,500 | — | — | 9,500 | 16.3% |
| **Total** | **20,000** | **18,442** | **20,000** | **58,442** | **100.0%** |

## License notes

This project combines images from four source datasets under different licenses :

- **DGM4** — S-Lab License 1.0 (non-commercial only).
- **SID-Set** — CC-BY-4.0 (attribution required, commercial use allowed); incorporates COCO / OpenImages V7 / Flickr30k content.
- **RRDataset** — CC-BY-NC-4.0 (attribution required, non-commercial only).
- **Sens-VisualNews** — code/annotations are academic/non-commercial use only; underlying images are real photojournalism from news outlets, sourced via VisualNews — verify VisualNews's own terms before redistributing raw images.

This code and manifest metadata are provided for research/academic purposes; the images themselves remain subject to their original datasets' licenses.

## Citing the source datasets

```
DGM4:            Shao, Wu, Liu. "Detecting and Grounding Multi-Modal Media Manipulation." CVPR 2023. arXiv:2304.02556
SID-Set:         Huang et al. "SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model." CVPR 2025. arXiv:2412.04292
RRDataset:       Li et al. "Bridging the Gap Between Ideal and Real-world Evaluation." arXiv:2509.09172 (2025)
Sens-VisualNews: Goulas et al. "Sens-VisualNews: A Benchmark Dataset for Sensational Image Detection." arXiv:2605.10394 (2026)
```

## Acknowledgements

Carried out at ALCOR Lab, Sapienza University of Rome, under the supervision of Prof. Irene Amerini and Simone Teglia, as part of a Master 1 internship at ENS Paris-Saclay (SIEN Department).
