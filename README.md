# Redacted Document Analysis

This repository contains the current document curation and analysis workflow for paired redacted and unredacted document data. The active work is split into three script packages:

- `transcription_scripts/`: OpenAI-based PDF transcription, page alignment, text diffing, and bracketed span output.
- `box_scripts/`: image/PDF redaction-box detection with overlay and JSON outputs.
- `modeling_scripts/`: token-level forward redaction classifier built from cleaned `postprocessed/` labels.

The repo also contains the working document corpus, generated transcription artifacts, image inputs, and box-detection outputs.

## Repository Layout

```text
box_scripts/
  run_box_pipeline.py       # box detector CLI
  box_pipeline.py           # box detection and PDF/image handling

transcription_scripts/
  run_openai_docs_pipeline.py
  openai_transcribe_pdfs.py
  postprocess_transcriptions.py
  openai_pdf_pipeline.py

modeling_scripts/
  build_forward_dataset.py
  summarize_document_lengths.py
  train_forward_model.py
  plot_forward_results.py
  run_forward_cv.py
  run_modeling_pipeline.py

docs/
  cibcia.csv
  redacted_pdfs/
  unredacted_pdfs/

images/
  image inputs for box detection

transcriptions/
  full transcription/alignment/diff outputs

box_outputs_images/
  box detector outputs for images/

requirements.txt
```

## Setup

Use one shared virtual environment from the repo root.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```cmd
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows cmd:

```cmd
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the OpenAI key in the terminal where transcription commands will run:

```cmd
set OPENAI_API_KEY=your_key_here
```

On macOS or Linux shells such as `zsh`:

```bash
export OPENAI_API_KEY=your_key_here
```

The key must include file-upload permission. The earlier failed runs in this repo
were mostly `401` errors with `Missing scopes: api.files.write`.

## Transcription And Diff Pipeline

Run the full OpenAI transcription, page alignment, local diffing, and bracketed span generation:

```bash
python transcription_scripts/run_openai_docs_pipeline.py --docs_root docs --out_root transcriptions --model gpt-5.4-mini --skip_existing
```

Run only a small test batch:

```bash
python transcription_scripts/run_openai_docs_pipeline.py --docs_root docs --out_root transcriptions_test5 --max_pairs 5 --model gpt-5.4-mini --skip_existing
```

Run postprocessing only after transcription files already exist:

```bash
python transcription_scripts/postprocess_transcriptions.py --docs_root docs --out_root transcriptions --skip_existing
```

Important output per pair:

```text
transcriptions/<pair_key>/document/
  redacted.transcription.txt
  unredacted.transcription.txt
  pair_metadata.json

transcriptions/<pair_key>/difference/
  redacted.aligned.txt
  unredacted.aligned.txt
  unredacted_bracketed.aligned.txt
  redaction_chunks.aligned.txt
  page_alignment.json
  difference.summary.json
```

The bracketed output serializes predicted spans across the whole document:

```text
[[PRED_REDACTION_1]]example text[[/PRED_REDACTION_1]]
[[PRED_REDACTION_2]]more text[[/PRED_REDACTION_2]]
```

## Redaction Label Cleanup

After transcription and diffing, run the high-precision cleanup pass:

```bash
python transcription_scripts/postprocess_redaction_labels.py --in_root transcriptions --out_root postprocessed
```

The cleanup rejects full-page/admin-page candidates, page furniture,
standalone dates, map/legend labels, one-token spans, OCR-equivalent
differences, and non-empty redacted-side alternate conflicts. Full-page/admin
filtering uses `full_page_coverage = 0.95` with `min_full_page_tokens = 50`,
so short paragraph-length pages can remain useful training examples while
larger page-wide mismatches are rejected. The alternate-conflict rule
separately catches short candidate spans, long mismatched alternates, and
`alternate_near_full_page_coverage = 0.90` boundary cases where the redacted
scan produced competing OCR text instead of a clean gap.

The current restored corpus produces 40,360 post-recut candidate segments:

| Token bucket | Candidate segments | Kept | Excluded | Exclusion rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10,539 | 0 | 10,539 | 100.00% |
| 2 | 11,285 | 216 | 11,069 | 98.09% |
| 3-5 | 2,792 | 780 | 2,012 | 72.06% |
| 6-10 | 4,594 | 1,522 | 3,072 | 66.87% |
| 11-20 | 1,751 | 1,232 | 519 | 29.64% |
| 21-50 | 2,395 | 1,696 | 699 | 29.19% |
| 51-100 | 2,634 | 1,466 | 1,168 | 44.34% |
| 101+ | 4,370 | 1,001 | 3,369 | 77.09% |
| **Total** | **40,360** | **7,913** | **32,447** | **80.39%** |

## Forward Modeling

Full terminal sequence from a fresh shell:

```bash
cd /Users/jimyhc/Desktop/research/IARPA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python modeling_scripts/run_modeling_pipeline.py --postprocessed-root postprocessed --output-root modeling_outputs --epochs 3 --cv-folds 5
```

The first training command downloads `answerdotai/ModernBERT-base` from
Hugging Face if it is not already cached. Training uses the first 4,096
ModernBERT tokens per document to fit the local memory window. Training always
selects the fastest available device in this order: CUDA, then Apple MPS, then
CPU. The CV command trains five additional models, so it is the slowest step.

To run the main train/test pass without the five-fold CV step:

```bash
python modeling_scripts/run_modeling_pipeline.py --postprocessed-root postprocessed --output-root modeling_outputs --epochs 3 --skip-cv
```

Build token-level labels directly from the cleaned `postprocessed/` spans:

```bash
python3 modeling_scripts/build_forward_dataset.py --postprocessed-root postprocessed --output-dir modeling_outputs/forward_dataset --cv-folds 5
```

Summarize document lengths against the training memory window and
write histogram figures:

```bash
python modeling_scripts/summarize_document_lengths.py --dataset-jsonl modeling_outputs/forward_dataset/documents.jsonl --output-dir modeling_outputs/forward_dataset
```

This writes `document_lengths.csv`, `document_length_summary.json`, and
histograms under `modeling_outputs/forward_dataset/figures/`.

Current corpus length summary at the 4,096-token training cutoff:

| Count type | p50 | p90 | p95 | p99 | Max | Docs over 4,096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset word tokens | 1,307 | 2,913 | 3,401 | 4,280 | 6,130 | 31 |
| ModernBERT subtokens | 1,884 | 4,199 | 4,907 | 6,117 | 8,974 | 264 |

At this cutoff, 4,350 of 341,230 positive word labels fall outside the first
window. At the span level, 7,783 redaction spans are fully inside the cutoff,
7 cross the cutoff, and 122 are fully outside.

Train the long-context forward classifier:

```bash
python modeling_scripts/train_forward_model.py --dataset-dir modeling_outputs/forward_dataset --output-dir modeling_outputs/forward_model --epochs 3 --save-model
```

The forward model is fixed to `answerdotai/ModernBERT-base`, with a
4,096-token first-window training policy for local memory stability.

Run document-level cross-validation:

```bash
python modeling_scripts/run_forward_cv.py --dataset-dir modeling_outputs/forward_dataset --output-root modeling_outputs/forward_cv --folds 5 --epochs 3
```

Generate evaluation figures:

```bash
python modeling_scripts/plot_forward_results.py --dataset-summary modeling_outputs/forward_dataset/dataset_summary.json --training-summary modeling_outputs/forward_model/training_summary.json --test-predictions modeling_outputs/forward_model/test_word_predictions.jsonl
```

## Box Detection Pipeline

By default, the box detector uses the repo-root `images/` folder when no `--input` or `--docs_root` is provided:

```bash
python box_scripts/run_box_pipeline.py --out box_outputs_images --save_debug_masks
```

Run on a specific input file or folder:

```bash
python box_scripts/run_box_pipeline.py --input images --out box_outputs_images --save_debug_masks
```

Run on the paired PDF corpus:

```bash
python box_scripts/run_box_pipeline.py --docs_root docs --source_kind redacted --out box_outputs_docs --max_files 10 --max_pages 2 --save_debug_masks
```

Each processed image/page writes:

```text
<out>/<item_key>/
  <item_key>.source.png
  <item_key>.redaction_boxes.png
  <item_key>.redaction_boxes.json
  debug/
```

## Modeling Direction

The curated outputs are being prepared for token/span-level modeling. The forward model estimates whether each token is a target span given full document context:

```text
p_theta(m_i = 1 | x)
```

The inverse pipeline will generate candidate fills for missing spans, insert them into context, and score them with a combination of language plausibility, learned targeting likelihood, length/layout evidence, and metadata priors:

```text
score(y_k) =
  log p_gen(y_k | context)
+ lambda * log p_theta(target span | x[y_k])
+ beta * log p_len(length(y_k) | observed mask)
+ gamma * log p_meta(y_k | document metadata)
```

Candidates can then be normalized into a posterior:

```text
P(y_k | evidence) = softmax(score(y_k))
```
