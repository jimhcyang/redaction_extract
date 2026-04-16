# Redacted Document Analysis

This repository contains the current document curation and analysis workflow for paired redacted and unredacted document data. The active work is split into two small script packages:

- `transcription_scripts/`: OpenAI-based PDF transcription, page alignment, text diffing, and bracketed span output.
- `box_scripts/`: image/PDF redaction-box detection with overlay and JSON outputs.

The repo also contains the working document corpus, generated transcription artifacts, image inputs, and box-detection outputs.

## Repository Layout

```text
box_scripts/
  run_box_pipeline.py       # box detector CLI
  box_pipeline.py           # box detection and PDF/image handling
  requirements.txt

transcription_scripts/
  run_openai_docs_pipeline.py
  openai_transcribe_pdfs.py
  postprocess_transcriptions.py
  openai_pdf_pipeline.py
  requirements.txt

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
```

## Setup

From repo root, create the two virtual environments:

```cmd
python -m venv .venv_transcription
python -m venv .venv_box
```

Install dependencies:

```cmd
.venv_transcription\Scripts\python.exe -m pip install -r transcription_scripts\requirements.txt
.venv_box\Scripts\python.exe -m pip install -r box_scripts\requirements.txt
```

Set the OpenAI key in the terminal where transcription commands will run:

```cmd
set OPENAI_API_KEY=your_key_here
```

## Transcription And Diff Pipeline

Run the full OpenAI transcription, page alignment, local diffing, and bracketed span generation:

```cmd
.venv_transcription\Scripts\python.exe transcription_scripts\run_openai_docs_pipeline.py --docs_root docs --out_root transcriptions --model gpt-5.2 --skip_existing
```

Run only a small test batch:

```cmd
.venv_transcription\Scripts\python.exe transcription_scripts\run_openai_docs_pipeline.py --docs_root docs --out_root transcriptions_test5 --max_pairs 5 --model gpt-5.2 --skip_existing
```

Run postprocessing only after transcription files already exist:

```cmd
.venv_transcription\Scripts\python.exe transcription_scripts\postprocess_transcriptions.py --docs_root docs --out_root transcriptions --skip_existing
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

## Box Detection Pipeline

By default, the box detector uses the repo-root `images/` folder when no `--input` or `--docs_root` is provided:

```cmd
.venv_box\Scripts\python.exe box_scripts\run_box_pipeline.py --out box_outputs_images --save_debug_masks
```

Run on a specific input file or folder:

```cmd
.venv_box\Scripts\python.exe box_scripts\run_box_pipeline.py --input images --out box_outputs_images --save_debug_masks
```

Run on the paired PDF corpus:

```cmd
.venv_box\Scripts\python.exe box_scripts\run_box_pipeline.py --docs_root docs --source_kind redacted --out box_outputs_docs --max_files 10 --max_pages 2 --save_debug_masks
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

## Git LFS

Large binary data such as PDFs and images is tracked with Git LFS. After cloning, install LFS before pulling the full corpus:

```cmd
git lfs install
git lfs pull
```
