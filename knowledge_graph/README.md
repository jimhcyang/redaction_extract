# Knowledge Graph

This folder is the only KG workflow you need.

It contains:

- `prepare_project.py`
- `stage_a_extract.py`
- `stage_a1_alias_resolution.py`
- `stage_b_graph.py`
- `stage_c_hierarchy.py`
- `run_pipeline.py`

The source corpus for the KG is:

- `postprocessed/*/document/unredacted.transcription.txt`

That is the cleaned unredacted text to build the graph from.

## Install

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## One Command Workflow

Set your OpenAI key first because Stage A and Stage A1 use it:

```bash
export OPENAI_API_KEY=your_key_here
```

Then run the full pipeline:

```bash
python knowledge_graph/run_pipeline.py \
  --project-dir outputs/kg_cia_project \
  --corpus-root postprocessed \
  --n-docs 25
```

That now runs:

1. project preparation
2. Stage A extraction
3. Stage A1 alias generation with OpenAI adjudication
4. Stage B graph build
5. Stage C hierarchy build

By default, Stage A extracts paragraph-by-paragraph. To make one OpenAI call per document instead, add:

```bash
--chunk-mode document
```

That produces:

- `outputs/kg_cia_project/extraction/`
- `outputs/kg_cia_project/graphs/`
- `outputs/kg_cia_project/hierarchy/`

For a tiny smoke test:

```bash
python knowledge_graph/run_pipeline.py \
  --project-dir outputs/kg_cia_project_test \
  --corpus-root postprocessed \
  --limit 10 \
  --n-docs 10 \
  --chunk-mode document
```

## Stage By Stage

Prepare the project input:

```bash
python knowledge_graph/prepare_project.py \
  --corpus-root postprocessed \
  --project-dir outputs/kg_cia_project
```

Run extraction:

```bash
python knowledge_graph/stage_a_extract.py \
  outputs/kg_cia_project \
  input_documents.csv \
  --start-at 0 \
  --n-docs 25
```

Build the Stage A1 alias-resolution worklist before Stage B:

```bash
python knowledge_graph/stage_a1_alias_resolution.py \
  outputs/kg_cia_project
```

OpenAI adjudication is on by default. To run heuristics only:

```bash
python knowledge_graph/stage_a1_alias_resolution.py \
  outputs/kg_cia_project \
  --skip-openai
```

For one extraction call per document:

```bash
python knowledge_graph/stage_a_extract.py \
  outputs/kg_cia_project \
  input_documents.csv \
  --start-at 0 \
  --n-docs 25 \
  --chunk-mode document
```

Do not switch `chunk-mode` inside the same project directory after extraction has started. Use a fresh project directory if you want to compare paragraph mode versus document mode.

Build graphs:

```bash
python knowledge_graph/stage_b_graph.py outputs/kg_cia_project
```

Build hierarchy:

```bash
python knowledge_graph/stage_c_hierarchy.py outputs/kg_cia_project
```

## Alias Resolution

The default workflow does not require a repo-level seed alias dictionary.

If you want to provide one, you can optionally place:

- `PROJECT_DIR/entity_directory/entity_aliases.csv`

or pass an explicit alias directory with `--entity-dir` to Stage A1 or Stage B.

Stage A1 also writes a stronger review set before Stage B under:

- `entity_resolution/label_catalog.csv`
- `entity_resolution/alias_candidates_all.csv`
- `entity_resolution/alias_candidates_final.csv`
- `entity_resolution/alias_candidates_review.csv`
- `entity_resolution/alias_openai_review_queue.csv`
- `entity_resolution/alias_auto_accepted.csv`
- `entity_resolution/alias_openai_accepted.csv`
- `entity_resolution/alias_openai_rejected.csv`
- `entity_resolution/alias_openai_unresolved.csv`
- `entity_resolution/alias_manual_review.csv`
- `entity_resolution/entity_aliases_proposed.csv`
- `entity_resolution/entity_aliases_generated.csv`

`entity_aliases_generated.csv` is the main working dictionary after Stage A1:

- it includes any optional project-local seed aliases if they exist
- plus the auto-accepted Stage A1 merges
- Stage B will load it automatically if it exists

Recommended flow:

1. Run Stage A.
2. Run Stage A1.
3. Use `entity_resolution/entity_aliases_generated.csv` as the working alias dictionary.
4. Inspect the audit files if you want to tune thresholds later:
   - `alias_auto_accepted.csv`
   - `alias_openai_review_queue.csv`
   - `alias_openai_accepted.csv`
   - `alias_openai_rejected.csv`
   - `alias_openai_unresolved.csv`
5. Run Stage B and Stage C.

## Results Notebook

For analyst-facing review after a run, use:

- `knowledge_graph/notebooks/results_viewer.ipynb`

It is read-only. Set `PROJECT_DIR` inside the notebook to the run you want to inspect, for example:

- `outputs/kg_cia_smoke_para50`
