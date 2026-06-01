# Knowledge Graph

This folder is the only KG workflow you need.

It contains:

- `_prepare_project.py`
- `stage_a_extract.py`
- `stage_a1_context_catalog.py`
- `stage_a2_rule_resolution.py`
- `stage_a3_sense_clustering.py`
- `stage_a4_llm_resolution.py`
- `stage_a5_consolidation.py`
- `entity_resolution/`
- `entity_resolution/common.py`
- `entity_resolution/scoring.py`
- `entity_resolution/a1_context.py`
- `entity_resolution/a2_rule_resolution.py`
- `entity_resolution/a3_sense_clustering.py`
- `entity_resolution/a4_llm_resolution.py`
- `entity_resolution/a5_consolidation.py`
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

Set your OpenAI key first because Stage A, Stage A4, and Stage A5 use it:

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
3. Stage A1 context cataloging
4. Stage A2 rule resolution
5. Stage A3 sense clustering
6. Stage A4 LLM resolution
7. Stage A5 canonical consolidation
8. Stage B graph build
9. Stage C hierarchy build

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
python knowledge_graph/_prepare_project.py \
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

Run the post-extraction entity-resolution stages before Stage B:

```bash
python knowledge_graph/stage_a1_context_catalog.py outputs/kg_cia_project
python knowledge_graph/stage_a2_rule_resolution.py outputs/kg_cia_project
python knowledge_graph/stage_a3_sense_clustering.py outputs/kg_cia_project
python knowledge_graph/stage_a4_llm_resolution.py outputs/kg_cia_project
python knowledge_graph/stage_a5_consolidation.py outputs/kg_cia_project
```

OpenAI adjudication is on by default. To stop after A1/A2/A3 and keep A4 unresolved clusters separate:

```bash
python knowledge_graph/stage_a4_llm_resolution.py \
  outputs/kg_cia_project \
  --skip-openai
```

To run A5 locally without extra consolidation adjudication:

```bash
python knowledge_graph/stage_a5_consolidation.py \
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

## Stage A1-A5 Outputs

The `A1/A2/A3/A4/A5` scripts together are now the corpus-specific canonicalization workflow. They write the durable files that Stage B consumes:

- `entity_resolution/entity_canonical_map.csv`
- `entity_resolution/entities_resolved.csv`
- `entity_resolution/entity_alias_map_for_notebook_b.csv`
- `entity_resolution/entity_resolution_summary.json`

It also writes supporting audit files:

- `entity_resolution/a1_mention_catalog.csv`
- `entity_resolution/a2_rule_aliases.csv`
- `entity_resolution/a2_rule_resolved_entities.csv`
- `entity_resolution/a2_ambiguous_labels.csv`
- `entity_resolution/a3_sense_assignments.csv`
- `entity_resolution/a3_sense_clusters.csv`
- `entity_resolution/a3_cluster_candidates.csv`
- `entity_resolution/a4_cluster_review_queue.csv`
- `entity_resolution/a4_cluster_decisions.csv`
- `entity_resolution/a5_canonical_catalog.csv`
- `entity_resolution/a5_candidate_pairs.csv`
- `entity_resolution/a5_review_queue.csv`
- `entity_resolution/a5_pair_decisions.csv`
- `entity_resolution/a5_consolidation_map.csv`
- `entity_resolution/a5_entities_resolved.csv`
- `entity_resolution/a5_entity_canonical_map.csv`
- `entity_resolution/entity_inventory.csv`
- `entity_resolution/entity_resolution_llm_adjudicated.csv`
- `entity_resolution/alias_auto_accepted.csv`
- `entity_resolution/a4_cluster_adjudications.jsonl`
- `entity_resolution/a5_adjudications.jsonl`
- `entity_resolution/entity_aliases_generated.csv`

Optional high-confidence seed overrides can be provided at:

- `PROJECT_DIR/config/entity_alias_overrides.csv`

Recommended flow:

1. Run Stage A.
2. Run `stage_a1_context_catalog.py`, `stage_a2_rule_resolution.py`, `stage_a3_sense_clustering.py`, `stage_a4_llm_resolution.py`, and `stage_a5_consolidation.py` to produce the final `entities_resolved.csv` and `entity_canonical_map.csv`.
3. Stage B consumes the mention-level `entities_resolved.csv` first, with `entity_canonical_map.csv` as a label-level fallback.
4. Inspect the A1/A2/A3/A4/A5 audit files if you want to tune thresholds later.
5. Run Stage C.

Stage B also writes:

- `graphs/canonical_entity_mentions.csv`

That file lets you trace each canonical entity node back to its original `doc_id`, `paragraph_id`, `mention_key`, raw label, and description.

## Results Notebook

For analyst-facing review after a run, use:

- `knowledge_graph/notebooks/results_viewer.ipynb`

It is read-only. Set `PROJECT_DIR` inside the notebook to the run you want to inspect, for example:

- `outputs/kg_cia_smoke_para50`
