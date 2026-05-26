
# ==============================================================================
# # Notebook A — Scalable Production Extraction Layer
# 
# This notebook is the **extraction stage** of the Matryoshka Knowledge Graph system.
# 
# It is designed for scaling from a small test run, such as 200 documents, to larger runs such as **2,000** or **5,000** documents.
# 
# The notebook extracts a controlled knowledge graph from documents:
# 
# \[
# \text{Documents} \rightarrow \text{Entities, Events, Claims, Relations}
# \]
# 
# The output of this notebook is used downstream by the graph construction and hierarchy notebooks.
# 
# ## Main design goals
# 
# 1. **Stable schema** so later graph notebooks do not break.
# 2. **Drive-first checkpointing** so Colab runtime resets do not destroy progress.
# 3. **Batch execution** so large corpora can be processed over multiple sessions.
# 4. **Strict JSON validation** so malformed model outputs are caught safely.
# 5. **Error recovery** so one bad chunk does not stop the whole run.
# 6. **Cost/time monitoring** so large extraction jobs can be estimated and controlled.

# ==============================================================================
# ## 0. How to use this notebook
# 
# For a first run on a new corpus:
# 
# 1. Run from terminal or tmux.
#   Set PROJECT_DIR and CSV filename as command-line arguments.
# 2. Set `PROJECT_DIR`.
# 3. Set `SOURCE_CSV`.
# 4. Start with a small sample:
#    ```python
#    N_DOCS = 20
#    MAX_NEW_CHUNKS_THIS_RUN = 200
#    RUN_EXTRACTION = True
#    ```
# 5. Inspect outputs and errors.
# 6. Increase to 500 documents per run.
# 7. Continue in batches until all documents are processed.
# 
# For your 2,000/5,000 document runs, the recommended strategy is:
# 
# ```text
# Run 250--500 documents per Colab session,
# or set MAX_NEW_CHUNKS_THIS_RUN to a safe cap,
# then resume from checkpoint.
# ```
# 
# The checkpoint file is append-only and stored in Drive, so the notebook can safely resume after interruptions.

# ==============================================================================
# ## 1. Install and import dependencies

# ------------------------------------------------------------------------------

import os
import re
import json
import time
import shutil
import random
import hashlib
import traceback
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from tqdm.auto import tqdm

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    print("OpenAI package is not available. Install with: pip install openai")

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)


# ---------------------------------------------------------------------
# PROJECT DIRECTORY
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "PROJECT_DIR",
    help="Project folder"
)

parser.add_argument(
    "csv_name",
    help="Input CSV filename (relative to PROJECT_DIR)"
)

parser.add_argument(
    "--start-at",
    type=int,
    default=0,
    help="Row index to start processing"
)

parser.add_argument(
    "--n-docs",
    type=int,
    default=None,
    help="Number of documents to process"
)

parser.add_argument(
    "--max-new-chunks",
    type=int,
    default=None,
    help="Optional cap on new chunks processed in this run. Defaults to all pending chunks."
)

parser.add_argument(
    "--show-sample-prompt",
    action="store_true",
    help="Print a sample extraction prompt before running."
)

parser.add_argument(
    "--chunk-mode",
    choices=["paragraph", "document"],
    default="paragraph",
    help="Extraction unit for Stage A. 'paragraph' is safer; 'document' makes one LLM call per document."
)

args = parser.parse_args()

# Store project_dir in a variable for repeated use
PROJECT_DIR = Path(args.PROJECT_DIR).expanduser().resolve()
print("PROJECT_DIR",PROJECT_DIR)


# 1. Capture the inputs
SOURCE_CSV = PROJECT_DIR / args.csv_name
print("SOURCE_CSV:", SOURCE_CSV)

# Folder for extraction outputs.
EXTRACTION_DIR = PROJECT_DIR / "extraction"
RUN_DIR = PROJECT_DIR / "extraction_run"

EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_JSONL = RUN_DIR / "chunk_results.jsonl"
ERROR_JSONL = RUN_DIR / "chunk_errors.jsonl"
RUN_SUMMARY_JSONL = RUN_DIR / "run_summary.jsonl"

BACKUP_DIR = RUN_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

print("CHECKPOINT_JSONL:", CHECKPOINT_JSONL)
print("ERROR_JSONL:", ERROR_JSONL)

# ---------------------------------------------------------------------
# INPUT CSV
# ---------------------------------------------------------------------
# The source CSV should contain at least:
#   doc_id or id
#   body
#
# Optional:
#   date
#   subject
#   tag
#   concepts
#
# Change this path to your input CSV.
#SOURCE_CSV = PROJECT_DIR / "input_documents.csv"

print("PROJECT_DIR:", PROJECT_DIR)
print("RUN_DIR:", RUN_DIR)
print("SOURCE_CSV:", SOURCE_CSV)


# ==============================================================================
# ## 3. Run switches and scaling controls

# ------------------------------------------------------------------------------
# Code cell 7
# ---------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------
# For large-scale extraction, start with a cheaper reliable model.
# Use a stronger model for audits or difficult/error chunks if needed.
MODEL_NAME = "gpt-4o-mini"

# ---------------------------------------------------------------------
# RUN CONTROL
# ---------------------------------------------------------------------
RUN_EXTRACTION = True       # Set True when ready to call the API.
OVERWRITE_EXISTING = False  # True deletes current checkpoint files. Use with caution.

# Run only chunks that previously failed.
RUN_FAILED_CHUNKS_ONLY = False

# If True, chunks that previously failed will be skipped in normal mode.
# If False, they may be retried in later runs.
SKIP_PREVIOUS_ERRORS = True

# Document slice. For large runs, use batches:
#   START_AT = 0,    N_DOCS = 500
#   START_AT = 500,  N_DOCS = 500
#   START_AT = 1000, N_DOCS = 500
START_AT = args.start_at
N_DOCS=args.n_docs
print("START_AT:", START_AT)
print("N_DOCS:", N_DOCS)

CHUNK_MODE = args.chunk_mode
print("CHUNK_MODE:", CHUNK_MODE)

# Hard cap on new chunks processed in this session.
# If omitted, process all pending chunks for the selected document slice.
MAX_NEW_CHUNKS_THIS_RUN = args.max_new_chunks
print("REQUESTED_MAX_NEW_CHUNKS_THIS_RUN:", MAX_NEW_CHUNKS_THIS_RUN)

# Sleep between calls. Increase if rate-limited.
SLEEP_SECONDS = 0.05

# Logging and backup frequency.
LOG_EVERY = 25
BACKUP_EVERY = 250

# ---------------------------------------------------------------------
# CHUNKING CONTROLS
# ---------------------------------------------------------------------
MIN_CHARS_PER_CHUNK = 80
MAX_CHARS_PER_CHUNK = 3000

# If a paragraph is longer than MAX_CHARS_PER_CHUNK, it will be split.
PARAGRAPH_SPLIT_OVERLAP = 150

# ---------------------------------------------------------------------
# ERROR STORAGE
# ---------------------------------------------------------------------
# Store full chunk text in error file.
# This helps debugging, but can make error logs large.
STORE_FULL_TEXT_IN_ERRORS = True

EXTRACTION_UNIT_LABEL = "documents" if CHUNK_MODE == "document" else "chunks"

# ---------------------------------------------------------------------
# FILES
# ---------------------------------------------------------------------




# ==============================================================================
# ## 4. Load source documents

# ------------------------------------------------------------------------------
# Code cell 9
if not SOURCE_CSV.exists():
    raise FileNotFoundError(
        f"Cannot find SOURCE_CSV: {SOURCE_CSV}\n"
        "Upload/copy your document CSV to this path, or change SOURCE_CSV."
    )

df = pd.read_csv(SOURCE_CSV)

print("Loaded source documents:", df.shape)
print("Columns:", df.columns.tolist())

required_cols = {"body"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# If doc_id is missing, create one from id or row number.
if "doc_id" not in df.columns:
    if "id" in df.columns:
        df["doc_id"] = df["id"].astype(str)
    else:
        df["doc_id"] = [f"DOC_{i:06d}" for i in range(len(df))]

if "id" not in df.columns:
    df["id"] = df["doc_id"].astype(str)

for col in ["subject", "date", "tag", "concepts"]:
    if col not in df.columns:
        df[col] = ""

display(df.head(3))


# ==============================================================================
# ## 5. Optional sampling for test runs

# ------------------------------------------------------------------------------
# Code cell 11
# For a random small test, set RANDOM_SAMPLE_N to an integer.
# For production, keep RANDOM_SAMPLE_N = None.
RANDOM_SAMPLE_N = None
RANDOM_SEED = 42

if RANDOM_SAMPLE_N is not None:
    work_source_df = df.sample(n=min(RANDOM_SAMPLE_N, len(df)), random_state=RANDOM_SEED).copy()
else:
    if N_DOCS is None:
        work_source_df = df.iloc[START_AT:].copy()
    else:
        work_source_df = df.iloc[START_AT:START_AT + N_DOCS].copy()

print("Documents selected for this run:", work_source_df.shape)
display(work_source_df[["doc_id", "id", "subject", "date"]].head())


# ==============================================================================
# ## 6. Chunking utilities

# ------------------------------------------------------------------------------
# Code cell 13
def is_effectively_empty(text):
    if text is None or pd.isna(text):
        return True
    text = str(text).strip()
    if not text:
        return True
    # Remove punctuation/whitespace and check remaining length.
    core = re.sub(r"[\W_]+", "", text)
    return len(core) < 10


def split_long_text(text, max_chars=MAX_CHARS_PER_CHUNK, overlap=PARAGRAPH_SPLIT_OVERLAP):
    """
    Split very long text into overlapping character windows.
    This is a fallback for long paragraphs.
    """
    text = str(text)
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHARS_PER_CHUNK:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def split_into_paragraph_chunks(body):
    """
    Paragraph-first chunking:
    1. Split on blank lines.
    2. Merge small paragraphs.
    3. Split overly long paragraphs.
    4. Skip tiny boilerplate chunks.
    """
    if is_effectively_empty(body):
        return []

    body = str(body).replace("\r\n", "\n").replace("\r", "\n")

    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]

    # If no blank-line paragraphs, fall back to line chunks.
    if len(raw_paragraphs) <= 1:
        raw_paragraphs = [p.strip() for p in body.split("\n") if p.strip()]

    chunks = []
    buffer = ""

    for para in raw_paragraphs:
        if len(para) > MAX_CHARS_PER_CHUNK:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(split_long_text(para))
            continue

        if not buffer:
            buffer = para
        elif len(buffer) + 2 + len(para) <= MAX_CHARS_PER_CHUNK:
            buffer += "\n\n" + para
        else:
            chunks.append(buffer.strip())
            buffer = para

    if buffer.strip():
        chunks.append(buffer.strip())

    # Final cleanup
    chunks = [c for c in chunks if len(c.strip()) >= MIN_CHARS_PER_CHUNK and not is_effectively_empty(c)]
    return chunks


def split_into_document_units(body):
    """
    Whole-document mode:
    1. Preserve the document as a single extraction unit.
    2. Skip empty or near-empty documents.
    """
    if is_effectively_empty(body):
        return []

    body = str(body).replace("\r\n", "\n").replace("\r", "\n").strip()
    if is_effectively_empty(body):
        return []
    return [body]


def split_text_units(body):
    if CHUNK_MODE == "document":
        return split_into_document_units(body)
    return split_into_paragraph_chunks(body)


def make_unit_id(index):
    if CHUNK_MODE == "document":
        return f"doc{index}"
    return f"p{index}"


# Quick smoke test
example_text = "Paragraph one.\n\nParagraph two with more content."
print(split_into_paragraph_chunks(example_text))


# ==============================================================================
# ## 7. Controlled ontology

# ------------------------------------------------------------------------------
# Code cell 15
# ---------------------------------------------------------------------
# Keep this ontology stable across large runs.
#
# If you change relation types midway, downstream graph behavior changes.
# Add new relation types only after auditing a sample.
# ---------------------------------------------------------------------

ALLOWED_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Government",
    "Location",
    "Country",
    "Military",
    "ReligiousGroup",
    "PoliticalGroup",
    "Document",
    "Other",
]

ALLOWED_EVENT_TYPES = [
    "Meeting",
    "Communication",
    "Demonstration",
    "Violence",
    "Arrest",
    "Travel",
    "Speech",
    "Negotiation",
    "PolicyAction",
    "MilitaryAction",
    "EconomicAction",
    "Other",
]

ALLOWED_RELATIONS = [
    "EVENT_PARTICIPANT",
    "EVENT_LOCATION",
    "EVENT_TIME",
    "AFFILIATION",
    "SUPPORT",
    "OPPOSITION",
    "NEGOTIATION",
    "REQUEST",
    "AGREEMENT",
    "COMMUNICATION",
    "TEMPORAL",
    "CAUSAL",
    "CLAIM_SPEAKER",
    "CLAIM_TARGET",
    "MENTIONS",
    "OTHER_RELATED",
]

relation_list = "\n".join(f"- {r}" for r in ALLOWED_RELATIONS)
entity_type_list = "\n".join(f"- {t}" for t in ALLOWED_ENTITY_TYPES)
event_type_list = "\n".join(f"- {t}" for t in ALLOWED_EVENT_TYPES)

print("Allowed relations:")
print(relation_list)


# ==============================================================================
# ## 8. Expected JSON schema and validation

# ------------------------------------------------------------------------------
# Code cell 17
EXPECTED_TOP_LEVEL_KEYS = ["entities", "events", "claims", "relations"]


def empty_extraction():
    return {
        "entities": [],
        "events": [],
        "claims": [],
        "relations": [],
    }


def safe_float(x, default=0.7):
    try:
        return float(x)
    except Exception:
        return default


def validate_and_normalize_extraction(obj):
    """
    Make the output safe for downstream notebooks.
    If a field is missing or malformed, replace with an empty list.
    """
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object.")

    out = empty_extraction()

    for key in EXPECTED_TOP_LEVEL_KEYS:
        value = obj.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ValueError(f"Field {key} must be a list.")
        out[key] = value

    # Normalize entities
    norm_entities = []
    for i, e in enumerate(out["entities"]):
        if not isinstance(e, dict):
            continue
        norm_entities.append({
            "local_id": str(e.get("local_id", f"ent_{i}")),
            "label": str(e.get("label", "")).strip(),
            "type": str(e.get("type", "Other")).strip() or "Other",
            "description": str(e.get("description", "")).strip(),
            "confidence": safe_float(e.get("confidence", 0.7)),
        })
    out["entities"] = [e for e in norm_entities if e["label"]]

    # Normalize events
    norm_events = []
    for i, ev in enumerate(out["events"]):
        if not isinstance(ev, dict):
            continue
        norm_events.append({
            "local_id": str(ev.get("local_id", f"event_{i}")),
            "label": str(ev.get("label", "")).strip(),
            "type": str(ev.get("type", "Other")).strip() or "Other",
            "date": str(ev.get("date", "")).strip(),
            "location": str(ev.get("location", "")).strip(),
            "description": str(ev.get("description", "")).strip(),
            "confidence": safe_float(ev.get("confidence", 0.7)),
        })
    out["events"] = [ev for ev in norm_events if ev["label"]]

    # Normalize claims
    norm_claims = []
    for i, cl in enumerate(out["claims"]):
        if not isinstance(cl, dict):
            continue
        norm_claims.append({
            "local_id": str(cl.get("local_id", f"claim_{i}")),
            "claim_text": str(cl.get("claim_text", "")).strip(),
            "speaker": str(cl.get("speaker", "")).strip(),
            "target": str(cl.get("target", "")).strip(),
            "stance": str(cl.get("stance", "")).strip(),
            "confidence": safe_float(cl.get("confidence", 0.7)),
        })
    out["claims"] = [cl for cl in norm_claims if cl["claim_text"]]

    # Normalize relations
    norm_relations = []
    for i, r in enumerate(out["relations"]):
        if not isinstance(r, dict):
            continue
        rel = str(r.get("relation", "")).strip()
        if rel not in ALLOWED_RELATIONS:
            rel = "OTHER_RELATED"

        norm_relations.append({
            "source": str(r.get("source", "")).strip(),
            "target": str(r.get("target", "")).strip(),
            "relation": rel,
            "evidence": str(r.get("evidence", "")).strip(),
            "confidence": safe_float(r.get("confidence", 0.7)),
        })

    out["relations"] = [
        r for r in norm_relations
        if r["source"] and r["target"] and r["source"] != r["target"]
    ]

    return out


# ==============================================================================
# ## 9. Extraction prompt

# ------------------------------------------------------------------------------
# Code cell 19
PROMPT_TEMPLATE = """
You are extracting a controlled knowledge graph from a historical/diplomatic document chunk.

Be conservative.
Use only information explicitly supported by the text.
Do not invent entities, events, claims, relations, dates, or locations.
If the chunk contains no useful extractable information, return empty arrays.

Allowed entity types:
{entity_type_list}

Allowed event types:
{event_type_list}

Allowed relation types:
{relation_list}

Important distinctions:
- Entity = person, organization, place, country, group, or institution.
- Event = something that happened or was reported to happen.
- Claim = a statement, allegation, interpretation, or assertion made by a source.
- Relation = a connection between entities, events, or claims.
- Claims are not necessarily verified truth. Preserve them as claims.

Metadata:
doc_id={doc_id}
source_id={source_id}
paragraph_id={paragraph_id}
subject={subject}
date={date}

Return ONLY valid JSON with this exact structure:

{{
  "entities": [
    {{
      "local_id": "ent_0",
      "label": "entity name",
      "type": "Person|Organization|Government|Location|Country|Military|ReligiousGroup|PoliticalGroup|Document|Other",
      "description": "brief description if supported",
      "confidence": 0.7
    }}
  ],
  "events": [
    {{
      "local_id": "event_0",
      "label": "short event label",
      "type": "Meeting|Communication|Demonstration|Violence|Arrest|Travel|Speech|Negotiation|PolicyAction|MilitaryAction|EconomicAction|Other",
      "date": "date if explicitly available, otherwise empty",
      "location": "location if explicitly available, otherwise empty",
      "description": "brief event description",
      "confidence": 0.7
    }}
  ],
  "claims": [
    {{
      "local_id": "claim_0",
      "claim_text": "the claim or assertion",
      "speaker": "who made the claim if known",
      "target": "target of the claim if known",
      "stance": "support|oppose|neutral|unclear",
      "confidence": 0.7
    }}
  ],
  "relations": [
    {{
      "source": "local_id or label",
      "target": "local_id or label",
      "relation": "one of the allowed relation types",
      "evidence": "short supporting phrase from the text",
      "confidence": 0.7
    }}
  ]
}}

Text:
{text}
"""


def build_prompt(doc_id, source_id, paragraph_id, subject, date, text):
    return PROMPT_TEMPLATE.format(
        entity_type_list=entity_type_list,
        event_type_list=event_type_list,
        relation_list=relation_list,
        doc_id=doc_id,
        source_id=source_id,
        paragraph_id=paragraph_id,
        subject=subject,
        date=date,
        text=text,
    )


# ==============================================================================
# ## 10. Checkpoint and resume utilities

# ------------------------------------------------------------------------------
# Code cell 21
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    """
    Append one JSON object to a JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path, tolerate_bad_lines=True):
    """
    Load JSONL safely.
    If tolerate_bad_lines=True, malformed lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
                if not tolerate_bad_lines:
                    raise
    if bad:
        print(f"Warning: skipped {bad} malformed JSONL lines in {path}")
    return rows


def make_chunk_key(doc_id, source_id, paragraph_id):
    return (str(doc_id), str(source_id), str(paragraph_id))


def record_key_from_result(r):
    p = r.get("provenance", {})
    return make_chunk_key(p.get("doc_id", ""), p.get("source_id", ""), p.get("paragraph_id", ""))


def record_key_from_error(r):
    return make_chunk_key(r.get("doc_id", ""), r.get("source_id", ""), r.get("paragraph_id", ""))


def infer_record_chunk_mode(record):
    provenance = record.get("provenance", {}) if isinstance(record, dict) else {}
    raw_mode = str(provenance.get("chunk_mode", "")).strip().lower()
    if raw_mode in {"paragraph", "document"}:
        return raw_mode

    paragraph_id = str(
        provenance.get("paragraph_id", record.get("paragraph_id", ""))
    ).strip().lower()
    if re.fullmatch(r"doc\d+", paragraph_id):
        return "document"
    if re.fullmatch(r"p\d+", paragraph_id):
        return "paragraph"
    return None


def backup_checkpoint_files(tag=None):
    """
    Copy checkpoint/error/summary files to timestamped backup files.
    """
    if tag is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in [CHECKPOINT_JSONL, ERROR_JSONL, RUN_SUMMARY_JSONL]:
        if Path(path).exists():
            dst = BACKUP_DIR / f"{Path(path).stem}_{tag}{Path(path).suffix}"
            shutil.copy2(path, dst)

    print(f"Backup complete: {BACKUP_DIR}")


if OVERWRITE_EXISTING:
    for p in [CHECKPOINT_JSONL, ERROR_JSONL, RUN_SUMMARY_JSONL]:
        if Path(p).exists():
            print("Removing:", p)
            Path(p).unlink()

existing_results = load_jsonl(CHECKPOINT_JSONL)
existing_errors = load_jsonl(ERROR_JSONL)

completed_keys = {
    record_key_from_result(r)
    for r in existing_results
    if "provenance" in r
}

error_keys = {
    record_key_from_error(r)
    for r in existing_errors
}

print("Existing result chunks:", len(existing_results))
print("Completed keys:", len(completed_keys))
print("Existing error chunks:", len(existing_errors))
print("Error keys:", len(error_keys))

existing_chunk_modes = {
    mode
    for mode in [*(infer_record_chunk_mode(r) for r in existing_results), *(infer_record_chunk_mode(r) for r in existing_errors)]
    if mode is not None
}
if existing_chunk_modes and existing_chunk_modes != {CHUNK_MODE}:
    raise RuntimeError(
        "Existing extraction artifacts were created with a different chunk mode. "
        f"Found {sorted(existing_chunk_modes)} but current run uses '{CHUNK_MODE}'. "
        "Use a fresh project directory or remove extraction/ and extraction_run/ before switching modes."
    )


# ==============================================================================
# ## 11. OpenAI client and API call with retry/backoff

# ------------------------------------------------------------------------------
# Code cell 23
if OpenAI is None:
    raise ImportError("OpenAI package not installed. Run: pip install openai")

# In Colab, set your key before running:
# import os
# os.environ["OPENAI_API_KEY"] = "sk-..."

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=60.0,
    max_retries=2,
    )


if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY is not set. API calls will fail until you set it.")


def parse_json_response(text):
    """
    Robustly parse a JSON object from model output.
    """
    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract first JSON object.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Could not parse JSON from model output.")


def call_llm_json(prompt, model=MODEL_NAME, max_retries=4):
    """
    Call OpenAI chat completion and request JSON output.
    Uses exponential backoff for transient failures.
    """
    last_err = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "developer",
                        "content": "Return only valid JSON matching the requested schema.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = response.choices[0].message.content
            usage = getattr(response, "usage", None)

            return content, usage

        except Exception as e:
            last_err = e
            sleep = min(60, (2 ** attempt) + random.random())
            print(f"API error attempt {attempt+1}/{max_retries}: {type(e).__name__}: {e}")
            print(f"Sleeping {sleep:.1f} seconds...")
            time.sleep(sleep)

    raise last_err


def process_chunk_unified(doc_id, source_id, paragraph_id, subject, date, text, model=MODEL_NAME):
    """
    Process one chunk and return a checkpoint-ready record.
    """
    prompt = build_prompt(
        doc_id=doc_id,
        source_id=source_id,
        paragraph_id=paragraph_id,
        subject=subject,
        date=date,
        text=text,
    )

    raw_text, usage = call_llm_json(prompt, model=model)

    parsed = parse_json_response(raw_text)
    extraction = validate_and_normalize_extraction(parsed)

    usage_dict = {}
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    return {
        "provenance": {
            "doc_id": str(doc_id),
            "source_id": str(source_id),
            "paragraph_id": str(paragraph_id),
            "chunk_mode": CHUNK_MODE,
            "subject": str(subject),
            "date": str(date),
            "model": model,
            "created_at": now_iso(),
            "chunk_char_len": len(str(text)),
            "chunk_sha1": hashlib.sha1(str(text).encode("utf-8")).hexdigest(),
        },
        "usage": usage_dict,
        "extraction": extraction,
    }


# ==============================================================================
# ## 12. Optional prompt preview and one-call smoke test

# ------------------------------------------------------------------------------
# Code cell 25
# Preview one prompt before running the whole job.
sample_row = work_source_df.iloc[0]
sample_body = str(sample_row["body"]) if pd.notna(sample_row["body"]) else ""
sample_chunks = split_text_units(sample_body)

print(f"Sample {EXTRACTION_UNIT_LABEL}:", len(sample_chunks))
if args.show_sample_prompt and sample_chunks:
    preview = build_prompt(
        doc_id=str(sample_row["doc_id"]),
        source_id=str(sample_row["id"]),
        paragraph_id=make_unit_id(0),
        subject=str(sample_row.get("subject", "")),
        date=str(sample_row.get("date", "")),
        text=sample_chunks[0],
    )
    print(preview[:2500])

# Optional one-call test:
RUN_ONE_CALL_TEST = False

if RUN_ONE_CALL_TEST and sample_chunks:
    test_out = process_chunk_unified(
        doc_id=str(sample_row["doc_id"]),
        source_id=str(sample_row["id"]),
        paragraph_id=make_unit_id(0),
        subject=str(sample_row.get("subject", "")),
        date=str(sample_row.get("date", "")),
        text=sample_chunks[0],
        model=MODEL_NAME,
    )
    print(json.dumps(test_out, indent=2, ensure_ascii=False)[:4000])


# ==============================================================================
# ## 13. Main checkpointed extraction loop

# ------------------------------------------------------------------------------
# Code cell 27
run_started_at = now_iso()
processed_now = 0
skipped_completed = 0
skipped_previous_error = 0
skipped_empty = 0
errors_now = 0

all_results = list(existing_results)
all_errors = list(existing_errors)

start_time = time.time()

# Prepare error-only mode lookup.
if RUN_FAILED_CHUNKS_ONLY:
    target_keys = set(error_keys)
    print("RUN_FAILED_CHUNKS_ONLY enabled. Target error keys:", len(target_keys))
else:
    target_keys = None

# --- Precompute pending chunks for this run ---
# This makes progress reporting accurate and avoids surprises when a few documents
# produce many paragraph chunks.

pending_chunks = []

for row_idx, row in work_source_df.iterrows():
    doc_id = str(row["doc_id"])
    source_id = str(row["id"])
    subject = str(row.get("subject", ""))
    date = str(row.get("date", ""))
    body = str(row["body"]) if pd.notna(row["body"]) else ""

    if is_effectively_empty(body):
        continue

    chunks = split_text_units(body)

    for j, chunk in enumerate(chunks):
        paragraph_id = make_unit_id(j)
        key = make_chunk_key(doc_id, source_id, paragraph_id)

        if RUN_FAILED_CHUNKS_ONLY and key not in error_keys:
            continue

        if key in completed_keys:
            continue

        if (not RUN_FAILED_CHUNKS_ONLY) and SKIP_PREVIOUS_ERRORS and key in error_keys:
            continue

        pending_chunks.append({
            "row_idx": row_idx,
            "doc_id": doc_id,
            "source_id": source_id,
            "paragraph_id": paragraph_id,
            "subject": subject,
            "date": date,
            "chunk": chunk,
            "key": key,
        })

if MAX_NEW_CHUNKS_THIS_RUN is None:
    MAX_NEW_CHUNKS_THIS_RUN = len(pending_chunks)
    chunk_cap_label = "all pending chunks"
else:
    MAX_NEW_CHUNKS_THIS_RUN = max(0, min(MAX_NEW_CHUNKS_THIS_RUN, len(pending_chunks)))
    chunk_cap_label = f"user cap={MAX_NEW_CHUNKS_THIS_RUN}"

pending_chunks = pending_chunks[:MAX_NEW_CHUNKS_THIS_RUN]

print("Selected documents:", len(work_source_df))
print(f"Pending {EXTRACTION_UNIT_LABEL} this run:", len(pending_chunks))
print("Chunk cap this run:", chunk_cap_label)


if not RUN_EXTRACTION:
    print("RUN_EXTRACTION is False. Set RUN_EXTRACTION=True to run extraction.")
else:
    pbar = tqdm(total=len(pending_chunks), desc=f"{EXTRACTION_UNIT_LABEL.title()} processed")

    for row_idx, row in work_source_df.iterrows():
        doc_id = str(row["doc_id"])
        source_id = str(row["id"])
        subject = str(row.get("subject", ""))
        date = str(row.get("date", ""))
        body = str(row["body"]) if pd.notna(row["body"]) else ""

        if is_effectively_empty(body):
            skipped_empty += 1
            continue

        chunks = split_text_units(body)

        for j, chunk in enumerate(chunks):
            paragraph_id = make_unit_id(j)
            key = make_chunk_key(doc_id, source_id, paragraph_id)

            if RUN_FAILED_CHUNKS_ONLY and key not in target_keys:
                continue

            if key in completed_keys:
                skipped_completed += 1
                continue

            if (not RUN_FAILED_CHUNKS_ONLY) and SKIP_PREVIOUS_ERRORS and key in error_keys:
                skipped_previous_error += 1
                continue

            if processed_now >= MAX_NEW_CHUNKS_THIS_RUN:
                print("Reached MAX_NEW_CHUNKS_THIS_RUN.")
                break

            try:
                out = process_chunk_unified(
                    doc_id=doc_id,
                    source_id=source_id,
                    paragraph_id=paragraph_id,
                    subject=subject,
                    date=date,
                    text=chunk,
                    model=MODEL_NAME,
                )

                append_jsonl(CHECKPOINT_JSONL, out)
                all_results.append(out)
                completed_keys.add(key)
                processed_now += 1

                if processed_now % LOG_EVERY == 0:
                    elapsed = time.time() - start_time
                    rate = processed_now / max(elapsed, 1)
                    print(
                        f"Processed {processed_now} {EXTRACTION_UNIT_LABEL} | "
                        f"elapsed={elapsed/60:.1f} min | "
                        f"rate={rate:.3f} {EXTRACTION_UNIT_LABEL}/sec"
                    )

                if processed_now % BACKUP_EVERY == 0:
                    backup_checkpoint_files(tag=f"after_{processed_now}_chunks")

                pbar.update(1)
                time.sleep(SLEEP_SECONDS)

            except Exception as e:
                errors_now += 1
                err = {
                    "doc_id": doc_id,
                    "source_id": source_id,
                    "paragraph_id": paragraph_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=2),
                    "text_preview": chunk[:500],
                    "created_at": now_iso(),
                    "model": MODEL_NAME,
                }
                if STORE_FULL_TEXT_IN_ERRORS:
                    err["full_text"] = chunk

                append_jsonl(ERROR_JSONL, err)
                all_errors.append(err)
                error_keys.add(key)

        if processed_now >= MAX_NEW_CHUNKS_THIS_RUN:
            break

    pbar.close()

run_finished_at = now_iso()
elapsed = time.time() - start_time

run_summary = {
    "run_started_at": run_started_at,
    "run_finished_at": run_finished_at,
    "model": MODEL_NAME,
    "chunk_mode": CHUNK_MODE,
    "source_csv": str(SOURCE_CSV),
    "project_dir": str(PROJECT_DIR),
    "start_at": START_AT,
    "n_docs": N_DOCS,
    "documents_in_work_df": len(work_source_df),
    "processed_now": processed_now,
    "skipped_completed": skipped_completed,
    "skipped_previous_error": skipped_previous_error,
    "skipped_empty": skipped_empty,
    "errors_now": errors_now,
    "total_stored_results": len(load_jsonl(CHECKPOINT_JSONL)),
    "total_stored_errors": len(load_jsonl(ERROR_JSONL)),
    "elapsed_seconds": elapsed,
}

append_jsonl(RUN_SUMMARY_JSONL, run_summary)

print(json.dumps(run_summary, indent=2))
backup_checkpoint_files(tag="run_end")


# ==============================================================================
# ## 14. Token and cost diagnostics

# ------------------------------------------------------------------------------
# Code cell 29
# This cell summarizes token usage when the API response includes usage metadata.
# Pricing changes over time, so this does not compute dollars by default.
# You can multiply total_tokens by the current model price manually.

results = load_jsonl(CHECKPOINT_JSONL)

usage_rows = []
for r in results:
    u = r.get("usage", {}) or {}
    p = r.get("provenance", {}) or {}
    usage_rows.append({
        "doc_id": p.get("doc_id", ""),
        "source_id": p.get("source_id", ""),
        "paragraph_id": p.get("paragraph_id", ""),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "total_tokens": u.get("total_tokens"),
        "chunk_char_len": p.get("chunk_char_len"),
    })

usage_df = pd.DataFrame(usage_rows)

if len(usage_df):
    display(usage_df.describe(include="all"))
    print("Total prompt tokens:", pd.to_numeric(usage_df["prompt_tokens"], errors="coerce").sum())
    print("Total completion tokens:", pd.to_numeric(usage_df["completion_tokens"], errors="coerce").sum())
    print("Total tokens:", pd.to_numeric(usage_df["total_tokens"], errors="coerce").sum())
else:
    print("No usage rows yet.")


# ==============================================================================
# ## 15. Flatten checkpoint into CSV tables

# ------------------------------------------------------------------------------
# Code cell 31
results = load_jsonl(CHECKPOINT_JSONL)

entity_rows = []
event_rows = []
claim_rows = []
relation_rows = []

for r in results:
    p = r.get("provenance", {})
    ext = r.get("extraction", {})

    base = {
        "doc_id": p.get("doc_id", ""),
        "source_id": p.get("source_id", ""),
        "paragraph_id": p.get("paragraph_id", ""),
        "subject": p.get("subject", ""),
        "date": p.get("date", ""),
        "model": p.get("model", ""),
    }

    for e in ext.get("entities", []):
        row = dict(base)
        row.update(e)
        row["node_key"] = f'{base["doc_id"]}::{base["paragraph_id"]}::{e.get("local_id", "")}'
        entity_rows.append(row)

    for ev in ext.get("events", []):
        row = dict(base)
        row.update(ev)
        row["node_key"] = f'{base["doc_id"]}::{base["paragraph_id"]}::{ev.get("local_id", "")}'
        event_rows.append(row)

    for cl in ext.get("claims", []):
        row = dict(base)
        row.update(cl)
        row["node_key"] = f'{base["doc_id"]}::{base["paragraph_id"]}::{cl.get("local_id", "")}'
        claim_rows.append(row)

    for rel in ext.get("relations", []):
        row = dict(base)
        row.update(rel)
        relation_rows.append(row)

entities_df = pd.DataFrame(entity_rows)
events_df = pd.DataFrame(event_rows)
claims_df = pd.DataFrame(claim_rows)
#NEW STANCE code
# --- Normalize claim stance values ---
# This is a local schema-cleaning step.
# It does not call the LLM and does not change chunk_results.jsonl.

ALLOWED_STANCES = {
    "support",
    "oppose",
    "neutral",
    "unclear",
}

if len(claims_df) and "stance" in claims_df.columns:
    claims_df["stance"] = (
        claims_df["stance"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    claims_df.loc[
        ~claims_df["stance"].isin(ALLOWED_STANCES),
        "stance"
    ] = "unclear"
#End of STANCE Code 



relations_df = pd.DataFrame(relation_rows)

#New code after 1000

ALLOWED_ENTITY_TYPES_SET = set(ALLOWED_ENTITY_TYPES)
ALLOWED_EVENT_TYPES_SET = set(ALLOWED_EVENT_TYPES)

if len(entities_df) and "type" in entities_df.columns:
    entities_df["type"] = entities_df["type"].astype(str).str.strip()
    entities_df.loc[
        ~entities_df["type"].isin(ALLOWED_ENTITY_TYPES_SET),
        "type"
    ] = "Other"

if len(events_df) and "type" in events_df.columns:
    events_df["type"] = events_df["type"].astype(str).str.strip()
    events_df.loc[
        ~events_df["type"].isin(ALLOWED_EVENT_TYPES_SET),
        "type"
    ] = "Other"
#End of new code after 1000


entities_path = EXTRACTION_DIR / "entities.csv"
events_path = EXTRACTION_DIR / "events.csv"
claims_path = EXTRACTION_DIR / "claims.csv"
relations_path = EXTRACTION_DIR / "relations.csv"

entities_df.to_csv(entities_path, index=False)
events_df.to_csv(events_path, index=False)
claims_df.to_csv(claims_path, index=False)
relations_df.to_csv(relations_path, index=False)

print("Saved:")
print(entities_path, entities_df.shape)
print(events_path, events_df.shape)
print(claims_path, claims_df.shape)
print(relations_path, relations_df.shape)

display(entities_df.head())
display(events_df.head())
display(claims_df.head())
display(relations_df.head())


# ==============================================================================
# ## 16. Diagnostics and quality audit

# ------------------------------------------------------------------------------
# Code cell 33
print("Checkpoint results:", len(load_jsonl(CHECKPOINT_JSONL)))
print("Checkpoint errors:", len(load_jsonl(ERROR_JSONL)))

diagnostics = {
    "entities": len(entities_df),
    "events": len(events_df),
    "claims": len(claims_df),
    "relations": len(relations_df),
    "unique_docs_extracted": entities_df["doc_id"].nunique() if len(entities_df) and "doc_id" in entities_df else 0,
}

print(json.dumps(diagnostics, indent=2))

if len(entities_df):
    print("\nTop entity types:")
    display(entities_df["type"].value_counts().head(20))

if len(events_df):
    print("\nTop event types:")
    display(events_df["type"].value_counts().head(20))

if len(relations_df):
    print("\nTop relation types:")
    display(relations_df["relation"].value_counts().head(30))

# Random audit sample.
AUDIT_N = 10

print("\nRandom extraction audit sample:")
audit_rows = []
for r in random.sample(results, k=min(AUDIT_N, len(results))):
    p = r.get("provenance", {})
    ext = r.get("extraction", {})
    audit_rows.append({
        "doc_id": p.get("doc_id"),
        "paragraph_id": p.get("paragraph_id"),
        "n_entities": len(ext.get("entities", [])),
        "n_events": len(ext.get("events", [])),
        "n_claims": len(ext.get("claims", [])),
        "n_relations": len(ext.get("relations", [])),
    })

display(pd.DataFrame(audit_rows))


# ==============================================================================
# ## print Unique entities and unique events

# ------------------------------------------------------------------------------

unique_entities = entities_df["label"].nunique() if "label" in entities_df.columns else 0
unique_events = events_df["label"].nunique() if "label" in events_df.columns else 0

print("Unique entities:", unique_entities)
print("Unique events:", unique_events)


# ==============================================================================
# ## 17. Inspect and optionally retry errors

# ------------------------------------------------------------------------------
# Code cell 35
errors = load_jsonl(ERROR_JSONL)

print("Total errors:", len(errors))

if errors:
    err_df = pd.DataFrame(errors)
    display(err_df[["doc_id", "source_id", "paragraph_id", "error_type", "error", "text_preview"]].head(50))

    print("\nError types:")
    display(err_df["error_type"].value_counts())

# To retry only failed chunks in a later run:
# 1. Set RUN_FAILED_CHUNKS_ONLY = True
# 2. Set SKIP_PREVIOUS_ERRORS = False
# 3. Set RUN_EXTRACTION = True
# 4. Re-run checkpoint utility cell and main extraction loop.


# ==============================================================================
# ## 18. Output contract for downstream notebooks
# 
# This notebook writes the following files:
# 
# ```text
# kg_matryoshka_project/
#   extraction/
#     entities.csv
#     events.csv
#     claims.csv
#     relations.csv
# 
#   extraction_run/
#     chunk_results.jsonl
#     chunk_errors.jsonl
#     run_summary.jsonl
#     backups/
# ```
# 
# The downstream graph notebook expects:
# 
# - `extraction/entities.csv`
# - `extraction/events.csv`
# - `extraction/claims.csv`
# - `extraction/relations.csv`
# 
# Do not change these filenames unless you also update downstream notebooks.
