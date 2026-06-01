#!/usr/bin/env python3
"""
Stage A4: adjudicate remaining sense clusters and finalize mention-level entities_resolved.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_graph.entity_resolution import a4_llm_resolution as stage_impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A4 LLM resolution from Stage A3 outputs.")
    parser.set_defaults(run_openai=True)
    parser.add_argument("PROJECT_DIR", help="Project directory containing A3 outputs in entity_resolution/")
    parser.add_argument("--run-openai", dest="run_openai", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-openai", dest="run_openai", action="store_false", help="Skip A4 OpenAI adjudication and keep unresolved clusters separate.")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for A4 cluster adjudication when enabled.")
    parser.add_argument("--max-openai-candidates", type=int, default=None, help="Optional cap on A4 sense clusters to adjudicate in a single run.")
    parser.add_argument("--openai-min-mentions", type=int, default=1, help="Minimum cluster mention count before OpenAI adjudication is attempted, unless the label already split into multiple clusters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
    stage_impl.run_stage_a4_llm_resolution(
        project_dir,
        run_openai=args.run_openai,
        model=args.openai_model,
        max_clusters=args.max_openai_candidates,
        openai_min_mentions=args.openai_min_mentions,
    )


if __name__ == "__main__":
    main()
