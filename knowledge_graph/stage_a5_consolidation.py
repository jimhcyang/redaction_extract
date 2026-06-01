#!/usr/bin/env python3
"""
Stage A5: consolidate post-A4 canonical entities into cleaner final canonicals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_graph.entity_resolution import a5_consolidation as stage_impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A5 consolidation from Stage A4 outputs.")
    parser.set_defaults(run_openai=True)
    parser.add_argument("PROJECT_DIR", help="Project directory containing A4 outputs in entity_resolution/")
    parser.add_argument("--run-openai", dest="run_openai", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-openai", dest="run_openai", action="store_false", help="Skip A5 OpenAI consolidation adjudication and keep candidate canonical entities separate.")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for A5 consolidation adjudication when enabled.")
    parser.add_argument("--max-openai-candidates", type=int, default=None, help="Optional cap on A5 canonical-pair adjudications in a single run.")
    parser.add_argument("--top-k-neighbors", type=int, default=10, help="Nearest-neighbor pool size for A5 canonical candidate retrieval.")
    parser.add_argument("--review-min-score", type=float, default=0.72, help="Minimum A5 pair score that goes to OpenAI review.")
    parser.add_argument("--attention-review-min-score", type=float, default=0.58, help="Lower review threshold for A5 entities already flagged as likely over-split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
    stage_impl.run_stage_a5_consolidation(
        project_dir,
        run_openai=args.run_openai,
        model=args.openai_model,
        max_candidates=args.max_openai_candidates,
        top_k_neighbors=args.top_k_neighbors,
        review_min_score=args.review_min_score,
        attention_review_min_score=args.attention_review_min_score,
    )


if __name__ == "__main__":
    main()
