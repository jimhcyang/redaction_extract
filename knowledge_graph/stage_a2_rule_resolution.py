#!/usr/bin/env python3
"""
Stage A2: apply high-precision rule resolution and identify ambiguous labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_graph.entity_resolution import a2_rule_resolution as stage_impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A2 rule resolution from Stage A1 outputs.")
    parser.add_argument("PROJECT_DIR", help="Project directory containing entity_resolution/a1_mention_catalog.csv and label_catalog.csv")
    parser.add_argument("--entity-dir", default=None, help="Legacy ignored argument retained for compatibility.")
    parser.add_argument("--sense-ambiguity-threshold", type=float, default=0.45, help="Minimum label ambiguity score that triggers A3 sense clustering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
    entity_dir = Path(args.entity_dir).expanduser().resolve() if args.entity_dir else None
    stage_impl.run_stage_a2_rule_resolution(
        project_dir,
        entity_dir=entity_dir,
        ambiguity_threshold=args.sense_ambiguity_threshold,
    )


if __name__ == "__main__":
    main()
