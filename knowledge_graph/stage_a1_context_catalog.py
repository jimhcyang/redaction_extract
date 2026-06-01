#!/usr/bin/env python3
"""
Stage A1: build the mention/context catalog from Stage A extraction outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_graph.entity_resolution import a1_context as stage_impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A1 context cataloging from Stage A extraction outputs.")
    parser.add_argument("PROJECT_DIR", help="Project directory containing extraction/ and optional config/entity_alias_overrides.csv")
    parser.add_argument("--entity-dir", default=None, help="Legacy ignored argument retained for compatibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
    entity_dir = Path(args.entity_dir).expanduser().resolve() if args.entity_dir else None
    stage_impl.run_stage_a1_context_catalog(project_dir, entity_dir=entity_dir)


if __name__ == "__main__":
    main()
