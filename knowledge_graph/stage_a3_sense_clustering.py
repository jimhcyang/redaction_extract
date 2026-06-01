#!/usr/bin/env python3
"""
Stage A3: cluster ambiguous labels into mention-level senses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_graph.entity_resolution import a3_sense_clustering as stage_impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A3 sense clustering from Stage A2 outputs.")
    parser.add_argument("PROJECT_DIR", help="Project directory containing A2 outputs in entity_resolution/")
    parser.add_argument("--top-k-neighbors", type=int, default=15, help="Nearest-neighbor pool size for candidate-label retrieval.")
    parser.add_argument("--max-candidate-labels-per-cluster", type=int, default=5, help="Maximum candidate canonical labels to keep per sense cluster.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
    stage_impl.run_stage_a3_sense_clustering(
        project_dir,
        top_k_neighbors=args.top_k_neighbors,
        max_candidate_labels_per_cluster=args.max_candidate_labels_per_cluster,
    )


if __name__ == "__main__":
    main()
