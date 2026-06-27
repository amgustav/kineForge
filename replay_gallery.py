from __future__ import annotations

import argparse
from pathlib import Path

from kineforge.gallery import load_json, write_replay_gallery_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static kineForge replay gallery from matrix artifacts.")
    parser.add_argument("--summary", required=True, help="Path to matrix_summary.json.")
    parser.add_argument("--replay-index", required=True, help="Path to replay_index.json.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    replay_index_path = Path(args.replay_index)
    output_path = Path(args.output)
    gallery = write_replay_gallery_html(
        output_path,
        load_json(summary_path),
        load_json(replay_index_path),
    )
    print(f"Replay gallery: {output_path}")
    print(f"Scenarios: {gallery['scenario_count']}")


if __name__ == "__main__":
    main()
