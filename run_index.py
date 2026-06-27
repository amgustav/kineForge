from __future__ import annotations

import argparse
from pathlib import Path

from kineforge.registry import build_run_index, write_run_index_csv, write_run_index_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index local kineForge run artifacts.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing kineForge run outputs.")
    parser.add_argument("--output", default="runs/run_index.json", help="Output JSON path.")
    parser.add_argument("--csv", default="runs/run_index.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_run_index(Path(args.runs_dir))
    json_path = Path(args.output)
    csv_path = Path(args.csv)
    write_run_index_json(json_path, index)
    write_run_index_csv(csv_path, index)
    print(f"Run index JSON: {json_path}")
    print(f"Run index CSV: {csv_path}")
    print(f"Runs indexed: {index['run_count']}")


if __name__ == "__main__":
    main()
