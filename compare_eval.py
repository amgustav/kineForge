from __future__ import annotations

import argparse
import json
from pathlib import Path

from kineforge.matrix import compare_summary_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two kineForge eval matrix summary JSON files.")
    parser.add_argument("--before", required=True, help="Baseline matrix_summary.json path.")
    parser.add_argument("--after", required=True, help="Candidate matrix_summary.json path.")
    parser.add_argument("--output", help="Optional path for comparison JSON. Prints to stdout when omitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else None
    comparison = compare_summary_files(Path(args.before), Path(args.after), output_path)
    if output_path is None:
        print(json.dumps(comparison, indent=2, sort_keys=True))
    else:
        print(f"Comparison JSON: {output_path}")


if __name__ == "__main__":
    main()
