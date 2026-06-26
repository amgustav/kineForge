from __future__ import annotations

import argparse

from kineforge.reports import prepare_run_dir, timestamp, write_json
from kineforge.sweeps import (
    build_sweep_summary,
    load_sweep_config,
    run_sweep_variant,
    write_sweep_report_html,
    write_sweep_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a kineForge config sweep.")
    parser.add_argument("--config", default="configs/sweeps/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_config = load_sweep_config(
        args.config,
        seed_override=args.seed,
        timesteps_override=args.timesteps,
        episodes_override=args.episodes,
    )
    run_timestamp = timestamp()
    sweep_dir = prepare_run_dir("sweep", run_timestamp)

    variant_results = {}
    for variant in sweep_config.variants:
        variant_dir = sweep_dir / "variants" / variant.name
        print(f"Running variant {variant.name}: seed={variant.seed} timesteps={variant.timesteps}")
        variant_results[variant.name] = run_sweep_variant(variant_dir, sweep_config, variant, run_timestamp)

    summary = build_sweep_summary(
        sweep_config=sweep_config,
        output_dir=sweep_dir,
        run_timestamp=run_timestamp,
        variant_results=variant_results,
    )
    summary_path = sweep_dir / "sweep_summary.json"
    report_path = sweep_dir / "sweep_report.html"
    csv_path = sweep_dir / "summary.csv"
    write_json(summary_path, summary)
    write_sweep_report_html(report_path, summary)
    write_sweep_summary_csv(csv_path, summary)

    print(f"Sweep dir: {sweep_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"HTML report: {report_path}")
    print(f"CSV summary: {csv_path}")
    for row in summary["ranked_variants"]:
        print(
            f"#{row['rank']} {row['variant']}: {row['gate_status']} "
            f"success_rate={row['success_rate']:.3f} mean_final_distance={row['mean_final_distance']:.4f}"
        )


if __name__ == "__main__":
    main()
