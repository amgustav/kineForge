from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _relative_link(path_value: str, output_dir: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    path_abs = path if path.is_absolute() else Path.cwd() / path
    try:
        return path_abs.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return str(path_value)


def _ranked_names(summary: Mapping[str, Any]) -> list[str]:
    ranked = summary.get("ranked_scenarios")
    if not ranked:
        return list(summary["scenarios"].keys())
    return [str(row["scenario"]) for row in ranked]


def build_replay_gallery(summary: Mapping[str, Any], replay_index: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    scenarios = []
    for name in _ranked_names(summary):
        scenario = summary["scenarios"][name]
        replay_path = str(replay_index.get("scenarios", {}).get(name, {}).get("trajectory_png", ""))
        scenarios.append(
            {
                "name": name,
                "gate_status": scenario["gate_status"],
                "success_rate": float(scenario["summary"].get("success_rate", 0.0)),
                "mean_final_distance": float(scenario["summary"].get("mean_final_distance", 0.0)),
                "failure_modes": list(scenario.get("failure_modes", ())),
                "description": str(scenario.get("description", "")),
                "limitations": list(scenario.get("limitations", ())),
                "trajectory_png": _relative_link(replay_path, output_dir),
                "scorecard_json": _relative_link(str(scenario.get("scorecard_json", "")), output_dir),
            }
        )
    return {
        "run_id": f"eval-matrix-{summary['timestamp']}",
        "gate_profile": summary.get("gate_profile", "standard"),
        "matrix_preset": summary.get("matrix_preset", "default"),
        "scenario_count": int(summary.get("scenario_count", len(scenarios))),
        "scenarios": scenarios,
    }


def write_replay_gallery_html(path: Path, summary: Mapping[str, Any], replay_index: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    gallery = build_replay_gallery(summary, replay_index, path.parent)
    cards = []
    for scenario in gallery["scenarios"]:
        failures = ", ".join(scenario["failure_modes"]) or "none"
        limitations = "; ".join(scenario["limitations"])
        image = (
            f'<a href="{escape(scenario["trajectory_png"])}"><img src="{escape(scenario["trajectory_png"])}" alt="{escape(scenario["name"])} trajectory"></a>'
            if scenario["trajectory_png"]
            else "<p>No trajectory image recorded.</p>"
        )
        scorecard = (
            f'<a href="{escape(scenario["scorecard_json"])}">scorecard</a>'
            if scenario["scorecard_json"]
            else "scorecard unavailable"
        )
        cards.append(
            "<article class=\"card\">"
            f"<h2>{escape(scenario['name'])}</h2>"
            f"<p><strong>Gate:</strong> {escape(scenario['gate_status'])}</p>"
            f"<p><strong>Success rate:</strong> {scenario['success_rate']:.3f}</p>"
            f"<p><strong>Mean final distance:</strong> {scenario['mean_final_distance']:.4f} m</p>"
            f"<p><strong>Failures:</strong> {escape(failures)}</p>"
            f"<p>{escape(scenario['description'])}</p>"
            f"<p class=\"limitations\">{escape(limitations)}</p>"
            f"{image}"
            f"<p>{scorecard}</p>"
            "</article>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>kineForge replay gallery - {escape(gallery['run_id'])}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #17202a; }}
    .summary {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; margin-bottom: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 0.5rem; padding: 1rem; background: #fff; }}
    .card h2 {{ margin-top: 0; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; }}
    .limitations {{ color: #57606a; font-size: 0.92rem; }}
    code {{ background: #f6f8fa; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>kineForge replay gallery</h1>
  <dl class="summary">
    <dt>run</dt><dd><code>{escape(gallery['run_id'])}</code></dd>
    <dt>matrix preset</dt><dd><code>{escape(gallery['matrix_preset'])}</code></dd>
    <dt>gate profile</dt><dd><code>{escape(gallery['gate_profile'])}</code></dd>
    <dt>scenarios</dt><dd>{gallery['scenario_count']}</dd>
  </dl>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return gallery
