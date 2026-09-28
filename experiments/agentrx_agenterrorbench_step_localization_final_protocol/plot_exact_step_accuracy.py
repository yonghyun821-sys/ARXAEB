from __future__ import annotations

import csv
import html
from pathlib import Path

from taxonomy_experiment.svg_table import write_accuracy_table


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CONDITIONS = [
    "label_only",
    "own_taxonomy",
    "foreign_taxonomy",
    "merged_taxonomy",
]
LABELS = ["Label-Only", "Own Taxonomy", "Foreign Taxonomy", "Merged Taxonomy"]
COLORS = ["#8B95A1", "#2F6BFF", "#F04452", "#00A878"]


def latest_complete_run() -> Path:
    candidates = [
        path.parent
        for path in RUNS.glob("*/results_summary.csv")
        if (path.parent / "run_status.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No completed step-localization run was found")
    return max(candidates, key=lambda path: (path / "results_summary.csv").stat().st_mtime)


def main() -> None:
    run_dir = latest_complete_run()
    with (run_dir / "results_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    values = {
        (row["dataset"], row["condition"]): float(row["exact_step_accuracy"]) * 100
        for row in rows
    }

    width, height = 1300, 650
    chart_top, chart_bottom = 115, 520
    maximum = 30.0
    panel_lefts = [90, 690]
    panel_width = 500
    datasets = [("agentrx", "AgentRx (N=73)"), ("agenterrorbench", "AgentErrorBench (N=171)")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.title{font-size:25px;font-weight:700}.panel{font-size:19px;font-weight:700}.tick{font-size:13px;fill:#667085}.label{font-size:13px}.value{font-size:14px;font-weight:400}.note{font-size:13px;fill:#667085}</style>',
        f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">Exact Failure-Step Accuracy by Taxonomy Guidance</text>',
    ]
    for tick in range(0, 31, 5):
        y = chart_bottom - (tick / maximum) * (chart_bottom - chart_top)
        parts.append(f'<line x1="70" y1="{y:.1f}" x2="1230" y2="{y:.1f}" stroke="#d9dee7" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="58" y="{y + 5:.1f}" text-anchor="end" class="tick">{tick}%</text>')
    parts.append('<text x="22" y="320" text-anchor="middle" transform="rotate(-90 22 320)" class="label">Exact-step accuracy (%)</text>')
    for panel_left, (dataset, title) in zip(panel_lefts, datasets):
        scores = [values[(dataset, condition)] for condition in CONDITIONS]
        parts.append(f'<text x="{panel_left + panel_width / 2}" y="84" text-anchor="middle" class="panel">{html.escape(title)}</text>')
        bar_width, gap = 82, 38
        start = panel_left + 30
        for index, (label, color, score) in enumerate(zip(LABELS, COLORS, scores)):
            x = start + index * (bar_width + gap)
            bar_height = (score / maximum) * (chart_bottom - chart_top)
            y = chart_bottom - bar_height
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_width / 2}" y="{y - 9:.1f}" text-anchor="middle" class="value">{score:.2f}%</text>')
            for line_index, word in enumerate(label.split(" ")):
                parts.append(f'<text x="{x + bar_width / 2}" y="{548 + line_index * 17}" text-anchor="middle" class="label">{html.escape(word)}</text>')
    parts.append(f'<text x="{width / 2}" y="625" text-anchor="middle" class="note">Model: Gemini 2.5 Flash | Output: one visible candidate step ID</text>')
    parts.append('</svg>')

    output = run_dir / "exact_step_accuracy_by_condition.svg"
    output.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    table_output = run_dir / "exact_step_accuracy_table.svg"
    write_accuracy_table(
        table_output,
        title="Exact Failure-Step Accuracy",
        rows=rows,
        metric="exact_step_accuracy",
    )
    print(f"svg: {output}")
    print(f"table_svg: {table_output}")


if __name__ == "__main__":
    main()
