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
        for path in RUNS.glob("*/condition_results.csv")
        if (path.parent / "run_manifest.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No completed run with condition_results.csv was found")
    return max(candidates, key=lambda path: (path / "condition_results.csv").stat().st_mtime)


def main() -> None:
    run_dir = latest_complete_run()
    with (run_dir / "condition_results.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    values = {
        (row["dataset"], row["condition"]): float(row["accuracy"]) * 100
        for row in rows
    }

    datasets = [("agentrx", "AgentRx (N=73)"), ("agenterrorbench", "AgentErrorBench (N=171)")]
    width, height = 1300, 650
    chart_top, chart_bottom = 115, 520
    maximum = 32.0
    panel_lefts = [90, 690]
    panel_width = 500
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.title{font-size:25px;font-weight:700}.panel{font-size:19px;font-weight:700}.tick{font-size:13px;fill:#667085}.label{font-size:13px}.value{font-size:14px;font-weight:700}.note{font-size:13px;fill:#667085}</style>',
        f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">Attribution Accuracy by Taxonomy Guidance Condition</text>',
    ]
    for tick in range(0, 33, 5):
        y = chart_bottom - (tick / maximum) * (chart_bottom - chart_top)
        parts.append(f'<line x1="70" y1="{y:.1f}" x2="1230" y2="{y:.1f}" stroke="#d9dee7" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="58" y="{y + 5:.1f}" text-anchor="end" class="tick">{tick}%</text>')
    parts.append('<text x="22" y="320" text-anchor="middle" transform="rotate(-90 22 320)" class="label">Native-label exact-match accuracy (%)</text>')
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
            words = label.split(" ")
            for line_index, word in enumerate(words):
                parts.append(f'<text x="{x + bar_width / 2}" y="{548 + line_index * 17}" text-anchor="middle" class="label">{html.escape(word)}</text>')
    parts.append(f'<text x="{width / 2}" y="625" text-anchor="middle" class="note">Model: Gemini 2.5 Flash | Same native output-label space in all conditions</text>')
    parts.append('</svg>')
    svg_path = run_dir / "accuracy_by_condition.svg"
    svg_path.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    table_path = run_dir / "accuracy_table.svg"
    write_accuracy_table(
        table_path,
        title="Attribution Accuracy",
        rows=rows,
        metric="accuracy",
    )
    print(f"svg: {svg_path}")
    print(f"table_svg: {table_path}")


if __name__ == "__main__":
    main()
