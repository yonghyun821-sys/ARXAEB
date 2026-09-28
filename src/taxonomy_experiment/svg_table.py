from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def write_accuracy_table(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    metric: str,
) -> None:
    conditions = (
        "label_only",
        "own_taxonomy",
        "foreign_taxonomy",
        "merged_taxonomy",
    )
    headers = (
        "Label-Only",
        "Own Taxonomy",
        "Foreign Taxonomy",
        "Merged Taxonomy",
    )
    by_key = {(str(row["dataset"]), str(row["condition"])): row for row in rows}
    datasets = (("agentrx", "AgentRx"), ("agenterrorbench", "AgentErrorBench"))
    x_edges = (35, 245, 330, 530, 740, 955, 1165)
    centers = tuple((left + right) / 2 for left, right in zip(x_edges, x_edges[1:]))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="310" viewBox="0 0 1200 310">',
        '<rect width="1200" height="310" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}.title{font-size:25px;font-weight:700}.header{font-size:16px;font-weight:600}.cell{font-size:16px;font-weight:400}.note{font-size:13px;fill:#667085}.line{stroke:#cfd4dc;stroke-width:1}</style>',
        f'<text x="600" y="38" text-anchor="middle" class="title">{html.escape(title)}</text>',
        '<rect x="35" y="65" width="1130" height="55" fill="#f3f4f6"/>',
        '<rect x="35" y="65" width="1130" height="165" fill="none" stroke="#aeb4bd" stroke-width="1.2"/>',
    ]
    for x in x_edges[1:-1]:
        parts.append(f'<line x1="{x}" y1="65" x2="{x}" y2="230" class="line"/>')
    for y in (120, 175):
        parts.append(f'<line x1="35" y1="{y}" x2="1165" y2="{y}" class="line"/>')
    for x, label in zip(centers, ("Dataset", "N", *headers)):
        parts.append(f'<text x="{x}" y="98" text-anchor="middle" class="header">{html.escape(label)}</text>')
    for y, (dataset, label) in zip((153, 208), datasets):
        sample = by_key[(dataset, conditions[0])]
        values = [
            label,
            str(sample["n"]),
            *(f"{float(by_key[(dataset, condition)][metric]) * 100:.2f}%" for condition in conditions),
        ]
        for x, value in zip(centers, values):
            parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" class="cell">{html.escape(value)}</text>')
    parts.extend([
        '<text x="600" y="270" text-anchor="middle" class="note">Values are reported without highlighting a preferred condition.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
