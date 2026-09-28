from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.agentrx_agenterrorbench_joint_attribution_step import core as joint
from experiments.whowhen_agenterrorbench_step_localization import core as engine
from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, read_json, write_json
from taxonomy_experiment.token_count import TokenCounter


HERE = Path(__file__).resolve().parent
DATASETS = ("agentrx", "agenterrorbench")
CONDITIONS = ("label_only", "own_taxonomy", "foreign_taxonomy", "merged_taxonomy")
CONTRASTS = (
    ("own_taxonomy", "label_only"),
    ("foreign_taxonomy", "label_only"),
    ("merged_taxonomy", "label_only"),
    ("merged_taxonomy", "foreign_taxonomy"),
    ("merged_taxonomy", "own_taxonomy"),
)


def _task_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["task_hash"]) for row in rows)
    return {
        "trajectories": len(rows),
        "unique_task_hashes": len(counts),
        "duplicate_task_clusters": sum(value > 1 for value in counts.values()),
        "trajectories_in_duplicate_clusters": sum(
            value for value in counts.values() if value > 1
        ),
        "maximum_cluster_size": max(counts.values(), default=0),
    }


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    source_inputs = list(iter_jsonl(resolve_path(config, "source_joint_inputs")))
    source_gold = list(iter_jsonl(resolve_path(config, "source_joint_gold")))
    inputs = [
        {
            **row,
            "framework": row.get("domain") or row["dataset"],
            "benchmark": row["dataset"],
        }
        for row in source_inputs
    ]
    inputs.sort(key=lambda row: (row["dataset"], row["trajectory_id"]))
    gold = sorted(source_gold, key=lambda row: (row["dataset"], row["trajectory_id"]))
    input_keys = [(row["dataset"], row["trajectory_id"]) for row in inputs]
    gold_keys = [(row["dataset"], row["trajectory_id"]) for row in gold]
    if input_keys != gold_keys or len(input_keys) != len(set(input_keys)):
        raise ValueError("Frozen input/gold mapping is not one-to-one")
    counts = Counter(row["dataset"] for row in inputs)
    if counts != Counter({"agentrx": 73, "agenterrorbench": 171}):
        raise ValueError(f"Unexpected sample counts: {dict(counts)}")
    gold_index = {(row["dataset"], row["trajectory_id"]): row for row in gold}
    for row in inputs:
        candidates = [str(value) for value in row["candidate_step_ids"]]
        row["candidate_step_ids"] = candidates
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("Candidate step IDs must be nonempty and unique")
        reference = gold_index[(row["dataset"], row["trajectory_id"])]
        if str(reference["gold_step_id"]) not in candidates:
            raise ValueError("Gold step is not visible in the frozen candidate list")

    attribution_inputs = list(iter_jsonl(resolve_path(config, "source_attribution_inputs")))
    attribution_keys = {
        (row["dataset"], row["trajectory_id"], row["task_hash"])
        for row in attribution_inputs
    }
    step_keys = {
        (row["dataset"], row["trajectory_id"], row["task_hash"]) for row in inputs
    }
    if attribution_keys != step_keys:
        raise ValueError("Step sample differs from the preceding attribution sample")

    prepared_dir = resolve_path(config, "prepared")
    engine._freeze_jsonl(prepared_dir / "frozen_inputs.jsonl", inputs)
    engine._freeze_jsonl(prepared_dir / "frozen_gold.jsonl", gold)
    candidate_counts = {
        dataset: [
            len(row["candidate_step_ids"])
            for row in inputs
            if row["dataset"] == dataset
        ]
        for dataset in DATASETS
    }
    manifest = {
        "version": 1,
        "protocol": "step-only localization; taxonomy category is never an output target",
        "sampling": "exact reuse of the audited AgentRx 73 + AgentErrorBench 171 attribution sample",
        "conditions": list(CONDITIONS),
        "dataset_counts": dict(sorted(counts.items())),
        "total_trajectories": len(inputs),
        "planned_logical_predictions": len(inputs) * len(CONDITIONS),
        "candidate_policy": "all and only visible normalized trajectory step IDs; identical across conditions",
        "gold_separation": "gold_step_id exists only in frozen_gold.jsonl and is never passed to prompt construction",
        "sample_matches_preceding_attribution_experiment": True,
        "datasets": {
            dataset: {
                "task_structure": _task_stats(
                    [row for row in inputs if row["dataset"] == dataset]
                ),
                "candidate_count": {
                    "minimum": min(candidate_counts[dataset]),
                    "median": statistics.median(candidate_counts[dataset]),
                    "maximum": max(candidate_counts[dataset]),
                },
            }
            for dataset in DATASETS
        },
    }
    engine._freeze_json(prepared_dir / "sample_manifest.json", manifest)
    return {"inputs": inputs, "gold": gold, "manifest": manifest}


def _portable_prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    """Load the exact bundled step sample without requiring source datasets."""
    prepared_dir = resolve_path(config, "prepared")
    inputs = list(iter_jsonl(prepared_dir / "frozen_inputs.jsonl"))
    gold = list(iter_jsonl(prepared_dir / "frozen_gold.jsonl"))
    manifest = read_json(prepared_dir / "sample_manifest.json")
    input_keys = [(row["dataset"], row["trajectory_id"]) for row in inputs]
    gold_keys = [(row["dataset"], row["trajectory_id"]) for row in gold]
    if input_keys != gold_keys or len(input_keys) != 244 or len(input_keys) != len(set(input_keys)):
        raise RuntimeError("Bundled frozen step sample is incomplete")
    if manifest.get("dataset_counts") != {"agenterrorbench": 171, "agentrx": 73}:
        raise RuntimeError("Bundled step manifest does not match 73 + 171")
    gold_index = {
        (row["dataset"], row["trajectory_id"]): str(row["gold_step_id"])
        for row in gold
    }
    for row in inputs:
        candidates = [str(value) for value in row["candidate_step_ids"]]
        key = (row["dataset"], row["trajectory_id"])
        if not candidates or len(candidates) != len(set(candidates)):
            raise RuntimeError(f"Invalid bundled candidate list: {key}")
        if gold_index[key] not in candidates:
            raise RuntimeError(f"Bundled gold step is not visible: {key}")
        if "gold_step_id" in row:
            raise RuntimeError("Gold step leaked into a bundled step input")
    attribution_inputs = list(iter_jsonl(resolve_path(config, "source_attribution_inputs")))
    attribution_keys = {
        (row["dataset"], row["trajectory_id"], row["task_hash"])
        for row in attribution_inputs
    }
    step_keys = {
        (row["dataset"], row["trajectory_id"], row["task_hash"])
        for row in inputs
    }
    if attribution_keys != step_keys:
        raise RuntimeError("Bundled step and attribution samples differ")
    return {"inputs": inputs, "gold": gold, "manifest": manifest}


def load_source_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = {
        "agentrx": read_json(resolve_path(config, "source_agentrx_taxonomy")),
        "agenterrorbench": read_json(resolve_path(config, "source_agenterrorbench_taxonomy")),
    }
    expected = {"agentrx": 10, "agenterrorbench": 17}
    for dataset, taxonomy in sources.items():
        if len(taxonomy.get("categories") or []) != expected[dataset]:
            raise ValueError(f"Unexpected {dataset} taxonomy category count")
        engine._freeze_json(
            resolve_path(config, "frozen") / f"{dataset}_taxonomy.json", taxonomy
        )
    return sources


def obtain_merged_taxonomy(
    config: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    *,
    allow_generation: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del allow_generation
    merged = read_json(resolve_path(config, "source_merged_taxonomy"))
    source_audit = read_json(resolve_path(config, "source_merged_taxonomy_audit"))
    validation = joint.validate_merged_taxonomy(merged, sources)
    frozen = resolve_path(config, "frozen")
    engine._freeze_json(frozen / "native_merged_taxonomy.json", merged)
    audit = {
        "status": "reused_frozen_fresh_gpt5_merge_from_preceding_attribution_experiment",
        "new_merge_api_call_in_this_step_experiment": False,
        "source_merge_audit": source_audit,
        **validation,
    }
    engine._freeze_json(frozen / "native_merged_taxonomy_audit.json", audit)
    return merged, audit


def _taxonomy_payload(taxonomy: dict[str, Any], *, definitions: bool) -> dict[str, Any]:
    categories = []
    for row in taxonomy["categories"]:
        item = {"id": row["id"], "name": row["name"]}
        if definitions:
            item.update(
                {
                    "module": row.get("module"),
                    "definition": row["definition"],
                }
            )
        categories.append(item)
    return {
        "taxonomy_id": taxonomy["taxonomy_id"],
        "name": taxonomy["name"],
        "categories": categories,
    }


def _guidance(
    dataset: str,
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if condition == "label_only":
        return _taxonomy_payload(taxonomies[dataset], definitions=False)
    if condition == "own_taxonomy":
        return _taxonomy_payload(taxonomies[dataset], definitions=True)
    if condition == "foreign_taxonomy":
        other = "agenterrorbench" if dataset == "agentrx" else "agentrx"
        return _taxonomy_payload(taxonomies[other], definitions=True)
    if condition == "merged_taxonomy":
        return _taxonomy_payload(taxonomies["native_merged"], definitions=True)
    raise ValueError(f"Unknown condition: {condition}")


def build_prompt(
    record: dict[str, Any],
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, str]:
    values = {
        "task": record["task"],
        "diagnostic_guidance": json.dumps(
            _guidance(record["dataset"], condition, taxonomies),
            ensure_ascii=False,
            indent=2,
        ),
        "candidate_step_ids": json.dumps(
            record["candidate_step_ids"], ensure_ascii=False
        ),
        "trajectory": json.dumps(
            {
                "context": record.get("visible_context") or {},
                "steps": record["visible_trajectory"],
            },
            ensure_ascii=False,
            indent=2,
        ),
    }
    user = engine._prompt("attribution_user.txt").format(**values)
    return {
        "system_prompt": engine._prompt("attribution_system.txt"),
        "user_prompt": user,
        "base_user_prompt": re.sub(
            r"<DIAGNOSTIC_GUIDANCE>.*?</DIAGNOSTIC_GUIDANCE>",
            "<DIAGNOSTIC_GUIDANCE>[ONLY_ALLOWED_DIFFERENCE]</DIAGNOSTIC_GUIDANCE>",
            user,
            flags=re.DOTALL,
        ),
    }


def offline_preflight(
    config: dict[str, Any],
    prepared: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged, _ = obtain_merged_taxonomy(config, sources, allow_generation=False)
    taxonomies = {**sources, "native_merged": merged}
    gold_index = {
        (row["dataset"], row["trajectory_id"]): row for row in prepared["gold"]
    }
    checks = {
        "dataset_counts_exact": prepared["manifest"]["dataset_counts"]
        == {"agenterrorbench": 171, "agentrx": 73},
        "four_conditions": tuple(config["experiment"]["conditions"]) == CONDITIONS,
        "planned_rows_976": prepared["manifest"]["planned_logical_predictions"] == 976,
        "sample_matches_attribution_experiment": prepared["manifest"][
            "sample_matches_preceding_attribution_experiment"
        ],
        "gold_absent_from_inputs": all("gold_step_id" not in row for row in prepared["inputs"]),
        "gold_visible_but_prompt_hidden": True,
        "all_candidate_lists_nonempty_unique": True,
        "same_prompt_except_guidance": True,
        "candidate_list_identical_across_conditions": True,
        "step_only_output": True,
        "label_only_omits_definitions": True,
    }
    model = config["models"]["attribution"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    prompt_tokens: list[int] = []
    for record in prepared["inputs"]:
        key = (record["dataset"], record["trajectory_id"])
        checks["gold_visible_but_prompt_hidden"] &= str(
            gold_index[key]["gold_step_id"]
        ) in record["candidate_step_ids"]
        checks["all_candidate_lists_nonempty_unique"] &= bool(record["candidate_step_ids"]) and len(
            record["candidate_step_ids"]
        ) == len(set(record["candidate_step_ids"]))
        normalized = []
        candidate_text = json.dumps(record["candidate_step_ids"], ensure_ascii=False)
        for condition in CONDITIONS:
            built = build_prompt(record, condition, taxonomies)
            text = built["system_prompt"] + "\n" + built["user_prompt"]
            prompt_tokens.append(counter.count(text))
            normalized.append(built["base_user_prompt"])
            checks["candidate_list_identical_across_conditions"] &= candidate_text in text
            checks["step_only_output"] &= "predicted_failure" not in text and "failure_category" not in text
        checks["same_prompt_except_guidance"] &= len(set(normalized)) == 1
        label_prompt = build_prompt(record, "label_only", taxonomies)["user_prompt"]
        checks["label_only_omits_definitions"] &= all(
            str(category["definition"]) not in label_prompt
            for taxonomy in sources.values()
            for category in taxonomy["categories"]
        )
    maximum = max(prompt_tokens)
    checks["all_prompts_within_context"] = maximum + int(
        model["max_output_tokens"]
    ) < int(model["context_window"])
    guarded = math.ceil(
        sum(prompt_tokens)
        * float(config["cost_budget"]["input_token_safety_multiplier"])
    )
    prices = model["pricing_usd_per_million"]
    reserve = (
        guarded * float(prices["input"])
        + len(prompt_tokens)
        * int(model["max_output_tokens"])
        * float(prices["output"])
    ) / 1_000_000
    checks["conservative_cost_within_cap"] = reserve <= float(
        config["cost_budget"]["max_inference_usd"]
    )
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "READY_FOR_STEP_LOCALIZATION_INFERENCE" if not failed else "PREFLIGHT_FAILED",
        "api_calls_during_preflight": 0,
        "checks": checks,
        "failed_checks": failed,
        "dataset_counts": prepared["manifest"]["dataset_counts"],
        "total_trajectories": len(prepared["inputs"]),
        "planned_logical_predictions": len(prompt_tokens),
        "model": model["name"],
        "merged_taxonomy_policy": "reuse the newly generated frozen GPT-5 merge from the preceding attribution experiment",
        "cost_preflight": {
            "raw_prompt_tokens": sum(prompt_tokens),
            "guarded_prompt_tokens": guarded,
            "maximum_prompt_tokens": maximum,
            "projected_total_usd": reserve,
            "hard_cap_usd": float(config["cost_budget"]["max_inference_usd"]),
        },
    }
    write_json(HERE / "OFFLINE_PREFLIGHT.json", report)
    if failed:
        raise RuntimeError(f"STOP_BEFORE_API: offline preflight failed: {failed}")
    return report


def _rewrite_report(outputs: dict[str, Any]) -> None:
    run_dir = Path(outputs["run_directory"])
    with (run_dir / "results_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# AgentRx–AgentErrorBench failure-step localization results",
        "",
        "| Dataset | Condition | N | Exact step | Within ±1 | Mean rank error | Median rank error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['n']} | "
            f"{float(row['exact_step_accuracy']):.2%} | "
            f"{float(row['within_one_visible_step_accuracy']):.2%} | "
            f"{float(row['mean_absolute_visible_rank_error']):.3f} | "
            f"{float(row['median_absolute_visible_rank_error']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Primary metric is exact visible-step accuracy; within-one accuracy and rank distance are secondary.",
            "- All four conditions receive the identical complete visible candidate-step list.",
            "- Taxonomy categories are diagnostic guidance only and are never prediction targets.",
            "- Gold step is joined only after prediction and is never included in prompts.",
            "- Label-Only includes native category IDs/names but no definitions.",
            "- The same frozen GPT-5 merged taxonomy from the preceding attribution experiment is used for both datasets.",
            "- Results are reported as observed without enforcing an expected condition ordering.",
            "",
        ]
    )
    (run_dir / "FINAL_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _configure_engine() -> None:
    engine.HERE = HERE
    engine.DATASETS = DATASETS
    engine.CONDITIONS = CONDITIONS
    engine.CONTRASTS = CONTRASTS
    engine.prepare_data = _portable_prepare_data
    engine.load_source_taxonomies = load_source_taxonomies
    engine.obtain_merged_taxonomy = obtain_merged_taxonomy
    engine.offline_preflight = offline_preflight
    engine.build_prompt = build_prompt


def run_workflow(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
    analyze_only: bool = False,
) -> dict[str, Any]:
    _configure_engine()
    outputs = engine.run_workflow(
        config_path,
        preflight_only=preflight_only,
        analyze_only=analyze_only,
    )
    if not preflight_only:
        _rewrite_report(outputs)
    return outputs
