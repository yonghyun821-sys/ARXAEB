from __future__ import annotations

import csv
import copy
import json
import math
from pathlib import Path
from typing import Any

from experiments.agentrx_agenterrorbench_attribution_only_confirmatory import core as base
from experiments.agentrx_agenterrorbench_joint_attribution_step import core as merge_impl
from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, read_json, write_json
from taxonomy_experiment.token_count import TokenCounter


HERE = Path(__file__).resolve().parent


def _portable_prepare_data(
    config: dict[str, Any], taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Load the exact frozen 73 + 171 sample bundled with semifinal."""
    del taxonomies
    prepared = resolve_path(config, "prepared")
    inputs = list(iter_jsonl(prepared / "frozen_inputs.jsonl"))
    gold = list(iter_jsonl(prepared / "frozen_gold.jsonl"))
    manifest = read_json(prepared / "sample_manifest.json")
    input_keys = {(row["dataset"], row["trajectory_id"]) for row in inputs}
    gold_keys = {(row["dataset"], row["trajectory_id"]) for row in gold}
    if input_keys != gold_keys or len(inputs) != 244:
        raise RuntimeError("Bundled frozen attribution sample is incomplete")
    if manifest.get("dataset_counts") != {"agenterrorbench": 171, "agentrx": 73}:
        raise RuntimeError("Bundled attribution manifest does not match 73 + 171")
    if any("gold_category_id" in row or "gold_step_id" in row for row in inputs):
        raise RuntimeError("Gold information leaked into bundled attribution inputs")
    return {"inputs": inputs, "gold": gold, "manifest": manifest}


def _final_protocol_guidance(taxonomy: dict[str, Any] | None) -> str:
    """Match the final AEGIS-Who&When serialization exactly."""
    if taxonomy is None:
        return "[NONE]"
    payload = {
        "taxonomy_id": taxonomy["taxonomy_id"],
        "name": taxonomy["name"],
        "categories": [
            {
                "id": row["id"],
                "name": row["name"],
                "definition": row["definition"],
                "module": row.get("module"),
            }
            for row in taxonomy["categories"]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_clean_report(outputs: dict[str, str]) -> None:
    results_path = Path(outputs["condition_results"])
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cost_path = Path(outputs["cost_summary"])
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else {}
    lines = [
        "# AgentRx–AgentErrorBench final-protocol 결과",
        "",
        "AEGIS–Who&When 최종 실험과 같은 4조건 attribution-only protocol을 적용한 결과다.",
        "모든 조건은 동일한 dataset-native output label 목록을 사용하며 diagnostic guidance만 달라진다.",
        "",
        "| Dataset | Condition | N | Accuracy | Macro-F1 | Terminal invalid |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['n']} | "
            f"{float(row['accuracy']):.2%} | {float(row['macro_f1']):.4f} | "
            f"{row['terminal_invalid_outputs']} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "- 주지표는 native category ID exact-match accuracy이며 Macro-F1은 보조지표다.",
            "- gold category는 모든 prediction이 끝난 뒤 결합한다.",
            "- prompt에는 gold label, gold reason, gold step 또는 step-localization target이 들어가지 않는다.",
            "- Foreign/Merged taxonomy는 진단 guidance일 뿐 output label space를 바꾸지 않는다.",
            "- 기대한 조건 순서와 다르더라도 결과를 그대로 보고한다.",
            "",
            "## 비용",
            "",
            f"- 기록된 API 비용: `${float(cost.get('cost_usd', 0.0)):.6f}`",
            f"- 설정된 hard cap: `${float(cost.get('configured_cap_usd', 0.0)):.2f}`",
            "",
        ]
    )
    Path(outputs["final_results"]).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_workflow(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
    analyze_only: bool = False,
) -> dict[str, str]:
    # Reuse the already audited AgentRx/AEB loaders, freezing and statistical
    # analysis, while binding prompts, artifacts, cache and ledger to this new
    # experiment directory. No prior predictions are imported.
    base.HERE = HERE
    base._guidance_text = _final_protocol_guidance
    base.prepare_data = _portable_prepare_data
    config = load_config(config_path)

    sources = merge_impl.load_source_taxonomies(config)
    if preflight_only:
        # Keep preflight strictly offline. A maximum-size lossless proxy gives a
        # conservative attribution-token estimate without pre-empting the fresh
        # GPT-5 semantic merge requested for the real run.
        proxy = merge_impl._synthetic_merge(sources)
        prepared = base.prepare_data(config, {**sources, "merged": proxy})
        audit = base.run_offline_preflight(config, prepared, {**sources, "merged": proxy})
        merge_model = config["models"]["merge"]
        system, user = merge_impl._merge_prompts(sources)
        counter = TokenCounter.for_model(
            merge_model["name"], merge_model.get("tokenizer_encoding")
        )
        merge_input_tokens = counter.count(system + "\n" + user)
        guarded_merge_input = math.ceil(
            merge_input_tokens
            * float(config["merge_cost_budget"]["input_token_safety_multiplier"])
        )
        prices = merge_model["pricing_usd_per_million"]
        merge_reserve = (
            guarded_merge_input * float(prices["input"])
            + int(merge_model["max_output_tokens"]) * float(prices["output"])
        ) / 1_000_000
        # The portable bundle commits the validated GPT-5 merge used by the
        # original run. Reusing it avoids a nondeterministic regeneration call.
        merge_reserve = 0.0
        total_reserve = float(audit["cost_preflight"]["conservative_reserve_usd"])
        total_cap = float(config["cost_budget"]["max_total_usd"])
        audit.update(
            {
                "status": (
                    "READY_FOR_FROZEN_GPT5_MERGE_REUSE_AND_INFERENCE"
                    if total_reserve <= total_cap
                    else "PREFLIGHT_FAILED"
                ),
                "merge_policy": (
                    "reuse the bundled GPT-5 metadata-only merge after "
                    "deterministic exact-once provenance validation"
                ),
                "merge_api_calls_during_preflight": 0,
                "fresh_merge_plan": {
                    "model": merge_model["name"],
                    "reasoning_effort": merge_model.get("reasoning_effort"),
                    "raw_prompt_tokens": merge_input_tokens,
                    "guarded_prompt_tokens": guarded_merge_input,
                    "maximum_output_tokens": int(merge_model["max_output_tokens"]),
                    "conservative_reserve_usd": merge_reserve,
                    "merge_cap_usd": float(config["merge_cost_budget"]["max_usd"]),
                    "taxonomy_metadata_only": True,
                    "bundled_frozen_merge_reused": True,
                    "trajectory_gold_predictions_or_results_used": False,
                },
                "combined_conservative_reserve_usd": total_reserve,
                "combined_hard_cap_usd": total_cap,
            }
        )
        write_json(resolve_path(config, "experiment_root") / "OFFLINE_PREFLIGHT.json", audit)
        if audit["status"] == "PREFLIGHT_FAILED":
            raise RuntimeError("Combined merge + inference reserve exceeds the hard cap")
        return {
            "status": audit["status"],
            "offline_preflight": str(
                resolve_path(config, "experiment_root") / "OFFLINE_PREFLIGHT.json"
            ),
            "planned_predictions": str(audit["planned_logical_predictions"]),
            "projected_merge_usd": f"{merge_reserve:.6f}",
            "projected_total_usd": f"{total_reserve:.6f}",
        }

    merge_config = copy.deepcopy(config)
    merge_config["cost_budget"]["max_inference_usd"] = float(
        config["merge_cost_budget"]["max_usd"]
    )
    merge_config["cost_budget"]["ledger_path"] = config["merge_cost_budget"][
        "ledger_path"
    ]
    merged, merge_audit = merge_impl.obtain_merged_taxonomy(
        merge_config,
        sources,
        allow_generation=not analyze_only,
    )
    # Refuse to begin attribution if the bundled artifact fails the exact same
    # deterministic provenance validation used by the original study.
    merge_impl.validate_merged_taxonomy(merged, sources)
    outputs = base.run_workflow(
        config_path,
        preflight_only=False,
        analyze_only=analyze_only,
    )
    _write_clean_report(outputs)
    outputs["status"] = "FROZEN_GPT5_MERGE_REUSED_AND_FINAL_PROTOCOL_COMPLETE"
    outputs["merged_taxonomy"] = str(resolve_path(config, "source_merged_taxonomy"))
    outputs["merge_audit"] = str(resolve_path(config, "source_merged_taxonomy_audit"))
    outputs["merge_api_ledger"] = str(
        config["_root"] / config["merge_cost_budget"]["ledger_path"]
    )
    return outputs
