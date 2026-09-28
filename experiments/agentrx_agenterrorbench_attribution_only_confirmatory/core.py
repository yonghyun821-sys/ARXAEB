from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from experiments.agentrx_agenterrorbench_joint_attribution_step import core as joint_core
from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.datasets.agenterrorbench import (
    canonicalize_failure_type,
    canonicalize_module,
    load_agenterrorbench,
)
from taxonomy_experiment.datasets.agentrx import load_agentrx
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
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
MAX_PROVIDER_SEED = 2_147_483_647


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _escape_json_string_control_characters(value: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    replacements = {"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    for character in value:
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            continue
        if escaped:
            output.append(replacements.get(character, f"u{ord(character):04x}") if ord(character) < 0x20 else character)
            escaped = False
        elif character == "\\":
            output.append(character)
            escaped = True
        elif character == '"':
            output.append(character)
            in_string = False
        elif ord(character) < 0x20:
            output.append(replacements.get(character, f"\\u{ord(character):04x}"))
        else:
            output.append(character)
    return "".join(output)


class AttributionOutput(StrictModel):
    # Do not cap this field at schema-validation time. Some providers occasionally
    # place trailing JSON text inside the category string; accepting that malformed
    # value here lets the bounded invalid-label repair path request a clean ID.
    # Exact native-ID membership is still enforced before a prediction is saved.
    failure_category: str = Field(min_length=1)
    explanation: str = Field(min_length=1, max_length=700)
    evidence: str = Field(min_length=1, max_length=700)

    @field_validator("explanation", "evidence", mode="before")
    @classmethod
    def shorten_unscored_text(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) > 700:
            return value[:697].rstrip() + "..."
        return value

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> "AttributionOutput":
        text = bytes(json_data).decode("utf-8") if isinstance(json_data, (bytes, bytearray)) else json_data
        return super().model_validate_json(_escape_json_string_control_characters(text), **kwargs)


def _prompt(name: str) -> str:
    return (HERE / "prompts" / name).read_text(encoding="utf-8").strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _task_hash(value: str) -> str:
    return _sha256_bytes(" ".join(value.split()).casefold().encode("utf-8"))


def _canonical_step_id(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _provider_seed(value: int) -> int:
    return int(value) % MAX_PROVIDER_SEED or 1


def _freeze_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Frozen artifact differs; refusing overwrite: {path}")
        return
    write_json(path, value)


def _freeze_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if list(iter_jsonl(path)) != rows:
            raise FileExistsError(f"Frozen artifact differs; refusing overwrite: {path}")
        return
    write_jsonl(path, rows)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = {
        "agentrx": read_json(resolve_path(config, "source_agentrx_taxonomy")),
        "agenterrorbench": read_json(resolve_path(config, "source_agenterrorbench_taxonomy")),
    }
    merged = read_json(resolve_path(config, "source_merged_taxonomy"))
    merge_audit = read_json(resolve_path(config, "source_merged_taxonomy_audit"))
    validation = joint_core.validate_merged_taxonomy(merged, sources)
    frozen = resolve_path(config, "frozen")
    for dataset, taxonomy in sources.items():
        _freeze_json(frozen / f"{dataset}_taxonomy.json", taxonomy)
    _freeze_json(frozen / "merged_taxonomy.json", merged)
    _freeze_json(
        frozen / "merged_taxonomy_reuse_audit.json",
        {
            "status": "reused_preexisting_metadata_only_merge",
            "source_path": str(resolve_path(config, "source_merged_taxonomy")),
            "source_audit_path": str(resolve_path(config, "source_merged_taxonomy_audit")),
            "source_merge_audit": merge_audit,
            "prior_predictions_or_results_used_to_construct_merge": False,
            **validation,
        },
    )
    return {**sources, "merged": merged}


def _task_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["task_hash"] for row in rows)
    return {
        "trajectories": len(rows),
        "unique_task_hashes": len(counts),
        "duplicate_task_clusters": sum(value > 1 for value in counts.values()),
        "trajectories_in_duplicate_clusters": sum(value for value in counts.values() if value > 1),
        "maximum_cluster_size": max(counts.values(), default=0),
    }


def prepare_data(
    config: dict[str, Any], taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    root = config["_root"]
    rx_inputs_raw, rx_gold_raw, rx_loader_stats = load_agentrx(
        resolve_path(config, "agentrx_tau_trajectories"),
        resolve_path(config, "agentrx_tau_annotations"),
        resolve_path(config, "agentrx_magentic_trajectories"),
        resolve_path(config, "agentrx_magentic_annotations"),
        root,
    )
    rx_inputs = {row["trajectory_id"]: row for row in rx_inputs_raw}
    rx_by_name = {str(row["name"]): row for row in taxonomies["agentrx"]["categories"]}
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    exclusions: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}

    for reference in rx_gold_raw:
        source = rx_inputs[reference["trajectory_id"]]
        native_name = str(reference.get("failure_type") or "").strip()
        category = rx_by_name.get(native_name)
        candidate_ids = [_canonical_step_id(step.get("index")) for step in source.get("steps", [])]
        gold_step = _canonical_step_id(reference.get("critical_failure_step"))
        if category is None:
            exclusions["agentrx"]["noncanonical_native_label"] += 1
            continue
        if not candidate_ids or any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            exclusions["agentrx"]["invalid_visible_candidates"] += 1
            continue
        if gold_step not in candidate_ids:
            exclusions["agentrx"]["gold_step_not_visible"] += 1
            continue
        task = str(source.get("instruction") or "").strip()
        row = {
            "dataset": "agentrx",
            "trajectory_id": source["trajectory_id"],
            "domain": source["domain"],
            "task_hash": _task_hash(task),
            "trajectory_payload": {
                "task": task,
                "context": source.get("context") or {},
                "trajectory": source["steps"],
            },
        }
        inputs.append(row)
        gold.append(
            {
                "dataset": "agentrx",
                "trajectory_id": source["trajectory_id"],
                "gold_category_id": str(category["id"]),
                "gold_category_name": str(category["name"]),
                "task_hash": row["task_hash"],
            }
        )

    ae_inputs_raw, ae_gold_raw, ae_loader_stats = load_agenterrorbench(
        resolve_path(config, "agenterrorbench_trajectories"),
        resolve_path(config, "agenterrorbench_labels"),
        root,
    )
    ae_inputs = {row["trajectory_id"]: row for row in ae_inputs_raw}
    ae_by_id = {str(row["id"]): row for row in taxonomies["agenterrorbench"]["categories"]}
    for reference in ae_gold_raw:
        source = ae_inputs.get(reference["trajectory_id"])
        if source is None:
            exclusions["agenterrorbench"]["missing_visible_trajectory"] += 1
            continue
        failure_type = canonicalize_failure_type(reference.get("failure_type_raw"))
        module = canonicalize_module(reference.get("failure_module_raw"))
        category_id = f"{module}.{failure_type}" if module and failure_type else ""
        category = ae_by_id.get(category_id)
        if not failure_type:
            exclusions["agenterrorbench"]["missing_fine_grained_native_label"] += 1
            continue
        if category is None:
            exclusions["agenterrorbench"]["noncanonical_native_label"] += 1
            continue
        candidate_ids = [_canonical_step_id(step.get("index")) for step in source.get("steps", [])]
        gold_step = _canonical_step_id(reference.get("critical_failure_step"))
        if not candidate_ids or any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            exclusions["agenterrorbench"]["invalid_visible_candidates"] += 1
            continue
        if gold_step not in candidate_ids:
            exclusions["agenterrorbench"]["gold_step_not_visible"] += 1
            continue
        task = str(source.get("instruction") or "").strip()
        row = {
            "dataset": "agenterrorbench",
            "trajectory_id": source["trajectory_id"],
            "domain": source["domain"],
            "task_hash": _task_hash(task),
            "trajectory_payload": {
                "task": task,
                "context": source.get("context") or {},
                "trajectory": source["steps"],
            },
        }
        inputs.append(row)
        gold.append(
            {
                "dataset": "agenterrorbench",
                "trajectory_id": source["trajectory_id"],
                "gold_category_id": str(category["id"]),
                "gold_category_name": str(category["name"]),
                "task_hash": row["task_hash"],
            }
        )

    inputs.sort(key=lambda row: (row["dataset"], row["trajectory_id"]))
    gold.sort(key=lambda row: (row["dataset"], row["trajectory_id"]))
    keys = [(row["dataset"], row["trajectory_id"]) for row in inputs]
    gold_keys = [(row["dataset"], row["trajectory_id"]) for row in gold]
    if keys != gold_keys or len(keys) != len(set(keys)):
        raise ValueError("Prepared input/gold mapping is not one-to-one")
    counts = Counter(row["dataset"] for row in inputs)
    if counts != Counter({"agentrx": 73, "agenterrorbench": 171}):
        raise ValueError(f"Unexpected frozen sample counts: {dict(counts)}")

    prior_inputs = list(iter_jsonl(resolve_path(config, "prior_joint_inputs")))
    prior_keys = {(row["dataset"], row["trajectory_id"], row["task_hash"]) for row in prior_inputs}
    current_keys = {(row["dataset"], row["trajectory_id"], row["task_hash"]) for row in inputs}
    if current_keys != prior_keys:
        raise ValueError("Confirmatory sample differs from the prior joint experiment")

    prepared_dir = resolve_path(config, "prepared")
    input_path = prepared_dir / "frozen_inputs.jsonl"
    gold_path = prepared_dir / "frozen_gold.jsonl"
    _freeze_jsonl(input_path, inputs)
    _freeze_jsonl(gold_path, gold)
    manifest = {
        "version": 1,
        "confirmatory_protocol": "attribution-only; no step target or candidate list in prompt",
        "sampling": "exact reuse of the prior joint-eligible 73 AgentRx plus 171 AgentErrorBench trajectories",
        "conditions": list(CONDITIONS),
        "dataset_counts": dict(sorted(counts.items())),
        "total_trajectories": len(inputs),
        "planned_logical_predictions": len(inputs) * len(CONDITIONS),
        "inputs_sha256": _sha256_file(input_path),
        "gold_sha256": _sha256_file(gold_path),
        "prior_joint_input_sha256": _sha256_file(resolve_path(config, "prior_joint_inputs")),
        "exact_prior_joint_key_and_task_hash_match": True,
        "dataset_audits": {
            "agentrx": {
                "included": counts["agentrx"],
                "exclusions": dict(sorted(exclusions["agentrx"].items())),
                "task_structure": _task_stats([row for row in inputs if row["dataset"] == "agentrx"]),
                "source_loader_stats": rx_loader_stats,
            },
            "agenterrorbench": {
                "included": counts["agenterrorbench"],
                "exclusions": dict(sorted(exclusions["agenterrorbench"].items())),
                "task_structure": _task_stats([row for row in inputs if row["dataset"] == "agenterrorbench"]),
                "source_loader_stats": ae_loader_stats,
            },
        },
    }
    _freeze_json(prepared_dir / "sample_manifest.json", manifest)
    return {"inputs": inputs, "gold": gold, "manifest": manifest}


def _native_categories(dataset: str, taxonomies: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return list(taxonomies[dataset]["categories"])


def _native_ids(dataset: str, taxonomies: dict[str, dict[str, Any]]) -> list[str]:
    return [str(row["id"]) for row in _native_categories(dataset, taxonomies)]


def _output_label_block(dataset: str, taxonomies: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        f"- {row['id']} -> {row['name']}" for row in _native_categories(dataset, taxonomies)
    )


def _guidance_for(
    dataset: str, condition: str, taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if condition == "label_only":
        return None
    if condition == "own_taxonomy":
        return taxonomies[dataset]
    if condition == "foreign_taxonomy":
        return taxonomies["agenterrorbench" if dataset == "agentrx" else "agentrx"]
    if condition == "merged_taxonomy":
        return taxonomies["merged"]
    raise ValueError(f"Unknown condition: {condition}")


def _guidance_text(taxonomy: dict[str, Any] | None) -> str:
    if taxonomy is None:
        return "No category definitions are supplied in the label-only condition."
    payload = {
        "taxonomy_id": taxonomy["taxonomy_id"],
        "name": taxonomy["name"],
        "categories": [
            {
                "id": row["id"],
                "name": row["name"],
                "module": row.get("module"),
                "definition": row["definition"],
            }
            for row in taxonomy["categories"]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_prompt(
    record: dict[str, Any], condition: str, taxonomies: dict[str, dict[str, Any]]
) -> tuple[str, str]:
    guidance = _guidance_for(record["dataset"], condition, taxonomies)
    user = _prompt("attribution_user.txt").format(
        output_labels=_output_label_block(record["dataset"], taxonomies),
        diagnostic_guidance=_guidance_text(guidance),
        trajectory=json.dumps(record["trajectory_payload"], ensure_ascii=False, indent=2),
    )
    return _prompt("attribution_system.txt"), user


def run_offline_preflight(
    config: dict[str, Any],
    prepared: dict[str, Any],
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "dataset_counts_exact": prepared["manifest"]["dataset_counts"]
        == {"agenterrorbench": 171, "agentrx": 73},
        "exact_prior_joint_sample_reuse": prepared["manifest"]["exact_prior_joint_key_and_task_hash_match"],
        "four_conditions": tuple(config["experiment"]["conditions"]) == CONDITIONS,
        "planned_rows_976": prepared["manifest"]["planned_logical_predictions"] == 976,
        "gold_absent_from_inputs": all(
            not {"gold_category_id", "gold_category_name", "gold_step_id"}.intersection(row)
            for row in prepared["inputs"]
        ),
        "all_gold_ids_native": True,
        "same_output_space_across_conditions": True,
        "only_guidance_changes": True,
        "no_step_prediction_instruction": True,
        "label_only_has_no_definitions": True,
    }
    gold_index = {(row["dataset"], row["trajectory_id"]): row for row in prepared["gold"]}
    for record in prepared["inputs"]:
        reference = gold_index[(record["dataset"], record["trajectory_id"])]
        checks["all_gold_ids_native"] &= reference["gold_category_id"] in _native_ids(record["dataset"], taxonomies)
        prompts = {}
        for condition in CONDITIONS:
            system, user = build_prompt(record, condition, taxonomies)
            prompts[condition] = system + "\n" + user
            checks["no_step_prediction_instruction"] &= not re.search(
                r"predicted_step|candidate_step|gold_step|failure step|step localization",
                prompts[condition],
                flags=re.IGNORECASE,
            )
        label_block = _output_label_block(record["dataset"], taxonomies)
        checks["same_output_space_across_conditions"] &= all(
            label_block in prompt for prompt in prompts.values()
        )
        stripped = {
            condition: re.sub(
                r"<DIAGNOSTIC_GUIDANCE>.*?</DIAGNOSTIC_GUIDANCE>",
                "<DIAGNOSTIC_GUIDANCE>[ONLY_ALLOWED_DIFFERENCE]</DIAGNOSTIC_GUIDANCE>",
                prompt,
                flags=re.DOTALL,
            )
            for condition, prompt in prompts.items()
        }
        checks["only_guidance_changes"] &= len(set(stripped.values())) == 1
        checks["label_only_has_no_definitions"] &= all(
            str(category["definition"]) not in prompts["label_only"]
            for taxonomy in (taxonomies["agentrx"], taxonomies["agenterrorbench"], taxonomies["merged"])
            for category in taxonomy["categories"]
        )

    model = config["models"]["attribution"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    prompt_tokens = []
    for record in prepared["inputs"]:
        for condition in CONDITIONS:
            system, user = build_prompt(record, condition, taxonomies)
            prompt_tokens.append(counter.count(system + "\n" + user))
    maximum = max(prompt_tokens)
    checks["all_prompts_within_context"] = maximum + int(model["max_output_tokens"]) < int(model["context_window"])
    prices = model["pricing_usd_per_million"]
    guarded_input = math.ceil(
        sum(prompt_tokens) * float(config["cost_budget"]["input_token_safety_multiplier"])
    )
    reserve = (
        guarded_input * float(prices["input"])
        + len(prompt_tokens) * int(model["max_output_tokens"]) * float(prices["output"])
    ) / 1_000_000
    checks["conservative_cost_within_cap"] = reserve <= float(config["cost_budget"]["max_inference_usd"])
    failed = [name for name, passed in checks.items() if not passed]
    audit = {
        "status": "READY_FOR_ATTRIBUTION_ONLY_INFERENCE" if not failed else "PREFLIGHT_FAILED",
        "api_calls_during_preflight": 0,
        "checks": checks,
        "failed_checks": failed,
        "dataset_counts": prepared["manifest"]["dataset_counts"],
        "total_trajectories": len(prepared["inputs"]),
        "planned_logical_predictions": len(prompt_tokens),
        "model": model["name"],
        "merge_policy": "reuse frozen GPT-5 metadata-only merge; no new merge API call",
        "cost_preflight": {
            "raw_prompt_tokens": sum(prompt_tokens),
            "guarded_prompt_tokens": guarded_input,
            "maximum_prompt_tokens": maximum,
            "conservative_reserve_usd": reserve,
            "hard_cap_usd": float(config["cost_budget"]["max_inference_usd"]),
        },
    }
    write_json(resolve_path(config, "experiment_root") / "OFFLINE_PREFLIGHT.json", audit)
    if failed:
        raise RuntimeError(f"Offline preflight failed: {failed}")
    return audit


def _prediction_key(dataset: str, trajectory_id: str, condition: str) -> str:
    return f"{dataset}::{trajectory_id}::{condition}"


def _ordered_pairs(inputs: list[dict[str, Any]], seed: int) -> list[tuple[dict[str, Any], str]]:
    pairs = []
    for record in inputs:
        conditions = list(CONDITIONS)
        local = int(_sha256_bytes(f"{seed}|{record['dataset']}|{record['trajectory_id']}".encode())[:16], 16)
        random.Random(local).shuffle(conditions)
        pairs.extend((record, condition) for condition in conditions)
    random.Random(seed).shuffle(pairs)
    return pairs


def _canonical_category(value: str, valid_ids: list[str]) -> str | None:
    exact = str(value).strip()
    if exact in valid_ids:
        return exact
    normalized = re.sub(r"[^a-z0-9]+", "", exact.casefold())
    matches = [row for row in valid_ids if re.sub(r"[^a-z0-9]+", "", row.casefold()) == normalized]
    return matches[0] if len(matches) == 1 else None


def _call_attribution(
    *,
    llm: CachedLLM,
    config: dict[str, Any],
    system: str,
    user: str,
    valid_ids: list[str],
    seed: int,
) -> tuple[AttributionOutput, dict[str, Any], int, int, bool]:
    model = config["models"]["attribution"]
    invalid_retries = 0
    truncation_retries = 0
    prompt = user
    max_tokens = int(model["max_output_tokens"])
    while True:
        try:
            parsed, call = llm.call(
                model=model["name"],
                system_prompt=system,
                user_prompt=prompt,
                temperature=model.get("temperature"),
                reasoning_effort=model.get("reasoning_effort"),
                seed=_provider_seed(seed + invalid_retries + truncation_retries),
                provider=model.get("provider"),
                max_output_tokens=max_tokens,
                response_model=AttributionOutput,
            )
        except TruncatedResponseError:
            if truncation_retries >= int(config["execution"]["truncation_retries"]):
                raise
            truncation_retries += 1
            max_tokens = int(model["truncation_retry_max_output_tokens"])
            prompt = user + "\n\nReturn the same three fields concisely and no text outside JSON."
            continue
        category = _canonical_category(parsed.failure_category, valid_ids)
        if category is not None:
            return (
                parsed.model_copy(update={"failure_category": category}),
                call,
                invalid_retries,
                truncation_retries,
                False,
            )
        if invalid_retries >= int(config["execution"]["invalid_output_retries"]):
            # Preserve strict exact-ID scoring without aborting the complete run.
            # The invalid raw value is retained and will necessarily score wrong.
            return parsed, call, invalid_retries, truncation_retries, True
        invalid_retries += 1
        prompt = (
            user
            + "\n\nYour prior category was invalid. Copy exactly one category ID from: "
            + json.dumps(valid_ids, ensure_ascii=False)
            + ". Return only the requested JSON object."
        )


def run_predictions(
    config: dict[str, Any],
    prepared: dict[str, Any],
    taxonomies: dict[str, dict[str, Any]],
    run_dir: Path,
) -> Path:
    path = run_dir / "predictions.jsonl"
    existing = list(iter_jsonl(path)) if path.exists() else []
    completed = {str(row["prediction_key"]): row for row in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate prediction keys in resumable output")
    pairs = _ordered_pairs(prepared["inputs"], int(config["experiment"]["seed"]))
    expected = {
        _prediction_key(record["dataset"], record["trajectory_id"], condition)
        for record, condition in pairs
    }
    if set(completed) - expected:
        raise ValueError("Resumable predictions contain keys outside the frozen sample")

    smoke_keys = set()
    count = int(config["execution"]["smoke_trajectories_per_dataset"])
    for dataset in DATASETS:
        rows = sorted(
            (row for row in prepared["inputs"] if row["dataset"] == dataset),
            key=lambda row: row["trajectory_id"],
        )
        smoke_keys.update((row["dataset"], row["trajectory_id"]) for row in rows[:count])
    smoke = [pair for pair in pairs if (pair[0]["dataset"], pair[0]["trajectory_id"]) in smoke_keys]
    main = [pair for pair in pairs if pair not in smoke]
    print(f"[attr-only] smoke gate: {len(smoke)} rows; completed rows reuse cache/results", flush=True)
    llm = CachedLLM(config, resolve_path(config, "cache") / "attribution")
    model = config["models"]["attribution"]
    progress_every = int(config["execution"]["progress_every"])

    def execute(batch: list[tuple[dict[str, Any], str]], phase: str) -> None:
        for record, condition in batch:
            key = _prediction_key(record["dataset"], record["trajectory_id"], condition)
            if key in completed:
                continue
            valid_ids = _native_ids(record["dataset"], taxonomies)
            system, user = build_prompt(record, condition, taxonomies)
            seed = _provider_seed(int(_sha256_bytes(f"{config['experiment']['seed']}|{key}".encode())[:8], 16))
            parsed, call, invalid_retries, truncation_retries, terminal_invalid = _call_attribution(
                llm=llm,
                config=config,
                system=system,
                user=user,
                valid_ids=valid_ids,
                seed=seed,
            )
            row = {
                "prediction_key": key,
                "dataset": record["dataset"],
                "trajectory_id": record["trajectory_id"],
                "domain": record["domain"],
                "task_hash": record["task_hash"],
                "condition": condition,
                "predicted_category_id": parsed.failure_category,
                "explanation": parsed.explanation,
                "evidence": parsed.evidence,
                "model": model["name"],
                "prompt_version": config["experiment"]["prompt_version"],
                "cache_key": call["cache_key"],
                "cache_hit": call["cache_hit"],
                "response_metadata": call.get("response_metadata") or {},
                "invalid_output_retries": invalid_retries,
                "truncation_retries": truncation_retries,
                "terminal_invalid_output": terminal_invalid,
            }
            append_jsonl(path, row)
            completed[key] = row
            done = len(completed)
            if done % progress_every == 0 or done == len(expected):
                print(
                    f"[attr-only:{phase}] completed={done}/{len(expected)} "
                    f"dataset={record['dataset']} condition={condition}",
                    flush=True,
                )

    execute(smoke, "smoke")
    if len(completed) < len(expected):
        print(f"[attr-only] full run: {len(expected)} rows; smoke rows are reused", flush=True)
    execute(main, "main")
    if set(completed) != expected:
        raise RuntimeError("Prediction coverage incomplete")
    return path


def _macro_f1(rows: list[dict[str, Any]]) -> float:
    labels = sorted({row["gold_category_id"] for row in rows})
    scores = []
    for label in labels:
        tp = sum(row["gold_category_id"] == label and row["predicted_category_id"] == label for row in rows)
        fp = sum(row["gold_category_id"] != label and row["predicted_category_id"] == label for row in rows)
        fn = sum(row["gold_category_id"] == label and row["predicted_category_id"] != label for row in rows)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return statistics.fmean(scores) if scores else 0.0


def _exact_mcnemar(improved: int, regressed: int) -> float:
    total = improved + regressed
    if total == 0:
        return 1.0
    lower = min(improved, regressed)
    return min(1.0, 2 * sum(math.comb(total, index) for index in range(lower + 1)) / (2**total))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _cluster_ci(
    paired: list[dict[str, Any]], treatment: str, control: str, config: dict[str, Any], seed_salt: str
) -> tuple[float, float]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        clusters[row["task_hash"]].append(row)
    ids = sorted(clusters)
    rng = random.Random(int(config["analysis"]["bootstrap_seed"]) + int(_sha256_bytes(seed_salt.encode())[:8], 16))
    values = []
    for _ in range(int(config["analysis"]["bootstrap_resamples"])):
        sampled = [rng.choice(ids) for _ in ids]
        differences = [
            int(row[treatment]["correct"]) - int(row[control]["correct"])
            for cluster in sampled
            for row in clusters[cluster]
        ]
        values.append(statistics.fmean(differences))
    alpha = float(config["analysis"]["alpha"])
    return _percentile(values, alpha / 2), _percentile(values, 1 - alpha / 2)


def _cost_summary(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["cost_budget"]["ledger_path"])
    if not path.is_absolute():
        path = config["_root"] / path
    rows = list(iter_jsonl(path)) if path.exists() else []
    total = sum(float(row.get("cost_usd", 0.0)) for row in rows)
    return {
        "ledger_path": str(path),
        "billed_api_responses": len(rows),
        "cost_usd": total,
        "configured_cap_usd": float(config["cost_budget"]["max_inference_usd"]),
        "within_cap": total <= float(config["cost_budget"]["max_inference_usd"]),
    }


def analyze_results(
    config: dict[str, Any], prepared: dict[str, Any], run_dir: Path
) -> dict[str, str]:
    predictions = list(iter_jsonl(run_dir / "predictions.jsonl"))
    expected = len(prepared["inputs"]) * len(CONDITIONS)
    if len(predictions) != expected:
        raise RuntimeError(f"Expected {expected} predictions, found {len(predictions)}")
    gold = {(row["dataset"], row["trajectory_id"]): row for row in prepared["gold"]}
    scored = []
    for prediction in predictions:
        reference = gold[(prediction["dataset"], prediction["trajectory_id"])]
        scored.append(
            {
                **prediction,
                "gold_category_id": reference["gold_category_id"],
                "gold_category_name": reference["gold_category_name"],
                "correct": prediction["predicted_category_id"] == reference["gold_category_id"],
            }
        )
    scored.sort(key=lambda row: (row["dataset"], row["trajectory_id"], row["condition"]))
    write_jsonl(run_dir / "predictions_scored.jsonl", scored)
    _write_csv(
        run_dir / "predictions_scored.csv",
        scored,
        [
            "dataset", "trajectory_id", "domain", "task_hash", "condition",
            "gold_category_id", "gold_category_name", "predicted_category_id", "correct",
            "explanation", "evidence", "model", "invalid_output_retries",
            "truncation_retries", "cache_hit", "cache_key",
            "terminal_invalid_output",
        ],
    )

    summaries = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            subset = [row for row in scored if row["dataset"] == dataset and row["condition"] == condition]
            correct = sum(row["correct"] for row in subset)
            summaries.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(subset),
                    "correct": correct,
                    "accuracy": correct / len(subset),
                    "macro_f1": _macro_f1(subset),
                    "terminal_invalid_outputs": sum(
                        bool(row.get("terminal_invalid_output", False)) for row in subset
                    ),
                }
            )
    _write_csv(
        run_dir / "condition_results.csv",
        summaries,
        [
            "dataset", "condition", "n", "correct", "accuracy", "macro_f1",
            "terminal_invalid_outputs",
        ],
    )

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scored:
        grouped[(row["dataset"], row["trajectory_id"])][row["condition"]] = row
    contrasts = []
    transitions = []
    for dataset in DATASETS:
        paired = [
            {**conditions, "task_hash": conditions["label_only"]["task_hash"]}
            for (row_dataset, _), conditions in grouped.items()
            if row_dataset == dataset and set(conditions) == set(CONDITIONS)
        ]
        for treatment, control in CONTRASTS:
            deltas = [int(row[treatment]["correct"]) - int(row[control]["correct"]) for row in paired]
            improved = sum(value == 1 for value in deltas)
            regressed = sum(value == -1 for value in deltas)
            low, high = _cluster_ci(paired, treatment, control, config, f"{dataset}|{treatment}|{control}")
            contrasts.append(
                {
                    "dataset": dataset,
                    "treatment": treatment,
                    "control": control,
                    "n": len(paired),
                    "treatment_accuracy": statistics.fmean(int(row[treatment]["correct"]) for row in paired),
                    "control_accuracy": statistics.fmean(int(row[control]["correct"]) for row in paired),
                    "paired_delta": statistics.fmean(deltas),
                    "cluster_bootstrap_ci_low": low,
                    "cluster_bootstrap_ci_high": high,
                    "improved": improved,
                    "regressed": regressed,
                    "unchanged": len(paired) - improved - regressed,
                    "mcnemar_exact_p": _exact_mcnemar(improved, regressed),
                }
            )
            transitions.append(
                {
                    "dataset": dataset,
                    "treatment": treatment,
                    "control": control,
                    "prediction_changed": sum(
                        row[treatment]["predicted_category_id"] != row[control]["predicted_category_id"]
                        for row in paired
                    ),
                    "improved": improved,
                    "regressed": regressed,
                    "n": len(paired),
                }
            )
    _write_csv(
        run_dir / "paired_contrasts.csv",
        contrasts,
        [
            "dataset", "treatment", "control", "n", "treatment_accuracy",
            "control_accuracy", "paired_delta", "cluster_bootstrap_ci_low",
            "cluster_bootstrap_ci_high", "improved", "regressed", "unchanged",
            "mcnemar_exact_p",
        ],
    )
    _write_csv(
        run_dir / "condition_transitions.csv",
        transitions,
        ["dataset", "treatment", "control", "n", "prediction_changed", "improved", "regressed"],
    )

    per_class = []
    for dataset in DATASETS:
        labels = sorted({row["gold_category_id"] for row in scored if row["dataset"] == dataset})
        for condition in CONDITIONS:
            subset = [row for row in scored if row["dataset"] == dataset and row["condition"] == condition]
            for label in labels:
                support = sum(row["gold_category_id"] == label for row in subset)
                tp = sum(row["gold_category_id"] == label and row["predicted_category_id"] == label for row in subset)
                fp = sum(row["gold_category_id"] != label and row["predicted_category_id"] == label for row in subset)
                fn = support - tp
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / support if support else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                per_class.append(
                    {
                        "dataset": dataset, "condition": condition, "category_id": label,
                        "support": support, "true_positive": tp, "precision": precision,
                        "recall": recall, "f1": f1,
                    }
                )
    _write_csv(
        run_dir / "per_class_metrics.csv",
        per_class,
        ["dataset", "condition", "category_id", "support", "true_positive", "precision", "recall", "f1"],
    )

    prior_comparison = []
    prior_path = resolve_path(config, "prior_joint_scored_predictions")
    if prior_path.exists():
        with prior_path.open("r", encoding="utf-8-sig", newline="") as handle:
            prior = list(csv.DictReader(handle))
        prior_condition_map = {"native_merged_taxonomy": "merged_taxonomy"}
        for dataset in DATASETS:
            for condition in CONDITIONS:
                prior_name = "native_merged_taxonomy" if condition == "merged_taxonomy" else condition
                old = [row for row in prior if row["dataset"] == dataset and row["condition"] == prior_name]
                new = [row for row in scored if row["dataset"] == dataset and row["condition"] == condition]
                old_accuracy = statistics.fmean(row["cause_correct"].casefold() == "true" for row in old)
                new_accuracy = statistics.fmean(row["correct"] for row in new)
                prior_comparison.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "joint_prompt_cause_accuracy": old_accuracy,
                        "attribution_only_accuracy": new_accuracy,
                        "attribution_only_minus_joint": new_accuracy - old_accuracy,
                    }
                )
    comparison_path = run_dir / "joint_vs_attribution_only.csv"
    _write_csv(
        comparison_path,
        prior_comparison,
        ["dataset", "condition", "joint_prompt_cause_accuracy", "attribution_only_accuracy", "attribution_only_minus_joint"],
    )

    cost = _cost_summary(config)
    write_json(run_dir / "cost_summary.json", cost)
    by_key = {(row["dataset"], row["condition"]): row for row in summaries}
    lines = [
        "# AgentRx–AgentErrorBench Attribution-Only Confirmatory Results",
        "",
        "이 실험은 prior joint cause+step 표본을 정확히 재사용하되 step target과 step instruction을 완전히 제거했다.",
        "모든 조건은 동일한 target-native output IDs를 사용하고 diagnostic taxonomy guidance만 다르다.",
        "",
        "## Main results",
        "",
        "| Dataset | Condition | N | Accuracy | Macro-F1 | Terminal invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            row = by_key[(dataset, condition)]
            lines.append(
                f"| {dataset} | {condition} | {row['n']} | {row['accuracy']:.2%} | "
                f"{row['macro_f1']:.4f} | {row['terminal_invalid_outputs']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Exact native-category ID accuracy is primary; macro-F1 is secondary.",
            "- A category still outside the native ID set after bounded retries is retained as terminal invalid and scored wrong; it is never semantically remapped.",
            "- Gold was joined only after all predictions; no step target or step candidate list appeared in prompts.",
            "- The merged taxonomy is the frozen GPT-5 metadata-only artifact created before the prior joint predictions.",
            "- `joint_vs_attribution_only.csv` is a descriptive protocol comparison, not an independent replication.",
            "- Results must be reported even if they do not follow an expected condition ordering.",
            "",
            "## Cost",
            "",
            f"- Recorded API spend: `${cost['cost_usd']:.6f}` / `${cost['configured_cap_usd']:.2f}` cap",
            f"- Billed API responses: `{cost['billed_api_responses']}`",
            "",
        ]
    )
    report = run_dir / "FINAL_RESULTS.md"
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "final_results": str(report),
        "condition_results": str(run_dir / "condition_results.csv"),
        "paired_contrasts": str(run_dir / "paired_contrasts.csv"),
        "protocol_comparison": str(comparison_path),
        "scored_predictions": str(run_dir / "predictions_scored.csv"),
        "cost_summary": str(run_dir / "cost_summary.json"),
    }


def run_workflow(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
    analyze_only: bool = False,
) -> dict[str, str]:
    if preflight_only and analyze_only:
        raise ValueError("--preflight-only and --analyze-only cannot be combined")
    config = load_config(config_path)
    taxonomies = load_frozen_taxonomies(config)
    prepared = prepare_data(config, taxonomies)
    audit = run_offline_preflight(config, prepared, taxonomies)
    if preflight_only:
        return {
            "status": audit["status"],
            "offline_preflight": str(resolve_path(config, "experiment_root") / "OFFLINE_PREFLIGHT.json"),
            "planned_predictions": str(audit["planned_logical_predictions"]),
            "projected_total_usd": f"{audit['cost_preflight']['conservative_reserve_usd']:.6f}",
        }
    identity = {
        "experiment": config["experiment"],
        "model": config["models"]["attribution"],
        "inputs_sha256": prepared["manifest"]["inputs_sha256"],
        "gold_sha256": prepared["manifest"]["gold_sha256"],
        "merged_taxonomy_sha256": _sha256_json(taxonomies["merged"]),
        "scoring": "exact_native_category_id_no_relabeler",
    }
    run_id = "attr-only-" + _sha256_json(identity)[:16]
    run_dir = resolve_path(config, "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_json(run_dir / "run_identity.json", identity)
    if not analyze_only:
        run_predictions(config, prepared, taxonomies, run_dir)
    outputs = analyze_results(config, prepared, run_dir)
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "logical_predictions": 976,
            "model": config["models"]["attribution"]["name"],
            "outputs": outputs,
        },
    )
    return {"status": "COMPLETE", "run_id": run_id, "run_directory": str(run_dir), **outputs}
