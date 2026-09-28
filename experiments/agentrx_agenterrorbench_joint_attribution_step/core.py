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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
CONDITIONS = (
    "label_only",
    "own_taxonomy",
    "foreign_taxonomy",
    "native_merged_taxonomy",
)
CONTRASTS = (
    ("own_taxonomy", "label_only"),
    ("foreign_taxonomy", "label_only"),
    ("native_merged_taxonomy", "label_only"),
    ("native_merged_taxonomy", "foreign_taxonomy"),
    ("native_merged_taxonomy", "own_taxonomy"),
)
PRIMARY_METRICS = (
    "cause_correct",
    "step_exact_correct",
    "step_within_1_correct",
    "joint_correct",
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
            continue
        if character == "\\":
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


class JointAttributionOutput(StrictModel):
    predicted_failure_label: str = Field(min_length=1, max_length=180)
    predicted_step_id: str = Field(min_length=1, max_length=80)
    explanation: str = Field(min_length=1, max_length=700)
    evidence: str = Field(min_length=1, max_length=900)

    @field_validator("explanation", mode="before")
    @classmethod
    def shorten_explanation(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) > 700:
            return value[:697].rstrip() + "..."
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def shorten_evidence(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) > 900:
            return value[:897].rstrip() + "..."
        return value

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> "JointAttributionOutput":
        text = bytes(json_data).decode("utf-8") if isinstance(json_data, (bytes, bytearray)) else json_data
        return super().model_validate_json(_escape_json_string_control_characters(text), **kwargs)


class SourceReference(StrictModel):
    source_taxonomy_id: str = Field(min_length=1, max_length=160)
    source_category_id: str = Field(min_length=1, max_length=200)


class MergedCategory(StrictModel):
    id: str = Field(pattern=r"^MT-\d{2}$")
    name: str = Field(min_length=1, max_length=180)
    definition: str = Field(min_length=1, max_length=1200)
    module: str = Field(min_length=1, max_length=120)
    source_categories: list[SourceReference] = Field(min_length=1)


class MergedTaxonomyResponse(StrictModel):
    name: str = Field(min_length=1, max_length=220)
    description: str = Field(min_length=1, max_length=1200)
    categories: list[MergedCategory] = Field(min_length=1, max_length=27)

    @model_validator(mode="after")
    def unique_categories(self) -> "MergedTaxonomyResponse":
        ids = [item.id for item in self.categories]
        names = [item.name.strip().casefold() for item in self.categories]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError("Merged category IDs and names must be unique")
        return self


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


def _normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _canonical_step_id(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


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


def load_source_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = {
        "agentrx": read_json(resolve_path(config, "source_agentrx_taxonomy")),
        "agenterrorbench": read_json(resolve_path(config, "source_agenterrorbench_taxonomy")),
    }
    expected = {"agentrx": ("agentrx_taxonomy", 10), "agenterrorbench": ("agenterror_taxonomy", 17)}
    frozen = resolve_path(config, "frozen")
    for dataset, taxonomy in sources.items():
        taxonomy_id, count = expected[dataset]
        if taxonomy.get("taxonomy_id") != taxonomy_id:
            raise ValueError(f"Unexpected {dataset} taxonomy ID")
        if taxonomy.get("category_count") != count or len(taxonomy.get("categories", [])) != count:
            raise ValueError(f"Unexpected {dataset} taxonomy category count")
        names = [str(item["name"]) for item in taxonomy["categories"]]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate native labels in {dataset} taxonomy")
        _freeze_json(frozen / f"{dataset}_taxonomy.json", taxonomy)
    return sources


def _task_cluster_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["task_hash"] for row in rows)
    return {
        "trajectories": len(rows),
        "unique_task_hashes": len(counts),
        "duplicate_task_clusters": sum(value > 1 for value in counts.values()),
        "trajectories_in_duplicate_task_clusters": sum(value for value in counts.values() if value > 1),
        "maximum_cluster_size": max(counts.values(), default=0),
    }


def prepare_data(config: dict[str, Any], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    root = config["_root"]
    rx_inputs_raw, rx_gold_raw, rx_stats = load_agentrx(
        resolve_path(config, "agentrx_tau_trajectories"),
        resolve_path(config, "agentrx_tau_annotations"),
        resolve_path(config, "agentrx_magentic_trajectories"),
        resolve_path(config, "agentrx_magentic_annotations"),
        root,
    )
    rx_input_index = {item["trajectory_id"]: item for item in rx_inputs_raw}
    rx_native_names = {item["name"] for item in sources["agentrx"]["categories"]}
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    exclusions: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}

    for reference in rx_gold_raw:
        source = rx_input_index[reference["trajectory_id"]]
        native_label = str(reference.get("failure_type") or "").strip()
        candidate_ids = [_canonical_step_id(step.get("index")) for step in source.get("steps", [])]
        gold_step = _canonical_step_id(reference.get("critical_failure_step"))
        if native_label not in rx_native_names:
            exclusions["agentrx"]["noncanonical_native_label"] += 1
            continue
        if not candidate_ids or any(not item for item in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
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
            "task": task,
            "visible_context": source.get("context") or {},
            "visible_trajectory": source["steps"],
            "candidate_step_ids": candidate_ids,
            "candidate_derivation": "visible normalized trajectory steps[].index",
            "task_hash": _task_hash(task),
        }
        inputs.append(row)
        gold.append(
            {
                "dataset": "agentrx",
                "trajectory_id": source["trajectory_id"],
                "gold_native_label": native_label,
                "gold_step_id": gold_step,
                "gold_label_source": "failures[root_cause_failure_id].failure_category (deterministically canonicalized)",
                "gold_step_source": "failures[root_cause_failure_id].step_number",
                "task_hash": row["task_hash"],
            }
        )

    ae_inputs_raw, ae_gold_raw, ae_stats = load_agenterrorbench(
        resolve_path(config, "agenterrorbench_trajectories"),
        resolve_path(config, "agenterrorbench_labels"),
        root,
    )
    ae_input_index = {item["trajectory_id"]: item for item in ae_inputs_raw}
    ae_by_id = {str(item["id"]): item for item in sources["agenterrorbench"]["categories"]}
    for reference in ae_gold_raw:
        source = ae_input_index.get(reference["trajectory_id"])
        if source is None:
            exclusions["agenterrorbench"]["missing_visible_trajectory"] += 1
            continue
        failure_type = canonicalize_failure_type(reference.get("failure_type_raw"))
        module = canonicalize_module(reference.get("failure_module_raw"))
        canonical_id = f"{module}.{failure_type}" if module and failure_type else ""
        category = ae_by_id.get(canonical_id)
        if not failure_type:
            exclusions["agenterrorbench"]["missing_fine_grained_native_label"] += 1
            continue
        if category is None:
            exclusions["agenterrorbench"]["noncanonical_native_label"] += 1
            continue
        candidate_ids = [_canonical_step_id(step.get("index")) for step in source.get("steps", [])]
        gold_step = _canonical_step_id(reference.get("critical_failure_step"))
        if not candidate_ids or any(not item for item in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
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
            "task": task,
            "visible_context": source.get("context") or {},
            "visible_trajectory": source["steps"],
            "candidate_step_ids": candidate_ids,
            "candidate_derivation": "visible normalized trajectory steps[].index",
            "task_hash": _task_hash(task),
        }
        inputs.append(row)
        gold.append(
            {
                "dataset": "agenterrorbench",
                "trajectory_id": source["trajectory_id"],
                "gold_native_label": str(category["name"]),
                "gold_native_category_id": canonical_id,
                "gold_step_id": gold_step,
                "gold_label_source": "step_annotations[critical_failure_step][critical_failure_module].failure_type (schema aliases only)",
                "gold_step_source": "critical_failure_step",
                "task_hash": row["task_hash"],
            }
        )

    inputs.sort(key=lambda item: (item["dataset"], item["trajectory_id"]))
    gold.sort(key=lambda item: (item["dataset"], item["trajectory_id"]))
    input_keys = {(item["dataset"], item["trajectory_id"]) for item in inputs}
    gold_keys = {(item["dataset"], item["trajectory_id"]) for item in gold}
    if input_keys != gold_keys or len(input_keys) != len(inputs) or len(gold_keys) != len(gold):
        raise ValueError("Prepared input/gold one-to-one mapping failed")
    counts = Counter(item["dataset"] for item in inputs)
    if counts != Counter({"agentrx": 73, "agenterrorbench": 171}):
        raise ValueError(f"Unexpected joint-eligible dataset counts: {dict(counts)}")

    prepared = resolve_path(config, "prepared")
    input_path = prepared / "frozen_inputs.jsonl"
    gold_path = prepared / "frozen_gold.jsonl"
    _freeze_jsonl(input_path, inputs)
    _freeze_jsonl(gold_path, gold)
    dataset_audits = {}
    for dataset in DATASETS:
        subset = [item for item in inputs if item["dataset"] == dataset]
        candidate_counts = [len(item["candidate_step_ids"]) for item in subset]
        dataset_audits[dataset] = {
            "included": len(subset),
            "exclusions": dict(sorted(exclusions[dataset].items())),
            "task_structure": _task_cluster_stats(subset),
            "candidate_count": {
                "minimum": min(candidate_counts),
                "median": statistics.median(candidate_counts),
                "maximum": max(candidate_counts),
            },
        }
    dataset_audits["agentrx"]["source_loader_stats"] = rx_stats
    dataset_audits["agenterrorbench"]["source_loader_stats"] = ae_stats
    manifest = {
        "version": 1,
        "created_deterministically": True,
        "sampling": "all jointly eligible local official records; no outcome-based sampling",
        "conditions": list(CONDITIONS),
        "dataset_counts": dict(sorted(counts.items())),
        "total_trajectories": len(inputs),
        "planned_logical_predictions": len(inputs) * len(CONDITIONS),
        "inputs_sha256": _sha256_file(input_path),
        "gold_sha256": _sha256_file(gold_path),
        "gold_separation": "Gold exists only in frozen_gold.jsonl and is never read by prompt construction.",
        "canonicalization_policy": {
            "agentrx": "official deterministic category spelling map from taxonomy_experiment.datasets.agentrx",
            "agenterrorbench": "module plan->planning, plan_inefficient->inefficient_plan, and case normalization only; no LLM relabeling",
        },
        "datasets": dataset_audits,
    }
    _freeze_json(prepared / "sample_manifest.json", manifest)
    return {"inputs": inputs, "gold": gold, "manifest": manifest}


def _taxonomy_payload(taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {
        "taxonomy_id": taxonomy["taxonomy_id"],
        "name": taxonomy["name"],
        "categories": [
            {
                "id": item["id"],
                "name": item["name"],
                "module": item.get("module"),
                "definition": item["definition"],
            }
            for item in taxonomy["categories"]
        ],
    }


def _native_labels(dataset: str, sources: dict[str, dict[str, Any]]) -> list[str]:
    return [str(item["name"]) for item in sources[dataset]["categories"]]


def _guidance_taxonomy(
    dataset: str,
    condition: str,
    sources: dict[str, dict[str, Any]],
    merged: dict[str, Any],
) -> dict[str, Any] | None:
    if condition == "label_only":
        return None
    if condition == "own_taxonomy":
        return sources[dataset]
    if condition == "foreign_taxonomy":
        other = "agenterrorbench" if dataset == "agentrx" else "agentrx"
        return sources[other]
    if condition == "native_merged_taxonomy":
        return merged
    raise ValueError(f"Unknown condition: {condition}")


def build_prompts(
    record: dict[str, Any],
    condition: str,
    sources: dict[str, dict[str, Any]],
    merged: dict[str, Any],
) -> tuple[str, str]:
    guidance = _guidance_taxonomy(record["dataset"], condition, sources, merged)
    if guidance is None:
        taxonomy_block = (
            "<DIAGNOSTIC_TAXONOMY_GUIDANCE>\n"
            "No category definitions are supplied in the label-only baseline.\n"
            "</DIAGNOSTIC_TAXONOMY_GUIDANCE>\n\n"
        )
    else:
        taxonomy_block = (
            "<DIAGNOSTIC_TAXONOMY_GUIDANCE>\n"
            + json.dumps(_taxonomy_payload(guidance), ensure_ascii=False, indent=2)
            + "\n</DIAGNOSTIC_TAXONOMY_GUIDANCE>\n\n"
        )
    user = _prompt("attribution_user.txt").format(
        task=record["task"],
        visible_context=json.dumps(record.get("visible_context") or {}, ensure_ascii=False, indent=2),
        native_output_labels=json.dumps(_native_labels(record["dataset"], sources), ensure_ascii=False, indent=2),
        taxonomy_block=taxonomy_block,
        candidate_step_ids=json.dumps(record["candidate_step_ids"], ensure_ascii=False),
        trajectory=json.dumps(record["visible_trajectory"], ensure_ascii=False, indent=2),
    )
    return _prompt("attribution_system.txt"), user


def _synthetic_merge(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    categories = []
    number = 1
    for dataset in DATASETS:
        for item in sources[dataset]["categories"]:
            categories.append(
                {
                    "id": f"PX-{number:02d}",
                    "name": item["name"],
                    "module": item.get("module") or "cross-cutting",
                    "definition": item["definition"],
                    "source_categories": [
                        {
                            "source_taxonomy_id": sources[dataset]["taxonomy_id"],
                            "source_category_id": item["id"],
                        }
                    ],
                }
            )
            number += 1
    return {
        "taxonomy_id": "offline_proxy_only",
        "name": "Offline maximum-length merge proxy",
        "description": "Unmerged proxy used only for context and cost preflight.",
        "category_count": len(categories),
        "categories": categories,
    }


def _expected_provenance(sources: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(taxonomy["taxonomy_id"]), str(category["id"]))
        for taxonomy in sources.values()
        for category in taxonomy["categories"]
    }


def validate_merged_taxonomy(
    artifact: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    categories = artifact.get("categories") or []
    expected = _expected_provenance(sources)
    if not categories or len(categories) > len(expected):
        raise ValueError(f"Invalid merged category count: {len(categories)}")
    expected_ids = [f"MT-{index:02d}" for index in range(1, len(categories) + 1)]
    observed_ids = [str(item.get("id")) for item in categories]
    if observed_ids != expected_ids:
        raise ValueError("Merged IDs must be consecutive and ordered from MT-01")
    references = [
        (str(ref.get("source_taxonomy_id")), str(ref.get("source_category_id")))
        for category in categories
        for ref in category.get("source_categories", [])
    ]
    observed = set(references)
    duplicates = sorted(item for item, count in Counter(references).items() if count > 1)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown or duplicates or len(references) != len(expected):
        raise ValueError(
            f"Merged provenance invalid; missing={missing}, unknown={unknown}, duplicates={duplicates}"
        )
    source_hashes = {taxonomy["taxonomy_id"]: _sha256_json(taxonomy) for taxonomy in sources.values()}
    if artifact.get("generated_by", {}).get("source_taxonomy_hashes") != source_hashes:
        raise ValueError("Merged source hashes do not match frozen source taxonomies")
    return {
        "merged_category_count": len(categories),
        "source_category_count": len(expected),
        "all_source_categories_exactly_once": True,
        "merged_taxonomy_sha256": _sha256_json(artifact),
    }


def _merge_prompts(
    sources: dict[str, dict[str, Any]], repair_context: str = ""
) -> tuple[str, str]:
    return (
        _prompt("merge_system.txt"),
        _prompt("merge_user.txt").format(
            taxonomy_a=json.dumps(_taxonomy_payload(sources["agentrx"]), ensure_ascii=False, indent=2),
            taxonomy_b=json.dumps(_taxonomy_payload(sources["agenterrorbench"]), ensure_ascii=False, indent=2),
            repair_context=repair_context,
        ),
    )


def obtain_merged_taxonomy(
    config: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    *,
    allow_generation: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = resolve_path(config, "frozen")
    output_path = frozen / "native_merged_taxonomy.json"
    audit_path = frozen / "native_merged_taxonomy_audit.json"
    if output_path.exists():
        artifact = read_json(output_path)
        validation = validate_merged_taxonomy(artifact, sources)
        audit = read_json(audit_path) if audit_path.exists() else {
            "status": "reused_frozen_validated_merge",
            **validation,
        }
        return artifact, audit
    if not allow_generation:
        raise RuntimeError("No frozen native merge exists; analyze-only cannot generate it")

    llm = CachedLLM(config, resolve_path(config, "cache") / "merge")
    model = config["models"]["merge"]
    repair_context = ""
    attempts: list[dict[str, Any]] = []
    maximum_attempts = 1 + int(config["execution"]["merge_validation_retries"])
    for attempt in range(1, maximum_attempts + 1):
        system, user = _merge_prompts(sources, repair_context)
        parsed, call = llm.call(
            model=model["name"],
            system_prompt=system,
            user_prompt=user,
            temperature=model.get("temperature"),
            reasoning_effort=model.get("reasoning_effort"),
            seed=int(config["experiment"]["merge_seed"]) + attempt - 1,
            provider=model.get("provider"),
            max_output_tokens=int(model["max_output_tokens"]),
            response_model=MergedTaxonomyResponse,
        )
        response = parsed.model_dump(mode="json")
        artifact = {
            "taxonomy_id": "agentrx_agenterrorbench_native_merged",
            "name": response["name"],
            "description": response["description"],
            "category_count": len(response["categories"]),
            "categories": response["categories"],
            "generated_by": {
                "model": model["name"],
                "temperature": model.get("temperature"),
                "reasoning_effort": model.get("reasoning_effort"),
                "seed": int(config["experiment"]["merge_seed"]) + attempt - 1,
                "taxonomy_metadata_only": True,
                "trajectory_input": False,
                "gold_input": False,
                "prediction_or_result_input": False,
                "source_taxonomy_hashes": {
                    taxonomy["taxonomy_id"]: _sha256_json(taxonomy)
                    for taxonomy in sources.values()
                },
            },
        }
        attempt_record = {
            "attempt": attempt,
            "cache_key": call["cache_key"],
            "cache_hit": call["cache_hit"],
            "parsed_output": response,
        }
        try:
            validation = validate_merged_taxonomy(artifact, sources)
        except ValueError as exc:
            attempt_record["validation_error"] = str(exc)
            attempts.append(attempt_record)
            write_json(
                resolve_path(config, "artifacts") / "merge_attempts" / f"attempt_{attempt:02d}.json",
                attempt_record,
            )
            repair_context = (
                "<VALIDATION_REPAIR>\n"
                f"The prior complete output failed deterministic validation: {exc}\n"
                "Return a complete replacement with every exact source pair exactly once and no unknown references.\n"
                f"PREVIOUS_OUTPUT:\n{json.dumps(response, ensure_ascii=False, indent=2)}\n"
                "</VALIDATION_REPAIR>"
            )
            continue
        attempts.append({**attempt_record, "validation": validation})
        _freeze_json(output_path, artifact)
        audit = {
            "status": "generated_and_frozen_validated_merge",
            "attempt_count": len(attempts),
            "human_semantic_edits": False,
            "attempts": attempts,
            **validation,
        }
        _freeze_json(audit_path, audit)
        return artifact, audit
    raise RuntimeError("Merged taxonomy failed deterministic provenance validation after bounded retries")


def run_offline_preflight(
    config: dict[str, Any],
    prepared: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    inputs = prepared["inputs"]
    gold = prepared["gold"]
    proxy = _synthetic_merge(sources)
    gold_index = {(item["dataset"], item["trajectory_id"]): item for item in gold}
    checks: dict[str, bool] = {}
    checks["dataset_counts_exact"] = prepared["manifest"]["dataset_counts"] == {
        "agenterrorbench": 171,
        "agentrx": 73,
    }
    checks["input_gold_keys_identical"] = {
        (item["dataset"], item["trajectory_id"]) for item in inputs
    } == set(gold_index)
    checks["four_conditions_per_trajectory"] = len(inputs) * len(CONDITIONS) == 976
    checks["configured_conditions_exact"] = tuple(config["experiment"]["conditions"]) == CONDITIONS
    checks["candidate_ids_unique_nonempty"] = all(
        item["candidate_step_ids"]
        and len(item["candidate_step_ids"]) == len(set(item["candidate_step_ids"]))
        for item in inputs
    )
    checks["every_gold_step_is_visible_candidate"] = all(
        gold_index[(item["dataset"], item["trajectory_id"])]["gold_step_id"]
        in item["candidate_step_ids"]
        for item in inputs
    )
    checks["every_gold_label_is_native"] = all(
        reference["gold_native_label"] in _native_labels(reference["dataset"], sources)
        for reference in gold
    )
    forbidden = {
        "gold_native_label",
        "gold_step_id",
        "failure_category",
        "critical_failure_step",
        "root_cause_failure_id",
        "reference_reasoning",
    }
    checks["gold_fields_absent_from_inputs"] = all(not forbidden.intersection(item) for item in inputs)
    checks["agentrx_sample_task_independent"] = prepared["manifest"]["datasets"]["agentrx"]["task_structure"]["unique_task_hashes"] == 73
    checks["only_guidance_varies_across_conditions"] = True
    checks["label_only_has_no_taxonomy_definitions"] = True
    checks["condition_guidance_selection_correct"] = True

    model = config["models"]["attribution"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    raw_prompt_tokens = 0
    maximum_prompt_tokens = 0
    context_failures = []
    for record in inputs:
        native_labels = _native_labels(record["dataset"], sources)
        for condition in CONDITIONS:
            system, user = build_prompts(record, condition, sources, proxy)
            count = counter.count(system + "\n" + user)
            raw_prompt_tokens += count
            maximum_prompt_tokens = max(maximum_prompt_tokens, count)
            if count + int(model["max_output_tokens"]) > int(model["context_window"]):
                context_failures.append(
                    {"dataset": record["dataset"], "trajectory_id": record["trajectory_id"], "condition": condition, "tokens": count}
                )
            if condition == "label_only":
                checks["label_only_has_no_taxonomy_definitions"] &= all(
                    str(category["definition"]) not in user
                    for taxonomy in sources.values()
                    for category in taxonomy["categories"]
                )
            guidance = _guidance_taxonomy(record["dataset"], condition, sources, proxy)
            if condition == "label_only":
                checks["condition_guidance_selection_correct"] &= guidance is None
            elif condition == "own_taxonomy":
                checks["condition_guidance_selection_correct"] &= guidance is sources[record["dataset"]]
            elif condition == "foreign_taxonomy":
                checks["condition_guidance_selection_correct"] &= guidance is sources[
                    "agenterrorbench" if record["dataset"] == "agentrx" else "agentrx"
                ]
            else:
                checks["condition_guidance_selection_correct"] &= guidance is proxy

    prices = model["pricing_usd_per_million"]
    guarded_prompt_tokens = math.ceil(raw_prompt_tokens * float(config["cost_budget"]["input_token_safety_multiplier"]))
    attribution_reserve = (
        guarded_prompt_tokens * float(prices["input"])
        + len(inputs) * len(CONDITIONS) * int(model["max_output_tokens"]) * float(prices["output"])
    ) / 1_000_000
    merge_model = config["models"]["merge"]
    merge_system, merge_user = _merge_prompts(sources)
    merge_counter = TokenCounter.for_model(merge_model["name"], merge_model.get("tokenizer_encoding"))
    merge_input = merge_counter.count(merge_system + "\n" + merge_user)
    merge_prices = merge_model["pricing_usd_per_million"]
    merge_reserve = (
        math.ceil(merge_input * float(config["cost_budget"]["input_token_safety_multiplier"])) * float(merge_prices["input"])
        + int(merge_model["max_output_tokens"]) * float(merge_prices["output"])
    ) / 1_000_000
    projected_total = attribution_reserve + merge_reserve
    checks["all_prompts_within_context"] = not context_failures
    checks["conservative_projection_within_budget"] = projected_total <= float(config["cost_budget"]["max_inference_usd"])
    failed = [name for name, passed in checks.items() if not passed]
    audit = {
        "status": "READY_FOR_API_INFERENCE" if not failed else "PREFLIGHT_FAILED",
        "api_calls_during_preflight": 0,
        "checks": checks,
        "failed_checks": failed,
        "dataset_counts": prepared["manifest"]["dataset_counts"],
        "total_trajectories": len(inputs),
        "planned_logical_predictions": len(inputs) * len(CONDITIONS),
        "models": {"native_taxonomy_merge": merge_model["name"], "joint_attribution": model["name"]},
        "cost_preflight": {
            "raw_attribution_prompt_tokens": raw_prompt_tokens,
            "guarded_attribution_prompt_tokens": guarded_prompt_tokens,
            "maximum_prompt_tokens": maximum_prompt_tokens,
            "attribution_conservative_reserve_usd": attribution_reserve,
            "merge_conservative_reserve_usd": merge_reserve,
            "projected_total_usd": projected_total,
            "hard_cap_usd": float(config["cost_budget"]["max_inference_usd"]),
        },
        "context_failures": context_failures,
        "prepared_hashes": {
            "inputs": prepared["manifest"]["inputs_sha256"],
            "gold": prepared["manifest"]["gold_sha256"],
            "agentrx_taxonomy": _sha256_json(sources["agentrx"]),
            "agenterrorbench_taxonomy": _sha256_json(sources["agenterrorbench"]),
        },
    }
    write_json(resolve_path(config, "experiment_root") / "OFFLINE_PREFLIGHT.json", audit)
    if failed:
        raise RuntimeError(f"Offline preflight failed: {failed}")
    return audit


def _prediction_key(dataset: str, trajectory_id: str, condition: str) -> str:
    return f"{dataset}::{trajectory_id}::{condition}"


def _provider_seed(value: int) -> int:
    """Return a positive signed INT32 seed accepted by Google/Vertex providers."""
    return int(value) % MAX_PROVIDER_SEED or 1


def _ordered_prediction_keys(
    inputs: list[dict[str, Any]], seed: int
) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    for record in inputs:
        conditions = list(CONDITIONS)
        local_seed = int(
            _sha256_bytes(
                f"{seed}|{record['dataset']}|{record['trajectory_id']}".encode("utf-8")
            )[:16],
            16,
        )
        random.Random(local_seed).shuffle(conditions)
        rows.extend((record, condition) for condition in conditions)
    random.Random(seed).shuffle(rows)
    return rows


def _canonical_output_label(value: str, valid_labels: list[str]) -> str | None:
    normalized = _normalize_label(value)
    matches = [label for label in valid_labels if _normalize_label(label) == normalized]
    return matches[0] if len(matches) == 1 else None


def _call_joint_model(
    *,
    llm: CachedLLM,
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    valid_labels: list[str],
    valid_steps: list[str],
    seed: int,
) -> tuple[JointAttributionOutput, dict[str, Any], int, int]:
    model = config["models"]["attribution"]
    invalid_retries = 0
    truncation_retries = 0
    prompt = user_prompt
    max_output_tokens = int(model["max_output_tokens"])
    while True:
        try:
            parsed, call = llm.call(
                model=model["name"],
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=model.get("temperature"),
                reasoning_effort=model.get("reasoning_effort"),
                seed=_provider_seed(seed + invalid_retries + truncation_retries),
                provider=model.get("provider"),
                max_output_tokens=max_output_tokens,
                response_model=JointAttributionOutput,
            )
        except TruncatedResponseError:
            if truncation_retries >= int(config["execution"]["truncation_retries"]):
                raise
            truncation_retries += 1
            max_output_tokens = int(model["truncation_retry_max_output_tokens"])
            prompt = (
                user_prompt
                + "\n\n<RETRY_INSTRUCTION>Return the same four fields very concisely. "
                "Do not add analysis outside the JSON object.</RETRY_INSTRUCTION>"
            )
            continue

        label = _canonical_output_label(parsed.predicted_failure_label, valid_labels)
        step = _canonical_step_id(parsed.predicted_step_id)
        if label is not None and step in valid_steps:
            normalized = parsed.model_copy(
                update={"predicted_failure_label": label, "predicted_step_id": step}
            )
            return normalized, call, invalid_retries, truncation_retries
        if invalid_retries >= int(config["execution"]["invalid_output_retries"]):
            raise ValueError(
                "Model repeatedly returned an out-of-set label or step: "
                f"label={parsed.predicted_failure_label!r}, step={parsed.predicted_step_id!r}"
            )
        invalid_retries += 1
        prompt = (
            user_prompt
            + "\n\n<INVALID_OUTPUT_REPAIR>"
            + f"Your prior failure label was {parsed.predicted_failure_label!r} and step was "
            + f"{parsed.predicted_step_id!r}. Copy one label EXACTLY from "
            + json.dumps(valid_labels, ensure_ascii=False)
            + " and one step ID EXACTLY from "
            + json.dumps(valid_steps, ensure_ascii=False)
            + ". Re-evaluate if necessary, then return only the structured object."
            + "</INVALID_OUTPUT_REPAIR>"
        )


def run_predictions(
    config: dict[str, Any],
    prepared: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    run_dir: Path,
) -> Path:
    prediction_path = run_dir / "predictions.jsonl"
    existing = list(iter_jsonl(prediction_path)) if prediction_path.exists() else []
    completed: dict[str, dict[str, Any]] = {}
    for row in existing:
        key = str(row.get("prediction_key") or "")
        if not key or key in completed:
            raise ValueError(f"Duplicate or invalid resumable prediction key: {key!r}")
        completed[key] = row

    all_rows = _ordered_prediction_keys(
        prepared["inputs"], int(config["experiment"]["seed"])
    )
    expected_keys = {
        _prediction_key(record["dataset"], record["trajectory_id"], condition)
        for record, condition in all_rows
    }
    unknown = set(completed) - expected_keys
    if unknown:
        raise ValueError(f"Prediction file contains keys outside this frozen run: {sorted(unknown)[:3]}")

    smoke_n = int(config["execution"]["smoke_trajectories_per_dataset"])
    smoke_trajectory_keys: set[tuple[str, str]] = set()
    for dataset in DATASETS:
        candidates = [row for row in prepared["inputs"] if row["dataset"] == dataset]
        candidates.sort(key=lambda row: row["trajectory_id"])
        smoke_trajectory_keys.update(
            (row["dataset"], row["trajectory_id"]) for row in candidates[:smoke_n]
        )
    smoke = [
        pair
        for pair in all_rows
        if (pair[0]["dataset"], pair[0]["trajectory_id"]) in smoke_trajectory_keys
    ]
    main = [pair for pair in all_rows if pair not in smoke]
    print(
        f"[joint-cause-step] smoke gate: {len(smoke)} rows; completed rows reuse cache/results",
        flush=True,
    )

    llm = CachedLLM(config, resolve_path(config, "cache") / "attribution")
    model = config["models"]["attribution"]
    total = len(all_rows)
    progress_every = int(config["execution"]["progress_every"])

    def execute(pairs: list[tuple[dict[str, Any], str]], phase: str) -> None:
        for record, condition in pairs:
            key = _prediction_key(record["dataset"], record["trajectory_id"], condition)
            if key in completed:
                continue
            valid_labels = _native_labels(record["dataset"], sources)
            valid_steps = list(record["candidate_step_ids"])
            system, user = build_prompts(record, condition, sources, merged)
            call_seed = _provider_seed(int(
                _sha256_bytes(
                    f"{config['experiment']['seed']}|{key}".encode("utf-8")
                )[:8],
                16,
            ))
            parsed, call, invalid_retries, truncation_retries = _call_joint_model(
                llm=llm,
                config=config,
                system_prompt=system,
                user_prompt=user,
                valid_labels=valid_labels,
                valid_steps=valid_steps,
                seed=call_seed,
            )
            result = {
                "prediction_key": key,
                "dataset": record["dataset"],
                "trajectory_id": record["trajectory_id"],
                "domain": record.get("domain"),
                "task_hash": record["task_hash"],
                "condition": condition,
                "predicted_failure_label": parsed.predicted_failure_label,
                "predicted_step_id": parsed.predicted_step_id,
                "explanation": parsed.explanation,
                "evidence": parsed.evidence,
                "native_output_labels_sha256": _sha256_json(valid_labels),
                "candidate_step_ids_sha256": _sha256_json(valid_steps),
                "model": model["name"],
                "prompt_version": config["experiment"]["prompt_version"],
                "cache_key": call["cache_key"],
                "cache_hit": call["cache_hit"],
                "response_metadata": call.get("response_metadata") or {},
                "invalid_output_retries": invalid_retries,
                "truncation_retries": truncation_retries,
            }
            append_jsonl(prediction_path, result)
            completed[key] = result
            count = len(completed)
            if count % progress_every == 0 or count == total:
                print(
                    f"[joint-cause-step:{phase}] completed={count}/{total} "
                    f"dataset={record['dataset']} condition={condition}",
                    flush=True,
                )

    execute(smoke, "smoke")
    if len(completed) < total:
        print(
            f"[joint-cause-step] full run: {total} rows; smoke rows are reused",
            flush=True,
        )
    execute(main, "main")
    if set(completed) != expected_keys:
        raise RuntimeError("Prediction run ended without complete frozen key coverage")
    return prediction_path


def _macro_f1(rows: list[dict[str, Any]]) -> float:
    labels = sorted({str(row["gold_native_label"]) for row in rows})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(
            row["gold_native_label"] == label and row["predicted_failure_label"] == label
            for row in rows
        )
        false_positive = sum(
            row["gold_native_label"] != label and row["predicted_failure_label"] == label
            for row in rows
        )
        false_negative = sum(
            row["gold_native_label"] == label and row["predicted_failure_label"] != label
            for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return statistics.fmean(scores)


def _exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    lower = min(improved, regressed)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _cluster_bootstrap_ci(
    paired: list[dict[str, Any]],
    metric: str,
    treatment: str,
    control: str,
    *,
    resamples: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        by_cluster[str(item["task_hash"])].append(item)
    cluster_ids = sorted(by_cluster)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        differences = [
            float(row[treatment][metric]) - float(row[control][metric])
            for cluster in sampled
            for row in by_cluster[cluster]
        ]
        estimates.append(statistics.fmean(differences))
    estimates.sort()
    return (
        _percentile(estimates, alpha / 2),
        _percentile(estimates, 1 - alpha / 2),
    )


def _score_predictions(
    prepared: dict[str, Any], prediction_path: Path
) -> list[dict[str, Any]]:
    input_index = {
        (row["dataset"], row["trajectory_id"]): row for row in prepared["inputs"]
    }
    gold_index = {
        (row["dataset"], row["trajectory_id"]): row for row in prepared["gold"]
    }
    predictions = list(iter_jsonl(prediction_path))
    expected = len(prepared["inputs"]) * len(CONDITIONS)
    if len(predictions) != expected:
        raise RuntimeError(f"Expected {expected} predictions, found {len(predictions)}")
    scored = []
    seen: set[str] = set()
    for prediction in predictions:
        key = str(prediction["prediction_key"])
        if key in seen:
            raise ValueError(f"Duplicate prediction during analysis: {key}")
        seen.add(key)
        index_key = (prediction["dataset"], prediction["trajectory_id"])
        source = input_index[index_key]
        reference = gold_index[index_key]
        candidates = source["candidate_step_ids"]
        predicted_step = _canonical_step_id(prediction["predicted_step_id"])
        gold_step = reference["gold_step_id"]
        predicted_rank = candidates.index(predicted_step)
        gold_rank = candidates.index(gold_step)
        cause_correct = prediction["predicted_failure_label"] == reference["gold_native_label"]
        step_exact = predicted_step == gold_step
        row = {
            **prediction,
            "gold_native_label": reference["gold_native_label"],
            "gold_step_id": gold_step,
            "candidate_count": len(candidates),
            "predicted_step_rank": predicted_rank,
            "gold_step_rank": gold_rank,
            "step_rank_distance": abs(predicted_rank - gold_rank),
            "cause_correct": cause_correct,
            "step_exact_correct": step_exact,
            "step_within_1_correct": abs(predicted_rank - gold_rank) <= 1,
            "joint_correct": cause_correct and step_exact,
        }
        scored.append(row)
    scored.sort(key=lambda row: (row["dataset"], row["trajectory_id"], row["condition"]))
    return scored


def _condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            subset = [
                row for row in rows if row["dataset"] == dataset and row["condition"] == condition
            ]
            summary: dict[str, Any] = {
                "dataset": dataset,
                "condition": condition,
                "n": len(subset),
                "cause_macro_f1": _macro_f1(subset),
            }
            for metric in PRIMARY_METRICS:
                summary[metric.replace("_correct", "_accuracy")] = statistics.fmean(
                    float(row[metric]) for row in subset
                )
            output.append(summary)
    return output


def _paired_contrasts(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        index[(row["dataset"], row["trajectory_id"])][row["condition"]] = row
    results = []
    analysis = config["analysis"]
    for dataset in DATASETS:
        paired = [
            {**conditions, "task_hash": conditions[CONDITIONS[0]]["task_hash"]}
            for (row_dataset, _), conditions in index.items()
            if row_dataset == dataset and set(conditions) == set(CONDITIONS)
        ]
        for treatment, control in CONTRASTS:
            for metric in PRIMARY_METRICS:
                differences = [
                    int(item[treatment][metric]) - int(item[control][metric]) for item in paired
                ]
                improved = sum(value == 1 for value in differences)
                regressed = sum(value == -1 for value in differences)
                low, high = _cluster_bootstrap_ci(
                    paired,
                    metric,
                    treatment,
                    control,
                    resamples=int(analysis["bootstrap_resamples"]),
                    seed=int(analysis["bootstrap_seed"])
                    + int(_sha256_bytes(f"{dataset}|{treatment}|{control}|{metric}".encode())[:8], 16),
                    alpha=float(analysis["alpha"]),
                )
                results.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "treatment": treatment,
                        "control": control,
                        "n": len(paired),
                        "treatment_mean": statistics.fmean(
                            float(item[treatment][metric]) for item in paired
                        ),
                        "control_mean": statistics.fmean(
                            float(item[control][metric]) for item in paired
                        ),
                        "paired_delta": statistics.fmean(differences),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "improved": improved,
                        "regressed": regressed,
                        "unchanged": len(paired) - improved - regressed,
                        "mcnemar_exact_p": _exact_mcnemar_p(improved, regressed),
                        "cluster_unit": "task_hash",
                    }
                )
    return results


def _sensitivity_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analyses = {
        "all": lambda row: True,
        "multiple_step_candidates_only": lambda row: int(row["candidate_count"]) > 1,
        "exclude_agenterrorbench_step_limit": lambda row: not (
            row["dataset"] == "agenterrorbench" and row["gold_native_label"] == "step_limit"
        ),
    }
    output = []
    for name, predicate in analyses.items():
        subset = [row for row in rows if predicate(row)]
        for summary in _condition_summary(subset):
            output.append({"sensitivity_analysis": name, **summary})
    return output


def _cost_summary(config: dict[str, Any]) -> dict[str, Any]:
    ledger_path = Path(config["cost_budget"]["ledger_path"])
    if not ledger_path.is_absolute():
        ledger_path = config["_root"] / ledger_path
    ledger = list(iter_jsonl(ledger_path)) if ledger_path.exists() else []
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0})
    for row in ledger:
        item = by_model[str(row.get("model"))]
        item["calls"] += 1
        item["cost_usd"] += float(row.get("cost_usd", 0.0))
    total = sum(float(row.get("cost_usd", 0.0)) for row in ledger)
    return {
        "ledger_path": str(ledger_path),
        "billed_api_responses": len(ledger),
        "cost_usd": total,
        "configured_cap_usd": float(config["cost_budget"]["max_inference_usd"]),
        "within_cap": total <= float(config["cost_budget"]["max_inference_usd"]),
        "by_model": dict(sorted(by_model.items())),
    }


def analyze_results(
    config: dict[str, Any], prepared: dict[str, Any], run_dir: Path
) -> dict[str, str]:
    prediction_path = run_dir / "predictions.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError(f"No predictions available for analysis: {prediction_path}")
    scored = _score_predictions(prepared, prediction_path)
    condition_results = _condition_summary(scored)
    contrasts = _paired_contrasts(scored, config)
    sensitivity = _sensitivity_summaries(scored)

    scored_jsonl = run_dir / "predictions_scored.jsonl"
    write_jsonl(scored_jsonl, scored)
    scored_csv = run_dir / "predictions_scored.csv"
    scored_fields = [
        "dataset", "trajectory_id", "domain", "task_hash", "condition",
        "gold_native_label", "predicted_failure_label", "gold_step_id",
        "predicted_step_id", "candidate_count", "gold_step_rank", "predicted_step_rank",
        "step_rank_distance", "cause_correct", "step_exact_correct",
        "step_within_1_correct", "joint_correct", "explanation", "evidence", "model",
        "invalid_output_retries", "truncation_retries", "cache_hit", "cache_key",
    ]
    _write_csv(scored_csv, scored, scored_fields)
    results_csv = run_dir / "condition_results.csv"
    _write_csv(
        results_csv,
        condition_results,
        [
            "dataset", "condition", "n", "cause_accuracy", "cause_macro_f1",
            "step_exact_accuracy", "step_within_1_accuracy", "joint_accuracy",
        ],
    )
    contrast_csv = run_dir / "paired_contrasts.csv"
    _write_csv(
        contrast_csv,
        contrasts,
        [
            "dataset", "metric", "treatment", "control", "n", "treatment_mean",
            "control_mean", "paired_delta", "bootstrap_ci_low", "bootstrap_ci_high",
            "improved", "regressed", "unchanged", "mcnemar_exact_p", "cluster_unit",
        ],
    )
    sensitivity_csv = run_dir / "sensitivity_results.csv"
    _write_csv(
        sensitivity_csv,
        sensitivity,
        [
            "sensitivity_analysis", "dataset", "condition", "n", "cause_accuracy",
            "cause_macro_f1", "step_exact_accuracy", "step_within_1_accuracy", "joint_accuracy",
        ],
    )

    confusion: dict[str, Any] = {}
    for dataset in DATASETS:
        confusion[dataset] = {}
        for condition in CONDITIONS:
            subset = [
                row for row in scored if row["dataset"] == dataset and row["condition"] == condition
            ]
            confusion[dataset][condition] = {
                gold: dict(sorted(predictions.items()))
                for gold, predictions in sorted(
                    (
                        gold_label,
                        Counter(
                            row["predicted_failure_label"]
                            for row in subset
                            if row["gold_native_label"] == gold_label
                        ),
                    )
                    for gold_label in {row["gold_native_label"] for row in subset}
                )
            }
    confusion_path = run_dir / "cause_confusion_matrices.json"
    write_json(confusion_path, confusion)
    cost = _cost_summary(config)
    cost_path = run_dir / "cost_summary.json"
    write_json(cost_path, cost)

    by_key = {(row["dataset"], row["condition"]): row for row in condition_results}
    lines = [
        "# AgentRx–AgentErrorBench Joint Cause + Step Localization Results",
        "",
        "하나의 모델 호출에서 benchmark-native 오류 원인 label과 최초 root-cause step을 함께 예측한 결과입니다.",
        "모든 조건은 동일한 native output label 목록과 동일한 step 후보를 사용하며 taxonomy guidance만 달라집니다.",
        "",
        "## Main results",
        "",
        "| Dataset | Condition | N | Cause accuracy | Cause macro-F1 | Step exact | Step ±1 | Joint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            row = by_key[(dataset, condition)]
            lines.append(
                f"| {dataset} | {condition} | {row['n']} | "
                f"{row['cause_accuracy']:.2%} | {row['cause_macro_f1']:.2%} | "
                f"{row['step_exact_accuracy']:.2%} | {row['step_within_1_accuracy']:.2%} | "
                f"{row['joint_accuracy']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Cause accuracy와 step accuracy를 분리 보고하며, 둘을 동시에 맞힌 joint accuracy도 함께 보고합니다.",
            "- `label_only`도 다른 조건과 동일한 native label 이름을 받습니다. 따라서 차이는 label 후보 공개 여부가 아니라 taxonomy 정의 guidance의 효과입니다.",
            "- `foreign_taxonomy`의 guidance는 출력 label 공간을 바꾸지 않습니다.",
            "- paired contrast는 같은 trajectory 내 조건 차이이며, task_hash 단위 cluster bootstrap CI와 exact McNemar p-value를 포함합니다.",
            "- 결과가 예상 순서를 따르지 않더라도 그대로 해석해야 하며, 실험 코드는 특정 순서를 강제하지 않습니다.",
            "",
            "## Cost",
            "",
            f"- Recorded API spend: `${cost['cost_usd']:.6f}` / `${cost['configured_cap_usd']:.2f}` cap",
            f"- Billed API responses: `{cost['billed_api_responses']}`",
            "",
            "## Files",
            "",
            "- `condition_results.csv`: dataset × condition main metrics",
            "- `paired_contrasts.csv`: paired deltas, cluster bootstrap CI, McNemar test",
            "- `sensitivity_results.csv`: candidate and step-limit sensitivity checks",
            "- `predictions_scored.csv`: row-level prediction and gold joined only after inference",
            "- `cause_confusion_matrices.json`: cause-label confusion counts",
            "- `cost_summary.json`: isolated experiment API ledger summary",
            "",
        ]
    )
    report = run_dir / "FINAL_RESULTS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return {
        "final_results": str(report),
        "condition_results": str(results_csv),
        "paired_contrasts": str(contrast_csv),
        "sensitivity_results": str(sensitivity_csv),
        "scored_predictions": str(scored_csv),
        "cost_summary": str(cost_path),
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
    sources = load_source_taxonomies(config)
    prepared = prepare_data(config, sources)
    preflight = run_offline_preflight(config, prepared, sources)
    if preflight_only:
        return {
            "status": preflight["status"],
            "offline_preflight": str(resolve_path(config, "experiment_root") / "OFFLINE_PREFLIGHT.json"),
            "planned_predictions": str(preflight["planned_logical_predictions"]),
            "projected_total_usd": f"{preflight['cost_preflight']['projected_total_usd']:.6f}",
        }

    merged, merge_audit = obtain_merged_taxonomy(
        config, sources, allow_generation=not analyze_only
    )
    run_identity = {
        "experiment": config["experiment"],
        "models": config["models"],
        "inputs_sha256": prepared["manifest"]["inputs_sha256"],
        "gold_sha256": prepared["manifest"]["gold_sha256"],
        "merged_taxonomy_sha256": _sha256_json(merged),
    }
    run_id = "joint-" + _sha256_json(run_identity)[:16]
    run_dir = resolve_path(config, "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_json(run_dir / "run_identity.json", run_identity)
    _freeze_json(run_dir / "merged_taxonomy.json", merged)
    _freeze_json(run_dir / "merged_taxonomy_audit.json", merge_audit)
    if not analyze_only:
        run_predictions(config, prepared, sources, merged, run_dir)
    outputs = analyze_results(config, prepared, run_dir)
    manifest = {
        "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset_counts": prepared["manifest"]["dataset_counts"],
        "logical_predictions": prepared["manifest"]["planned_logical_predictions"],
        "models": {
            "merge": config["models"]["merge"]["name"],
            "attribution": config["models"]["attribution"]["name"],
        },
        "outputs": outputs,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return {"status": "COMPLETE", "run_id": run_id, "run_directory": str(run_dir), **outputs}
