from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.datasets.agenterrorbench import load_agenterrorbench
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
from taxonomy_experiment.token_count import TokenCounter


HERE = Path(__file__).resolve().parent
DATASETS = ("whowhen", "agenterrorbench")
CONDITIONS = (
    "baseline",
    "own_taxonomy",
    "foreign_taxonomy",
    "native_merged_taxonomy",
)
CONTRASTS = (
    ("own_taxonomy", "baseline"),
    ("foreign_taxonomy", "baseline"),
    ("native_merged_taxonomy", "baseline"),
    ("native_merged_taxonomy", "foreign_taxonomy"),
    ("native_merged_taxonomy", "own_taxonomy"),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _escape_json_string_control_characters(value: str) -> str:
    """Escape raw U+0000..U+001F characters only inside JSON strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    replacements = {
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
    }
    for character in value:
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            continue
        if escaped:
            if ord(character) < 0x20:
                # The preceding backslash is already in output. Complete a
                # valid JSON escape rather than retain the raw control byte.
                output.append(replacements.get(character, f"u{ord(character):04x}"))
            else:
                output.append(character)
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


class StepLocalizationOutput(StrictModel):
    predicted_step_id: str = Field(min_length=1, max_length=80)
    explanation: str = Field(min_length=1, max_length=400)
    evidence: str = Field(min_length=1, max_length=400)

    @field_validator("explanation", "evidence", mode="before")
    @classmethod
    def normalize_supporting_text(cls, value: Any) -> Any:
        # Gemini occasionally ignores maxLength in the response schema. These
        # fields are not scored, so retain the same schema and deterministically
        # shorten only overlong supporting prose before Pydantic enforces it.
        if isinstance(value, str) and len(value) > 400:
            return value[:397].rstrip() + "..."
        return value

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> "StepLocalizationOutput":
        # OpenRouter/Vertex can occasionally emit literal control characters
        # inside a quoted value. Repair only that JSON encoding defect; all
        # fields remain subject to the original strict schema validation.
        if isinstance(json_data, (bytes, bytearray)):
            text = bytes(json_data).decode("utf-8")
        else:
            text = json_data
        return super().model_validate_json(
            _escape_json_string_control_characters(text), **kwargs
        )


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
    categories: list[MergedCategory] = Field(min_length=1, max_length=31)

    @model_validator(mode="after")
    def validate_unique(self) -> "MergedTaxonomyResponse":
        ids = [item.id for item in self.categories]
        names = [item.name.strip().casefold() for item in self.categories]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError("Merged category IDs and names must be unique")
        return self


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _prompt(name: str) -> str:
    return (HERE / "prompts" / name).read_text(encoding="utf-8").strip()


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


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


def _freeze_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(f"Frozen taxonomy differs: {destination}")
        return
    shutil.copyfile(source, destination)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_step_id(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _task_hash(value: str) -> str:
    normalized = " ".join(value.split()).casefold()
    return _sha256_bytes(normalized.encode("utf-8"))


def _visible_whowhen_events(
    trajectory: Any, framework: str
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(trajectory, list):
        raise ValueError("Who&When trajectory is not a list")
    events: list[dict[str, Any]] = []
    candidate_ids: list[str] = []
    visible_raw = [raw for raw in trajectory if isinstance(raw, dict) and str(raw.get("kind") or "").casefold() != "user"]
    smol_start = (
        0
        if framework == "smolagents"
        and visible_raw
        and str(visible_raw[0].get("kind") or "").casefold() == "planning"
        else 1
    )
    for native_position, raw in enumerate(visible_raw, start=smol_start):
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").casefold()
        candidate = ""
        if framework == "smolagents":
            # Who&When's native smolagents step is the ordered non-user event
            # position. Initial planning is step 0; without it, action numbering
            # begins at 1. Later planning events consume a native step as well.
            candidate = str(native_position)
        elif kind == "action" and raw.get("step_number") is not None:
            candidate = _canonical_step_id(raw.get("step_number"))
        elif kind == "agent" and raw.get("round") is not None and raw.get("position") is not None:
            candidate = f"{_canonical_step_id(raw['round'])}.{_canonical_step_id(raw['position'])}"
        if not candidate:
            continue
        event = json.loads(json.dumps(raw, ensure_ascii=False))
        event["candidate_step_id"] = candidate
        events.append(event)
        candidate_ids.append(candidate)
    if not events:
        raise ValueError("Who&When trajectory contains no selectable visible events")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Who&When visible candidate step IDs are not unique")
    return events, candidate_ids


def _load_whowhen(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = load_from_disk(str(resolve_path(config, "whowhen_data")))
    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exclusions = Counter()
    for source in dataset:
        ground_truth = _maybe_json(source.get("ground_truth")) or {}
        if not isinstance(ground_truth, dict):
            exclusions["missing_or_invalid_ground_truth"] += 1
            continue
        if isinstance(ground_truth.get("mode"), list):
            exclusions["multi_mode_gold"] += 1
            continue
        gold_step = _canonical_step_id(ground_truth.get("step"))
        if not gold_step:
            exclusions["missing_gold_step"] += 1
            continue
        try:
            framework = str(source.get("framework") or "unknown")
            events, candidate_ids = _visible_whowhen_events(
                _maybe_json(source.get("trajectory")), framework
            )
        except ValueError:
            exclusions["invalid_visible_trajectory"] += 1
            continue
        if gold_step not in candidate_ids:
            exclusions["gold_step_not_in_visible_candidates"] += 1
            continue
        task = _maybe_json(source.get("task")) or {}
        query = str(task.get("query") if isinstance(task, dict) else task).strip()
        if not query:
            exclusions["missing_task_query"] += 1
            continue
        trajectory_id = str(source.get("id") or "").strip()
        if not trajectory_id:
            exclusions["missing_trajectory_id"] += 1
            continue
        task_hash = _task_hash(query)
        input_row = {
            "dataset": "whowhen",
            "trajectory_id": trajectory_id,
            "framework": framework,
            "benchmark": str(source.get("benchmark") or "unknown"),
            "task": query,
            "visible_trajectory": events,
            "candidate_step_ids": candidate_ids,
            "candidate_derivation": (
                "visible native events: smolagents ordered non-user event position; "
                "alfagent action.step_number; magentic-one agent round.position"
            ),
            "task_hash": task_hash,
        }
        gold_row = {
            "dataset": "whowhen",
            "trajectory_id": trajectory_id,
            "gold_step_id": gold_step,
            "gold_source_field": "ground_truth.step",
            "task_hash": task_hash,
        }
        eligible.append((input_row, gold_row))

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in eligible:
        grouped[pair[0]["task_hash"]].append(pair)
    seed = int(config["experiment"]["seed"])
    representatives = []
    for task_hash, pairs in grouped.items():
        representatives.append(
            min(
                pairs,
                key=lambda pair: _sha256_bytes(
                    f"{seed}:whowhen:representative:{task_hash}:{pair[0]['trajectory_id']}".encode("utf-8")
                ),
            )
        )
    representatives.sort(
        key=lambda pair: _sha256_bytes(
            f"{seed}:whowhen:sample:{pair[0]['task_hash']}".encode("utf-8")
        )
    )
    target = int(config["experiment"]["whowhen_sample_n"])
    if len(representatives) < target:
        raise ValueError(f"Insufficient unique-task Who&When pool: {len(representatives)} < {target}")
    selected = representatives[:target]
    inputs = sorted((pair[0] for pair in selected), key=lambda row: row["trajectory_id"])
    gold = sorted((pair[1] for pair in selected), key=lambda row: row["trajectory_id"])
    audit = {
        "source_rows": len(dataset),
        "valid_step_rows_before_task_deduplication": len(eligible),
        "unique_task_pool": len(representatives),
        "sampled_rows": len(inputs),
        "task_independent_sample": len({row["task_hash"] for row in inputs}) == len(inputs),
        "exclusions": dict(sorted(exclusions.items())),
        "framework_counts": dict(sorted(Counter(row["framework"] for row in inputs).items())),
        "candidate_derivation_uses_gold": False,
    }
    return inputs, gold, audit


def _load_agenterrorbench(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_inputs, raw_gold, stats = load_agenterrorbench(
        resolve_path(config, "agenterrorbench_trajectories"),
        resolve_path(config, "agenterrorbench_labels"),
        config["_root"],
    )
    inputs_by_id = {row["trajectory_id"]: row for row in raw_inputs}
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    exclusions = Counter()
    for official in raw_gold:
        source = inputs_by_id.get(official["trajectory_id"])
        if source is None:
            exclusions["missing_visible_trajectory"] += 1
            continue
        candidate_ids = [_canonical_step_id(step.get("index")) for step in source.get("steps", [])]
        if not candidate_ids or any(not item for item in candidate_ids):
            exclusions["invalid_visible_candidate"] += 1
            continue
        if len(candidate_ids) != len(set(candidate_ids)):
            exclusions["duplicate_visible_candidate"] += 1
            continue
        gold_step = _canonical_step_id(official.get("critical_failure_step"))
        if not gold_step:
            exclusions["missing_gold_step"] += 1
            continue
        if gold_step not in candidate_ids:
            exclusions["gold_step_not_in_visible_candidates"] += 1
            continue
        task = str(source.get("instruction") or "").strip()
        if not task:
            exclusions["missing_task_query"] += 1
            continue
        task_hash = _task_hash(task)
        visible_steps = []
        for step, candidate in zip(source["steps"], candidate_ids, strict=True):
            visible = json.loads(json.dumps(step, ensure_ascii=False))
            visible["candidate_step_id"] = candidate
            visible_steps.append(visible)
        input_row = {
            "dataset": "agenterrorbench",
            "trajectory_id": source["trajectory_id"],
            "framework": "single_agent",
            "benchmark": source.get("domain", "unknown"),
            "task": task,
            "visible_context": source.get("context") or {},
            "visible_trajectory": visible_steps,
            "candidate_step_ids": candidate_ids,
            "candidate_derivation": "normalized visible trajectory steps[].index",
            "task_hash": task_hash,
        }
        gold_row = {
            "dataset": "agenterrorbench",
            "trajectory_id": source["trajectory_id"],
            "gold_step_id": gold_step,
            "gold_source_field": "critical_failure_step",
            "task_hash": task_hash,
        }
        inputs.append(input_row)
        gold.append(gold_row)
    inputs.sort(key=lambda row: row["trajectory_id"])
    gold.sort(key=lambda row: row["trajectory_id"])
    target = int(config["experiment"]["agenterrorbench_sample_n"])
    if len(inputs) != target:
        raise ValueError(f"Expected {target} valid AgentErrorBench step rows, observed {len(inputs)}")
    task_counts = Counter(row["task_hash"] for row in inputs)
    audit = {
        "source_rows": stats["total_trajectories"],
        "sampled_rows": len(inputs),
        "unique_tasks": len(task_counts),
        "duplicate_task_clusters": sum(count > 1 for count in task_counts.values()),
        "trajectories_in_duplicate_task_clusters": sum(count for count in task_counts.values() if count > 1),
        "maximum_task_cluster_size": max(task_counts.values()),
        "exclusions": dict(sorted(exclusions.items())),
        "benchmark_counts": dict(sorted(Counter(row["benchmark"] for row in inputs).items())),
        "candidate_derivation_uses_gold": False,
        "statistical_dependency_policy": "retain all official trajectories; use task-cluster bootstrap for paired contrasts",
    }
    return inputs, gold, audit


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    whowhen_inputs, whowhen_gold, whowhen_audit = _load_whowhen(config)
    ae_inputs, ae_gold, ae_audit = _load_agenterrorbench(config)
    inputs = whowhen_inputs + ae_inputs
    gold = whowhen_gold + ae_gold
    input_keys = {(row["dataset"], row["trajectory_id"]) for row in inputs}
    gold_keys = {(row["dataset"], row["trajectory_id"]) for row in gold}
    if input_keys != gold_keys or len(input_keys) != len(inputs) or len(gold_keys) != len(gold):
        raise ValueError("Prepared input/gold one-to-one mapping failed")
    prepared = resolve_path(config, "prepared")
    _freeze_jsonl(prepared / "frozen_inputs.jsonl", inputs)
    _freeze_jsonl(prepared / "frozen_gold.jsonl", gold)
    manifest = {
        "version": 1,
        "sampling_seed": int(config["experiment"]["seed"]),
        "conditions": list(CONDITIONS),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in inputs).items())),
        "total_trajectories": len(inputs),
        "planned_prediction_rows": len(inputs) * len(CONDITIONS),
        "inputs_sha256": _sha256_file(prepared / "frozen_inputs.jsonl"),
        "gold_sha256": _sha256_file(prepared / "frozen_gold.jsonl"),
        "datasets": {"whowhen": whowhen_audit, "agenterrorbench": ae_audit},
        "candidate_policy": "IDs only, deterministically extracted from visible trajectory fields before gold comparison",
        "gold_separation": "Gold exists only in frozen_gold.jsonl and is never read by prompt construction",
    }
    _freeze_json(prepared / "sample_manifest.json", manifest)
    return {"inputs": inputs, "gold": gold, "manifest": manifest}


def load_source_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = {
        "whowhen": read_json(resolve_path(config, "source_whowhen_taxonomy")),
        "agenterrorbench": read_json(resolve_path(config, "source_agenterrorbench_taxonomy")),
    }
    if sources["whowhen"].get("taxonomy_id") != "whowhen_pro_text_observed_14":
        raise ValueError("Unexpected Who&When taxonomy")
    if sources["agenterrorbench"].get("taxonomy_id") != "agenterror_taxonomy":
        raise ValueError("Unexpected AgentErrorBench taxonomy")
    for dataset, taxonomy in sources.items():
        categories = taxonomy.get("categories") or []
        if not categories or int(taxonomy.get("category_count", -1)) != len(categories):
            raise ValueError(f"Invalid source taxonomy category count: {dataset}")
        for category in categories:
            if not all(str(category.get(key) or "").strip() for key in ("id", "name", "definition", "module")):
                raise ValueError(f"Incomplete source taxonomy category: {dataset}/{category.get('id')}")
    frozen = resolve_path(config, "frozen")
    _freeze_copy(resolve_path(config, "source_whowhen_taxonomy"), frozen / "whowhen_taxonomy.json")
    _freeze_copy(resolve_path(config, "source_agenterrorbench_taxonomy"), frozen / "agenterrorbench_taxonomy.json")
    return sources


def _taxonomy_payload(taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {
        "taxonomy_id": taxonomy["taxonomy_id"],
        "name": taxonomy["name"],
        "categories": [
            {
                "id": item["id"],
                "name": item["name"],
                "module": item["module"],
                "definition": item["definition"],
            }
            for item in taxonomy["categories"]
        ],
    }


def _taxonomy_block(taxonomy: dict[str, Any] | None) -> str:
    if taxonomy is None:
        return ""
    return (
        "<DIAGNOSTIC_TAXONOMY>\n"
        + json.dumps(_taxonomy_payload(taxonomy), ensure_ascii=False, indent=2)
        + "\n</DIAGNOSTIC_TAXONOMY>\n\n"
    )


def taxonomy_for_condition(
    dataset: str,
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if condition == "baseline":
        return None
    if condition == "own_taxonomy":
        return taxonomies[dataset]
    if condition == "foreign_taxonomy":
        return taxonomies["agenterrorbench" if dataset == "whowhen" else "whowhen"]
    if condition == "native_merged_taxonomy":
        return taxonomies["native_merged"]
    raise ValueError(f"Unknown condition: {condition}")


def build_prompt(
    record: dict[str, Any],
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, str]:
    # This function intentionally accepts no gold object.
    selected = taxonomy_for_condition(record["dataset"], condition, taxonomies)
    trajectory_payload: dict[str, Any] = {"events": record["visible_trajectory"]}
    if record.get("visible_context"):
        trajectory_payload["context"] = record["visible_context"]
    values = {
        "task": record["task"],
        "taxonomy_block": _taxonomy_block(selected),
        "candidate_step_ids": json.dumps(record["candidate_step_ids"], ensure_ascii=False),
        "trajectory": json.dumps(trajectory_payload, ensure_ascii=False, indent=2),
    }
    user = _prompt("attribution_user.txt").format(**values)
    base_user = _prompt("attribution_user.txt").format(**{**values, "taxonomy_block": ""})
    return {
        "system_prompt": _prompt("attribution_system.txt"),
        "user_prompt": user,
        "base_user_prompt": base_user,
    }


def _synthetic_merge(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    categories = []
    number = 1
    for dataset in DATASETS:
        for item in sources[dataset]["categories"]:
            categories.append(
                {
                    "id": f"PX-{number:02d}",
                    "name": item["name"],
                    "module": item["module"],
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
        "category_count": len(categories),
        "categories": categories,
    }


def _merge_prompts(
    sources: dict[str, dict[str, Any]], repair_context: str = ""
) -> tuple[str, str]:
    return (
        _prompt("merge_system.txt"),
        _prompt("merge_user.txt").format(
            taxonomy_a=json.dumps(_taxonomy_payload(sources["whowhen"]), ensure_ascii=False, indent=2),
            taxonomy_b=json.dumps(_taxonomy_payload(sources["agenterrorbench"]), ensure_ascii=False, indent=2),
            repair_context=repair_context,
        ),
    )


def _expected_provenance(sources: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (taxonomy["taxonomy_id"], str(category["id"]))
        for taxonomy in sources.values()
        for category in taxonomy["categories"]
    }


def validate_merged_taxonomy(
    artifact: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    categories = artifact.get("categories") or []
    if not categories or len(categories) > len(_expected_provenance(sources)):
        raise ValueError(f"Invalid merged category count: {len(categories)}")
    expected_ids = [f"MT-{index:02d}" for index in range(1, len(categories) + 1)]
    observed_ids = [str(item.get("id")) for item in categories]
    if observed_ids != expected_ids:
        raise ValueError(f"Merged IDs must be consecutive and ordered: {expected_ids}")
    references = [
        (str(ref.get("source_taxonomy_id")), str(ref.get("source_category_id")))
        for category in categories
        for ref in category.get("source_categories", [])
    ]
    expected = _expected_provenance(sources)
    observed = set(references)
    duplicates = sorted(item for item, count in Counter(references).items() if count > 1)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown or duplicates or len(references) != len(expected):
        raise ValueError(
            f"Merged provenance invalid; missing={missing}, unknown={unknown}, duplicates={duplicates}"
        )
    source_hashes = {
        taxonomy["taxonomy_id"]: _sha256_json(taxonomy)
        for taxonomy in sources.values()
    }
    if artifact.get("generated_by", {}).get("source_taxonomy_hashes") != source_hashes:
        raise ValueError("Merged source hashes do not match frozen source taxonomies")
    return {
        "merged_category_count": len(categories),
        "source_category_count": len(expected),
        "all_source_categories_exactly_once": True,
        "merged_taxonomy_sha256": _sha256_json(artifact),
    }


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
            "taxonomy_id": "whowhen_agenterrorbench_native_merged",
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
            write_json(resolve_path(config, "artifacts") / "merge_attempts" / f"attempt_{attempt:02d}.json", attempt_record)
            repair_context = (
                "<VALIDATION_REPAIR>\n"
                f"The preceding complete output failed deterministic validation: {exc}\n"
                "Return a complete replacement. Include every exact source taxonomy/category pair exactly once, "
                "remove unknown or duplicate references, and use consecutive ordered MT-01 style IDs.\n"
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


def _condition_order(seed: int, dataset: str, trajectory_id: str) -> list[str]:
    ordered = list(CONDITIONS)
    digest = hashlib.sha256(f"{seed}:{dataset}:{trajectory_id}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
    return ordered


def planned_keys(inputs: list[dict[str, Any]], seed: int) -> list[tuple[str, str, str]]:
    keys: list[tuple[str, str, str]] = []
    for row in sorted(inputs, key=lambda item: (item["dataset"], item["trajectory_id"])):
        keys.extend(
            (row["dataset"], row["trajectory_id"], condition)
            for condition in _condition_order(seed, row["dataset"], row["trajectory_id"])
        )
    return keys


def _ledger_summary(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["cost_budget"]["ledger_path"])
    if not path.is_absolute():
        path = config["_root"] / path
    rows = list(iter_jsonl(path)) if path.exists() else []
    return {
        "path": str(path),
        "actual_api_calls": len(rows),
        "total_recorded_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in rows),
        "prompt_tokens": sum(int((row.get("reported_usage") or {}).get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int((row.get("reported_usage") or {}).get("completion_tokens") or 0) for row in rows),
    }


def offline_preflight(
    config: dict[str, Any], prepared: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    inputs = prepared["inputs"]
    gold = prepared["gold"]
    expected_counts = {
        "whowhen": int(config["experiment"]["whowhen_sample_n"]),
        "agenterrorbench": int(config["experiment"]["agenterrorbench_sample_n"]),
    }
    counts = Counter(row["dataset"] for row in inputs)
    input_index = {(row["dataset"], row["trajectory_id"]): row for row in inputs}
    gold_index = {(row["dataset"], row["trajectory_id"]): row for row in gold}
    proxy = _synthetic_merge(sources)
    taxonomies = {**sources, "native_merged": proxy}
    keys = planned_keys(inputs, int(config["experiment"]["seed"]))

    checks = {
        "dataset_counts_exact": dict(counts) == expected_counts,
        "input_gold_keys_identical": set(input_index) == set(gold_index),
        "four_conditions_per_trajectory": len(keys) == len(inputs) * 4
        and all(
            {condition for dataset, trajectory_id, condition in keys if (dataset, trajectory_id) == key}
            == set(CONDITIONS)
            for key in input_index
        ),
        "configured_conditions_exact": tuple(config["experiment"]["conditions"]) == CONDITIONS,
        "candidate_ids_unique_nonempty": all(
            row["candidate_step_ids"]
            and len(row["candidate_step_ids"]) == len(set(row["candidate_step_ids"]))
            for row in inputs
        ),
        "every_gold_is_visible_candidate": all(
            gold_index[key]["gold_step_id"] in row["candidate_step_ids"]
            for key, row in input_index.items()
        ),
        "gold_fields_absent_from_inputs": all(
            not any(
                key.startswith("gold")
                or key in {
                    "ground_truth",
                    "failure_type",
                    "failure_category",
                    "faulty_agent",
                    "injection_metadata",
                    "reference_answer",
                    "reference_reasoning",
                    "critical_failure_step",
                }
                for key in row
            )
            for row in inputs
        ),
        "candidate_derivation_declares_visible_only": all(
            "visible" in row["candidate_derivation"] and "gold" not in row["candidate_derivation"]
            for row in inputs
        ),
        "whowhen_sample_task_independent": len(
            {row["task_hash"] for row in inputs if row["dataset"] == "whowhen"}
        )
        == expected_counts["whowhen"],
        "source_taxonomies_frozen": all(
            (resolve_path(config, "frozen") / f"{dataset}_taxonomy.json").exists()
            for dataset in DATASETS
        ),
    }

    baseline_clean = True
    base_invariant = True
    correct_taxonomy_selection = True
    for row in inputs:
        base_hashes = set()
        for condition in CONDITIONS:
            built = build_prompt(row, condition, taxonomies)
            base_hashes.add(_sha256_bytes(built["base_user_prompt"].encode("utf-8")))
            selected = taxonomy_for_condition(row["dataset"], condition, taxonomies)
            if condition == "baseline":
                baseline_clean &= "<DIAGNOSTIC_TAXONOMY>" not in built["user_prompt"]
                baseline_clean &= built["user_prompt"] == built["base_user_prompt"]
            elif condition == "own_taxonomy":
                correct_taxonomy_selection &= selected is sources[row["dataset"]]
            elif condition == "foreign_taxonomy":
                correct_taxonomy_selection &= selected is not sources[row["dataset"]]
            else:
                correct_taxonomy_selection &= selected is proxy
        base_invariant &= len(base_hashes) == 1
    checks["baseline_contains_no_taxonomy"] = baseline_clean
    checks["only_taxonomy_block_varies"] = base_invariant
    checks["condition_taxonomy_selection_correct"] = correct_taxonomy_selection

    model = config["models"]["attribution"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    prompt_tokens: list[int] = []
    context_failures = []
    for row in inputs:
        for condition in CONDITIONS:
            built = build_prompt(row, condition, taxonomies)
            count = counter.count(built["system_prompt"] + "\n" + built["user_prompt"])
            prompt_tokens.append(count)
            if count + int(model["truncation_retry_max_output_tokens"]) >= int(model["context_window"]):
                context_failures.append(
                    {
                        "dataset": row["dataset"],
                        "trajectory_id": row["trajectory_id"],
                        "condition": condition,
                        "prompt_tokens": count,
                    }
                )
    checks["all_prompts_within_context"] = not context_failures

    merge_model = config["models"]["merge"]
    merge_system, merge_user = _merge_prompts(sources)
    merge_tokens = TokenCounter.for_model(
        merge_model["name"], merge_model.get("tokenizer_encoding")
    ).count(merge_system + "\n" + merge_user)
    safety = float(config["cost_budget"]["input_token_safety_multiplier"])
    attr_prices = model["pricing_usd_per_million"]
    merge_prices = merge_model["pricing_usd_per_million"]
    attr_reserve = (
        math.ceil(sum(prompt_tokens) * safety) * float(attr_prices["input"])
        + len(prompt_tokens) * int(model["max_output_tokens"]) * float(attr_prices["output"])
    ) / 1_000_000
    merge_attempts = 1 + int(config["execution"]["merge_validation_retries"])
    merge_reserve = merge_attempts * (
        math.ceil(merge_tokens * safety) * float(merge_prices["input"])
        + int(merge_model["max_output_tokens"]) * float(merge_prices["output"])
    ) / 1_000_000
    projected = attr_reserve + merge_reserve
    checks["conservative_projection_within_budget"] = projected <= float(
        config["cost_budget"]["max_inference_usd"]
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "status": "READY_FOR_API_INFERENCE" if not failed else "STOP_BEFORE_API",
        "api_calls_during_preflight": 0,
        "checks": checks,
        "failed_checks": failed,
        "dataset_counts": dict(sorted(counts.items())),
        "total_trajectories": len(inputs),
        "planned_logical_predictions": len(keys),
        "candidate_policy": prepared["manifest"]["candidate_policy"],
        "candidate_count": {
            "minimum": min(len(row["candidate_step_ids"]) for row in inputs),
            "median": statistics.median(len(row["candidate_step_ids"]) for row in inputs),
            "maximum": max(len(row["candidate_step_ids"]) for row in inputs),
        },
        "task_structure": {
            dataset: {
                "trajectories": counts[dataset],
                "unique_task_hashes": len({row["task_hash"] for row in inputs if row["dataset"] == dataset}),
            }
            for dataset in DATASETS
        },
        "models": {
            "native_taxonomy_merge": merge_model["name"],
            "step_localization": model["name"],
        },
        "cost_preflight": {
            "attribution_proxy_prompt_tokens": sum(prompt_tokens),
            "maximum_prompt_tokens": max(prompt_tokens),
            "attribution_conservative_reserve_usd": attr_reserve,
            "merge_conservative_reserve_usd": merge_reserve,
            "projected_total_usd": projected,
            "hard_cap_usd": float(config["cost_budget"]["max_inference_usd"]),
            "note": "Proxy merge preserves all source definitions separately, so its prompt length is conservative for a semantic merge.",
        },
        "context_failures": context_failures,
        "prepared_hashes": {
            "inputs": _sha256_json(inputs),
            "gold": _sha256_json(gold),
            "manifest": _sha256_json(prepared["manifest"]),
            "whowhen_taxonomy": _sha256_json(sources["whowhen"]),
            "agenterrorbench_taxonomy": _sha256_json(sources["agenterrorbench"]),
        },
    }
    write_json(HERE / "OFFLINE_PREFLIGHT.json", report)
    if failed:
        raise RuntimeError(f"STOP_BEFORE_API: offline preflight failed: {failed}")
    return report


def _run_id(
    config: dict[str, Any],
    prepared: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    merged: dict[str, Any],
) -> str:
    identity = {
        "experiment": config["experiment"],
        "models": config["models"],
        "prompts": {
            path.name: _sha256_file(path)
            for path in sorted((HERE / "prompts").glob("*.txt"))
        },
        "prepared_inputs": _sha256_json(prepared["inputs"]),
        "source_taxonomies": {
            key: _sha256_json(value) for key, value in sources.items()
        },
        "native_merged_taxonomy": _sha256_json(merged),
        "scoring": "exact-step-id-plus-visible-candidate-rank-distance-v1",
    }
    return f"step-main-{_sha256_json(identity)[:16]}"


def _latest_predictions(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return latest
    for row in iter_jsonl(path):
        latest[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    return latest


def _call_step_model(
    config: dict[str, Any],
    llm: CachedLLM,
    record: dict[str, Any],
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
) -> tuple[StepLocalizationOutput, dict[str, Any], int, int]:
    model = config["models"]["attribution"]
    built = build_prompt(record, condition, taxonomies)
    user_prompt = built["user_prompt"]
    invalid_retries = int(config["execution"]["invalid_step_retries"])
    truncation_retries = int(config["execution"]["truncation_retries"])
    invalid_count = 0
    truncation_count = 0
    seed_digest = hashlib.sha256(
        f"{config['experiment']['seed']}:{record['dataset']}:{record['trajectory_id']}:{condition}".encode("utf-8")
    ).digest()
    # Google Vertex validates generation_config.seed as a signed INT32. Keep
    # the trajectory/condition-derived seed deterministic while constraining
    # it to the provider's accepted non-negative range.
    seed = int.from_bytes(seed_digest[:4], "big") % 2_147_483_647
    max_output_tokens = int(model["max_output_tokens"])
    while True:
        try:
            parsed, call = llm.call(
                model=model["name"],
                system_prompt=built["system_prompt"],
                user_prompt=user_prompt,
                temperature=model.get("temperature"),
                reasoning_effort=model.get("reasoning_effort"),
                # Keep retry seeds in Google's accepted signed INT32 range too.
                seed=(seed + invalid_count + truncation_count) % 2_147_483_647,
                provider=model.get("provider"),
                max_output_tokens=max_output_tokens,
                response_model=StepLocalizationOutput,
            )
        except TruncatedResponseError:
            if truncation_count >= truncation_retries:
                raise
            truncation_count += 1
            max_output_tokens = int(model["truncation_retry_max_output_tokens"])
            user_prompt += (
                "\n\n<TRUNCATION_RECOVERY>Return the same three fields without repeating "
                "claims. Keep explanation and evidence to one short sentence each.</TRUNCATION_RECOVERY>"
            )
            continue
        predicted = parsed.predicted_step_id.strip()
        if predicted in record["candidate_step_ids"]:
            parsed.predicted_step_id = predicted
            return parsed, call, invalid_count, truncation_count
        if invalid_count >= invalid_retries:
            raise ValueError(
                f"Model returned invalid step {predicted!r}; expected one of {record['candidate_step_ids']}"
            )
        invalid_count += 1
        user_prompt += (
            "\n\n<INVALID_STEP_CORRECTION>"
            f"The previous predicted_step_id {predicted!r} is invalid. Copy exactly one ID "
            f"from this list: {json.dumps(record['candidate_step_ids'], ensure_ascii=False)}. "
            "Do not change or reformat the ID.</INVALID_STEP_CORRECTION>"
        )


def _smoke_keys(
    inputs: list[dict[str, Any]], all_keys: list[tuple[str, str, str]], count: int
) -> list[tuple[str, str, str]]:
    selected: set[tuple[str, str]] = set()
    for dataset in DATASETS:
        rows = sorted(
            (row for row in inputs if row["dataset"] == dataset),
            key=lambda row: row["trajectory_id"],
        )[:count]
        selected.update((row["dataset"], row["trajectory_id"]) for row in rows)
    return [key for key in all_keys if key[:2] in selected]


def run_predictions(
    config: dict[str, Any],
    prepared: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    run_dir: Path,
) -> Path:
    taxonomies = {**sources, "native_merged": merged}
    inputs = {
        (row["dataset"], row["trajectory_id"]): row for row in prepared["inputs"]
    }
    all_keys = planned_keys(prepared["inputs"], int(config["experiment"]["seed"]))
    predictions_path = run_dir / "predictions.jsonl"
    completed = _latest_predictions(predictions_path)
    llm = CachedLLM(config, resolve_path(config, "cache") / "attribution")

    def execute(keys: list[tuple[str, str, str]], phase: str) -> None:
        nonlocal completed
        progress_every = int(config["execution"]["progress_every"])
        for key in keys:
            if key in completed:
                continue
            dataset, trajectory_id, condition = key
            record = inputs[(dataset, trajectory_id)]
            parsed, call, invalid_retries, truncation_retries = _call_step_model(
                config, llm, record, condition, taxonomies
            )
            result = {
                "dataset": dataset,
                "trajectory_id": trajectory_id,
                "task_hash": record["task_hash"],
                "framework": record["framework"],
                "benchmark": record["benchmark"],
                "condition": condition,
                "predicted_step_id": parsed.predicted_step_id,
                "explanation": parsed.explanation,
                "evidence": parsed.evidence,
                "candidate_step_ids": record["candidate_step_ids"],
                "candidate_step_ids_sha256": _sha256_json(record["candidate_step_ids"]),
                "visible_trajectory_sha256": _sha256_json(record["visible_trajectory"]),
                "model": config["models"]["attribution"]["name"],
                "cache_key": call["cache_key"],
                "cache_hit": call["cache_hit"],
                "invalid_step_retries": invalid_retries,
                "truncation_retries": truncation_retries,
                "response_metadata": call.get("response_metadata", {}),
            }
            append_jsonl(predictions_path, result)
            completed[key] = result
            done = len(completed)
            if done % progress_every == 0 or done == len(all_keys):
                print(
                    f"[step-localization:{phase}] completed={done}/{len(all_keys)} "
                    f"dataset={dataset} condition={condition}",
                    flush=True,
                )

    smoke = _smoke_keys(
        prepared["inputs"],
        all_keys,
        int(config["execution"]["smoke_trajectories_per_dataset"]),
    )
    print(
        f"[step-localization] smoke gate: {len(smoke)} rows; completed rows reuse cache/results",
        flush=True,
    )
    execute(smoke, "smoke")
    smoke_latest = _latest_predictions(predictions_path)
    if not all(key in smoke_latest for key in smoke):
        raise RuntimeError("Smoke gate did not complete all planned rows")
    if any(
        smoke_latest[key]["predicted_step_id"]
        not in inputs[key[:2]]["candidate_step_ids"]
        for key in smoke
    ):
        raise RuntimeError("Smoke gate produced an out-of-candidate step")
    print(
        f"[step-localization] full run: {len(all_keys)} rows; smoke rows are reused",
        flush=True,
    )
    execute(all_keys, "main")
    completed = _latest_predictions(predictions_path)
    missing = [key for key in all_keys if key not in completed]
    if missing:
        raise RuntimeError(f"Prediction run incomplete: {len(missing)} rows missing")
    return predictions_path


def _binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(0, min(successes, trials - successes) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    after: str,
    before: str,
    *,
    samples: int,
    seed: int,
    alpha: float,
) -> tuple[float, float, float]:
    by_trajectory: dict[str, dict[str, int]] = defaultdict(dict)
    task_by_trajectory: dict[str, str] = {}
    for row in rows:
        by_trajectory[row["trajectory_id"]][row["condition"]] = int(row["exact_correct"])
        task_by_trajectory[row["trajectory_id"]] = row["task_hash"]
    clusters: dict[str, list[float]] = defaultdict(list)
    for trajectory_id, outcomes in by_trajectory.items():
        clusters[task_by_trajectory[trajectory_id]].append(
            float(outcomes[after] - outcomes[before])
        )
    cluster_effects = [statistics.mean(values) for values in clusters.values()]
    point = statistics.mean(cluster_effects)
    rng = random.Random(seed)
    boot = [
        statistics.mean(
            cluster_effects[rng.randrange(len(cluster_effects))]
            for _ in cluster_effects
        )
        for _ in range(samples)
    ]
    return point, _percentile(boot, alpha / 2), _percentile(boot, 1 - alpha / 2)


def analyze(
    config: dict[str, Any],
    prepared: dict[str, Any],
    merged: dict[str, Any],
    merge_audit: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    latest = _latest_predictions(predictions_path)
    planned = planned_keys(prepared["inputs"], int(config["experiment"]["seed"]))
    missing = [key for key in planned if key not in latest]
    if missing:
        raise RuntimeError(f"Cannot analyze incomplete run; missing={len(missing)}")
    inputs = {
        (row["dataset"], row["trajectory_id"]): row for row in prepared["inputs"]
    }
    gold = {
        (row["dataset"], row["trajectory_id"]): row for row in prepared["gold"]
    }
    scored: list[dict[str, Any]] = []
    for key in planned:
        prediction = latest[key]
        base_key = key[:2]
        record = inputs[base_key]
        reference = gold[base_key]
        candidates = record["candidate_step_ids"]
        predicted = prediction["predicted_step_id"]
        gold_step = reference["gold_step_id"]
        predicted_rank = candidates.index(predicted)
        gold_rank = candidates.index(gold_step)
        distance = abs(predicted_rank - gold_rank)
        scored.append(
            {
                "dataset": key[0],
                "trajectory_id": key[1],
                "task_hash": record["task_hash"],
                "framework": record["framework"],
                "benchmark": record["benchmark"],
                "condition": key[2],
                "gold_step_id": gold_step,
                "predicted_step_id": predicted,
                "exact_correct": int(predicted == gold_step),
                "within_one_visible_step": int(distance <= 1),
                "absolute_visible_rank_error": distance,
                "candidate_count": len(candidates),
                "explanation": prediction["explanation"],
                "evidence": prediction["evidence"],
                "invalid_step_retries": prediction["invalid_step_retries"],
                "truncation_retries": prediction["truncation_retries"],
                "cache_hit": prediction["cache_hit"],
            }
        )
    scored.sort(key=lambda row: (row["dataset"], row["trajectory_id"], row["condition"]))
    _write_csv(
        run_dir / "scored_predictions.csv",
        scored,
        [
            "dataset",
            "trajectory_id",
            "task_hash",
            "framework",
            "benchmark",
            "condition",
            "gold_step_id",
            "predicted_step_id",
            "exact_correct",
            "within_one_visible_step",
            "absolute_visible_rank_error",
            "candidate_count",
            "explanation",
            "evidence",
            "invalid_step_retries",
            "truncation_retries",
            "cache_hit",
        ],
    )
    write_jsonl(run_dir / "scored_predictions.jsonl", scored)

    summaries = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            subset = [
                row
                for row in scored
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            nontrivial = [row for row in subset if row["candidate_count"] >= 2]
            summaries.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(subset),
                    "exact_step_accuracy": sum(row["exact_correct"] for row in subset) / len(subset),
                    "nontrivial_n": len(nontrivial),
                    "nontrivial_exact_step_accuracy": sum(
                        row["exact_correct"] for row in nontrivial
                    )
                    / len(nontrivial),
                    "within_one_visible_step_accuracy": sum(
                        row["within_one_visible_step"] for row in subset
                    )
                    / len(subset),
                    "mean_absolute_visible_rank_error": statistics.mean(
                        row["absolute_visible_rank_error"] for row in subset
                    ),
                    "median_absolute_visible_rank_error": statistics.median(
                        row["absolute_visible_rank_error"] for row in subset
                    ),
                    "invalid_step_retry_rows": sum(row["invalid_step_retries"] > 0 for row in subset),
                    "truncation_retry_rows": sum(row["truncation_retries"] > 0 for row in subset),
                }
            )
    _write_csv(
        run_dir / "results_summary.csv",
        summaries,
        list(summaries[0]),
    )

    scored_index = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row for row in scored
    }
    contrasts = []
    for dataset in DATASETS:
        trajectory_ids = sorted(
            row["trajectory_id"]
            for row in prepared["inputs"]
            if row["dataset"] == dataset
        )
        dataset_rows = [row for row in scored if row["dataset"] == dataset]
        for contrast_index, (after, before) in enumerate(CONTRASTS):
            improved = sum(
                scored_index[(dataset, trajectory_id, after)]["exact_correct"] == 1
                and scored_index[(dataset, trajectory_id, before)]["exact_correct"] == 0
                for trajectory_id in trajectory_ids
            )
            regressed = sum(
                scored_index[(dataset, trajectory_id, after)]["exact_correct"] == 0
                and scored_index[(dataset, trajectory_id, before)]["exact_correct"] == 1
                for trajectory_id in trajectory_ids
            )
            row_delta = statistics.mean(
                scored_index[(dataset, trajectory_id, after)]["exact_correct"]
                - scored_index[(dataset, trajectory_id, before)]["exact_correct"]
                for trajectory_id in trajectory_ids
            )
            cluster_delta, ci_low, ci_high = _cluster_bootstrap(
                dataset_rows,
                after,
                before,
                samples=int(config["analysis"]["bootstrap_resamples"]),
                seed=int(config["analysis"]["bootstrap_seed"]) + contrast_index + (0 if dataset == "whowhen" else 100),
                alpha=float(config["analysis"]["alpha"]),
            )
            contrasts.append(
                {
                    "dataset": dataset,
                    "after_condition": after,
                    "before_condition": before,
                    "n_trajectories": len(trajectory_ids),
                    "trajectory_weighted_exact_accuracy_delta": row_delta,
                    "task_cluster_weighted_delta": cluster_delta,
                    "task_cluster_bootstrap_ci_low": ci_low,
                    "task_cluster_bootstrap_ci_high": ci_high,
                    "improved": improved,
                    "regressed": regressed,
                    "mcnemar_exact_p": _binomial_two_sided(min(improved, regressed), improved + regressed),
                }
            )
    _write_csv(run_dir / "paired_contrasts.csv", contrasts, list(contrasts[0]))

    transition_rows = []
    for dataset in DATASETS:
        for row in prepared["inputs"]:
            if row["dataset"] != dataset:
                continue
            predictions = {
                condition: scored_index[(dataset, row["trajectory_id"], condition)]["predicted_step_id"]
                for condition in CONDITIONS
            }
            transition_rows.append(
                {
                    "dataset": dataset,
                    "trajectory_id": row["trajectory_id"],
                    "task_hash": row["task_hash"],
                    "gold_step_id": gold[(dataset, row["trajectory_id"])]["gold_step_id"],
                    **{f"{condition}_prediction": value for condition, value in predictions.items()},
                    "all_four_predictions_identical": len(set(predictions.values())) == 1,
                }
            )
    _write_csv(run_dir / "condition_transitions.csv", transition_rows, list(transition_rows[0]))

    ledger = _ledger_summary(config)
    status = {
        "status": "STEP_LOCALIZATION_EXPERIMENT_COMPLETE",
        "total_trajectories": len(prepared["inputs"]),
        "logical_prediction_rows": len(scored),
        "actual_api_calls_in_experiment_ledger": ledger["actual_api_calls"],
        "total_recorded_cost_usd": ledger["total_recorded_cost_usd"],
        "output_compliance_rate": 1.0,
        "all_predictions_in_visible_candidate_set": True,
        "merge_taxonomy_sha256": _sha256_json(merged),
        "merge_validation": merge_audit,
    }
    write_json(run_dir / "run_status.json", status)
    write_json(run_dir / "cost_summary.json", ledger)
    _freeze_json(run_dir / "native_merged_taxonomy.json", merged)
    _freeze_json(run_dir / "sample_manifest.json", prepared["manifest"])

    summary_lookup = {
        (row["dataset"], row["condition"]): row for row in summaries
    }
    report = [
        "# Failure-step localization results",
        "",
        f"- Status: {status['status']}",
        f"- Trajectories: {len(prepared['inputs'])}",
        f"- Logical predictions: {len(scored)}",
        f"- Merge model: {config['models']['merge']['name']}",
        f"- Step-localization model: {config['models']['attribution']['name']}",
        f"- Recorded API cost (USD): {ledger['total_recorded_cost_usd']:.6f}",
        "",
        "## Primary results",
        "",
        "| Dataset | Condition | N | Exact step | Nontrivial exact | Within +/-1 visible step | Mean rank error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            row = summary_lookup[(dataset, condition)]
            report.append(
                f"| {dataset} | {condition} | {row['n']} | "
                f"{row['exact_step_accuracy']:.2%} | "
                f"{row['nontrivial_exact_step_accuracy']:.2%} (n={row['nontrivial_n']}) | "
                f"{row['within_one_visible_step_accuracy']:.2%} | "
                f"{row['mean_absolute_visible_rank_error']:.3f} |"
            )
    report.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Exact step accuracy is primary. Within-one and rank distance are secondary because adjacent action/consequence events can be semantically close.",
            "- Nontrivial exact accuracy excludes trajectories with only one visible candidate; these rows cannot distinguish guidance conditions.",
            "- Candidate IDs came only from visible trajectory fields and were identical across all four conditions.",
            "- Taxonomy labels were never prediction targets and gold fields were never included in prompts.",
            "- Who&When uses one trajectory per task hash. AgentErrorBench retains all official trajectories and paired confidence intervals cluster by task hash.",
            "- native_merged_taxonomy is one fixed Who&When-AgentErrorBench merge used identically for both datasets.",
            "",
            "See scored_predictions.csv, results_summary.csv, paired_contrasts.csv, and condition_transitions.csv for full details.",
        ]
    )
    (run_dir / "FINAL_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "status": status["status"],
        "run_directory": str(run_dir),
        "final_results": str(run_dir / "FINAL_RESULTS.md"),
        "results_summary": str(run_dir / "results_summary.csv"),
        "paired_contrasts": str(run_dir / "paired_contrasts.csv"),
        "recorded_cost_usd": ledger["total_recorded_cost_usd"],
    }


def run_workflow(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
    analyze_only: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    prepared = prepare_data(config)
    sources = load_source_taxonomies(config)
    preflight = offline_preflight(config, prepared, sources)
    if preflight_only:
        return {
            "status": preflight["status"],
            "offline_preflight": str(HERE / "OFFLINE_PREFLIGHT.json"),
            "planned_trajectories": preflight["total_trajectories"],
            "planned_predictions": preflight["planned_logical_predictions"],
            "projected_upper_bound_usd": preflight["cost_preflight"]["projected_total_usd"],
        }

    merged, merge_audit = obtain_merged_taxonomy(
        config, sources, allow_generation=not analyze_only
    )
    run_id = _run_id(config, prepared, sources, merged)
    run_dir = resolve_path(config, "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "research_question": config["experiment"]["research_question"],
        "conditions": list(CONDITIONS),
        "models": config["models"],
        "sample_manifest_sha256": _sha256_json(prepared["manifest"]),
        "native_merged_taxonomy_sha256": _sha256_json(merged),
        "planned_trajectories": len(prepared["inputs"]),
        "planned_prediction_rows": len(prepared["inputs"]) * len(CONDITIONS),
        "candidate_step_policy": prepared["manifest"]["candidate_policy"],
        "prompt_difference_policy": "Only the diagnostic taxonomy block varies across conditions.",
    }
    _freeze_json(run_dir / "run_manifest.json", manifest)
    if not analyze_only:
        run_predictions(config, prepared, sources, merged, run_dir)
    return analyze(config, prepared, merged, merge_audit, run_dir)
