from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "experiments/agentrx_agenterrorbench_final_protocol_gemini25flash/prepared/frozen_inputs.jsonl": "712bbfe1496cde3a6ed84dda10164977e3b19a23e52c884c1e0aba376bca9355",
    "experiments/agentrx_agenterrorbench_final_protocol_gemini25flash/prepared/frozen_gold.jsonl": "0926f397698ff563b7dde6788a2159345625e4c006e95a60e7fce639c64e0715",
    "experiments/agentrx_agenterrorbench_step_localization_final_protocol/prepared/frozen_inputs.jsonl": "5555200fb01fe2af3fc5c8846547348cf1dda4a68cf1c7d3bb60b03149906b1f",
    "experiments/agentrx_agenterrorbench_step_localization_final_protocol/prepared/frozen_gold.jsonl": "a63eb8838a97c7caf1c2f9650e9d7bcacc96b6f9499f39ed27380f0b6db7e0f8",
    "taxonomies/agentrx_taxonomy.json": "6b1119799792cadcb24a29d8ee403a4abcb7fec62dbd7aeae3a8bb21cba06b51",
    "taxonomies/agenterror_taxonomy.json": "37c2fc888fc357952ed4d3bb1602816d47912db162e71cf62883fe12f3361e4e",
    "experiments/agentrx_agenterrorbench_final_protocol_gemini25flash/frozen/native_merged_taxonomy.json": "023534673257835063ae3950b55eb36125ffd4efc82f9612df6a97226fd583b5",
    "reference/prior_joint_predictions_scored.csv": "4d7370f0fc06573f673e0e2674ac7851c1aed6acd3d19e555f859840b1ef2307",
}


def main() -> None:
    failures = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"
        status = "OK" if actual == expected else "FAIL"
        print(f"[{status}] {relative}")
        if status == "FAIL":
            failures.append((relative, expected, actual))
    if failures:
        details = "\n".join(
            f"{path}: expected={expected}, actual={actual}"
            for path, expected, actual in failures
        )
        raise SystemExit(f"Bundle verification failed:\n{details}")
    print("BUNDLE_VERIFIED")


if __name__ == "__main__":
    main()
