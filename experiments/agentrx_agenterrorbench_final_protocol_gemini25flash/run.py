from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.agentrx_agenterrorbench_final_protocol_gemini25flash.core import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final AEGIS-Who&When attribution protocol on AgentRx and AgentErrorBench."
    )
    parser.add_argument(
        "--config",
        default="experiments/agentrx_agenterrorbench_final_protocol_gemini25flash/experiment.yaml",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    outputs = run_workflow(
        args.config,
        preflight_only=args.preflight_only,
        analyze_only=args.analyze_only,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
