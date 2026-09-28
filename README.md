# AgentRx–AgentErrorBench Semifinal Reproduction Bundle

이 폴더에는 다음 두 실험을 다른 컴퓨터에서 재실행하는 데 필요한 코드와
frozen sample이 들어 있습니다.

1. 오류 원인 attribution prediction
2. 공식 failure step localization

두 실험 모두 동일한 표본을 사용합니다.

- AgentRx: 73 trajectories
- AgentErrorBench: 171 trajectories
- 조건: Label-Only, Own Taxonomy, Foreign Taxonomy, Merged Taxonomy
- prediction model: `google/gemini-2.5-flash`, temperature 0
- merged taxonomy: 기존 실험에서 GPT-5로 생성하고 검증한 frozen artifact

현재 offline preflight의 보수적 상한은 attribution 약 `$11.23`, step
localization 약 `$11.14`입니다. 각 config의 runtime hard cap은 각각 `$23`,
`$25`이며 실제 OpenRouter 계정 한도와는 별개입니다. Frozen GPT-5 merge를
재사용하므로 taxonomy merge API 비용은 다시 발생하지 않습니다.

원본 dataset 전체는 포함하지 않았습니다. 대신 실제 실험에 사용된 정확한
frozen inputs/gold/taxonomy를 포함했기 때문에 Git clone 후 같은 244개 표본을
곧바로 재실행할 수 있습니다. Gold는 inference prompt에 포함되지 않고 평가
단계에서만 결합됩니다.

## 1. 설치

Windows PowerShell에서 이 `semifinal` 폴더를 repository root로 사용합니다.

```powershell
cd semifinal
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

`.env`를 열고 두 항목에 같은 OpenRouter API key를 넣어도 됩니다.

```dotenv
AGENTRX_AEB_FINAL_PROTOCOL_OPENROUTER_API_KEY=sk-or-v1-...
AGENTRX_AEB_STEP_LOCALIZATION_OPENROUTER_API_KEY=sk-or-v1-...
```

`.env`는 `.gitignore`에 포함되어 있으므로 Git에 올리지 마세요.

복사 또는 clone 과정에서 핵심 frozen artifact가 변하지 않았는지 확인합니다.

```powershell
.\.venv\Scripts\python.exe verify_bundle.py
```

정상이면 `BUNDLE_VERIFIED`가 출력됩니다.

## 2. Offline preflight

API 호출 없이 bundle과 비용·prompt 조건을 검증합니다.

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_final_protocol_gemini25flash\run.py --preflight-only

.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_step_localization_final_protocol\run.py --preflight-only
```

## 3. 실행 순서

먼저 오류 원인 attribution을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_final_protocol_gemini25flash\run.py
```

그다음 동일 표본의 step localization을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_step_localization_final_protocol\run.py
```

각 명령은 중단되면 동일한 명령으로 재개할 수 있습니다. 여러 terminal에서
같은 실험을 동시에 실행하면 안 됩니다.

## 4. 결과 위치

오류 원인 실험:

```text
experiments/agentrx_agenterrorbench_final_protocol_gemini25flash/runs/<run-id>/
```

Step 실험:

```text
experiments/agentrx_agenterrorbench_step_localization_final_protocol/runs/<run-id>/
```

완료 후 그래프와 표 그림을 다시 만들려면:

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_final_protocol_gemini25flash\plot_accuracy.py

.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_step_localization_final_protocol\plot_exact_step_accuracy.py
```

기존 실행에서 생성했던 결과와 SVG 그림은 API cache와 분리하여 다음 위치에
보존했습니다.

```text
reference_results/attribution/
reference_results/step_localization/
```

## 5. 포함·제외 범위

포함:

- 정확한 frozen inputs와 사후 평가용 gold
- AgentRx/AEB native taxonomy와 GPT-5 merged taxonomy
- 모든 prompt 및 실행·평가 코드
- 공통 Python package 코드
- 기존 결과 CSV/Markdown 및 Accuracy 그래프·표 SVG

제외:

- OpenRouter API key
- 기존 API response cache
- 기존 API cost ledger
- 원본 dataset 전체

따라서 이 폴더만 별도 Git repository로 올린 뒤 다른 컴퓨터에서 clone하여
실험할 수 있습니다. 기존 결과를 단순 확인할 때는 `reference_results`만 보면
되고, 새 API prediction을 만들 때는 위 실행 순서를 따르면 됩니다.
