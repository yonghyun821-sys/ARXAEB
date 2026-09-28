# AgentRx + AgentErrorBench step-localization experiment

이 실험은 앞선 attribution 실험과 동일한 AgentRx 73개 및 AgentErrorBench 171개 trajectory에서 taxonomy guidance가 공식 failure step을 찾는 데 영향을 주는지 측정한다. API inference는 새로 수행하며 기존 step prediction/cache는 재사용하지 않는다.

## 설계

| 항목 | 설정 |
|---|---|
| Dataset | AgentRx 73 + AgentErrorBench 171 |
| Conditions | Label-Only, Own, Foreign, Merged |
| Predictions | `244 × 4 = 976` |
| Model | `google/gemini-2.5-flash`, temperature 0 |
| Output | visible candidate step ID 하나 |
| Primary metric | exact-step accuracy |
| Secondary metrics | within-±1 visible-step accuracy, mean/median absolute rank error |
| Cost cap | `$25` |

네 조건은 동일한 task, trajectory 및 전체 visible candidate-step list를 받는다. 달라지는 것은 diagnostic guidance뿐이다.

- Label-Only: 해당 dataset native taxonomy의 category ID와 이름만 제공
- Own: 해당 dataset native taxonomy의 정의 제공
- Foreign: 상대 dataset taxonomy의 정의 제공
- Merged: 앞선 attribution 실험에서 새로 생성한 frozen GPT-5 merged taxonomy 제공

Taxonomy category는 출력 target이 아니다. Gold step은 별도 frozen gold 파일에 보관하고 prediction 완료 후에만 결합한다.

## `.env`

repository root의 `.env`에 다음을 추가한다.

```dotenv
AGENTRX_AEB_STEP_LOCALIZATION_OPENROUTER_API_KEY=sk-or-v1-본인키
```

## Offline preflight

API를 호출하지 않는다.

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_step_localization_final_protocol\run.py --preflight-only
```

## 본 실험

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_step_localization_final_protocol\run.py
```

중단되면 같은 명령을 다시 실행하면 완료 prediction과 cache를 재사용해 이어서 진행한다. 여러 terminal에서 동시에 실행하면 안 된다.

## 결과 파일

- `FINAL_RESULTS.md`: 핵심 결과표와 해석 경계
- `results_summary.csv`: dataset × condition별 step 지표
- `paired_contrasts.csv`: paired exact-accuracy delta, task-cluster bootstrap CI, McNemar p
- `condition_transitions.csv`: 조건별 step prediction 변화
- `scored_predictions.csv`: prediction과 사후 결합 gold
- `cost_summary.json`: 기록된 API 비용

