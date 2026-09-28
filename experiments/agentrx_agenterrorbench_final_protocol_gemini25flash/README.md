# AgentRx + AgentErrorBench final-protocol extension

이 실험은 AEGIS–Who&When 최종 실험의 핵심 protocol을 AgentRx와 AgentErrorBench에 그대로 적용하는 독립 추가실험이다. 기존 결과나 cache를 덮어쓰거나 재사용하지 않는다.

## 고정 설계

| 항목 | 설정 |
|---|---|
| AgentRx | 공식 annotation과 매칭되는 73 trajectories |
| AgentErrorBench | fine-grained native gold가 유효한 171 trajectories |
| Conditions | Label-Only, Own, Foreign, GPT-5 Merged |
| 총 prediction | `(73 + 171) × 4 = 976` |
| Model | `google/gemini-2.5-flash`, temperature `0` |
| Output space | 모든 조건에서 해당 dataset의 동일한 native category IDs |
| Primary metric | native category exact-match accuracy |
| Secondary metric | Macro-F1 |
| Merge model | fresh `openai/gpt-5`, reasoning effort `medium` |
| Hard cap | Merge `$2` + attribution `$23` = combined `$25` |

Label-Only에는 label ID와 이름만, Own/Foreign/Merged에는 각각의 taxonomy 정의를 diagnostic guidance로 제공한다. Foreign과 Merged도 output label space를 바꾸지 않는다. Gold label과 gold step은 prompt에 제공하지 않으며 prediction 완료 후 scoring 때만 결합한다.

Merged taxonomy는 이 새 실험에서 GPT-5가 두 source taxonomy의 metadata만 보고 **새로 한 번 생성**한다. trajectory, gold, 기존 prediction/result는 merge 입력에 사용하지 않는다. 모든 source category provenance가 정확히 한 번 포함됐는지 자동 검증을 통과해야만 attribution을 시작한다. 생성된 merge는 이 폴더에 동결되므로 중단 후 재실행할 때 다시 호출하지 않는다.

## 1. `.env`

repository root의 `.env`에 다음 한 줄을 추가한다.

```dotenv
AGENTRX_AEB_FINAL_PROTOCOL_OPENROUTER_API_KEY=sk-or-v1-본인키
```

## 2. 선택적 offline 확인

API를 호출하지 않는다.

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_final_protocol_gemini25flash\run.py --preflight-only
```

## 3. 본 실험

```powershell
.\.venv\Scripts\python.exe -u experiments\agentrx_agenterrorbench_final_protocol_gemini25flash\run.py
```

중단되면 같은 명령을 다시 실행한다. 새 폴더 내부의 완료 prediction과 cache를 재사용해 이어서 수행한다. 같은 명령을 여러 terminal에서 동시에 실행하면 안 된다.

첫 실행 순서는 `GPT-5 fresh merge → provenance 검증/동결 → Gemini 2.5 Flash 976 predictions → 분석`이다.

## 4. 결과

완료 시 terminal에 실제 run directory와 핵심 파일이 출력된다.

- `FINAL_RESULTS.md`: 주 결과표와 해석 경계
- `condition_results.csv`: dataset × condition Accuracy/Macro-F1
- `paired_contrasts.csv`: paired accuracy delta, cluster bootstrap CI, exact McNemar p
- `condition_transitions.csv`: 조건 사이 prediction 변화/개선/악화
- `per_class_metrics.csv`: category별 precision/recall/F1
- `predictions_scored.csv`: prediction과 사후 결합 gold
- `cost_summary.json`: 기록된 API 비용
- `frozen/native_merged_taxonomy.json`: 이번 실행에서 새로 생성된 GPT-5 merge
- `frozen/native_merged_taxonomy_audit.json`: merge provenance 검증 기록
- `runtime/merge_api_call_ledger.jsonl`: merge 호출 비용 기록

결과가 예상한 순서와 다르더라도 그대로 보고하도록 설계되어 있다.
