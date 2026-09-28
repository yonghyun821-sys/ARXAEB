# AgentRx–AgentErrorBench final-protocol 결과

AEGIS–Who&When 최종 실험과 같은 4조건 attribution-only protocol을 적용한 결과다.
모든 조건은 동일한 dataset-native output label 목록을 사용하며 diagnostic guidance만 달라진다.

| Dataset | Condition | N | Accuracy | Macro-F1 | Terminal invalid |
|---|---|---:|---:|---:|---:|
| agentrx | label_only | 73 | 23.29% | 0.1261 | 0 |
| agentrx | own_taxonomy | 73 | 26.03% | 0.1343 | 0 |
| agentrx | foreign_taxonomy | 73 | 19.18% | 0.1401 | 0 |
| agentrx | merged_taxonomy | 73 | 24.66% | 0.2348 | 0 |
| agenterrorbench | label_only | 171 | 14.04% | 0.0971 | 0 |
| agenterrorbench | own_taxonomy | 171 | 15.79% | 0.0869 | 0 |
| agenterrorbench | foreign_taxonomy | 171 | 14.62% | 0.1113 | 0 |
| agenterrorbench | merged_taxonomy | 171 | 16.96% | 0.1460 | 1 |

## 해석 경계

- 주지표는 native category ID exact-match accuracy이며 Macro-F1은 보조지표다.
- gold category는 모든 prediction이 끝난 뒤 결합한다.
- prompt에는 gold label, gold reason, gold step 또는 step-localization target이 들어가지 않는다.
- Foreign/Merged taxonomy는 진단 guidance일 뿐 output label space를 바꾸지 않는다.
- 기대한 조건 순서와 다르더라도 결과를 그대로 보고한다.

## 비용

- 기록된 API 비용: `$7.311831`
- 설정된 hard cap: `$23.00`
