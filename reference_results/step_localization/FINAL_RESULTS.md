# AgentRx–AgentErrorBench failure-step localization results

| Dataset | Condition | N | Exact step | Within ±1 | Mean rank error | Median rank error |
|---|---|---:|---:|---:|---:|---:|
| agentrx | label_only | 73 | 24.66% | 31.51% | 18.603 | 5.000 |
| agentrx | own_taxonomy | 73 | 26.03% | 32.88% | 16.110 | 4.000 |
| agentrx | foreign_taxonomy | 73 | 20.55% | 27.40% | 16.644 | 5.000 |
| agentrx | merged_taxonomy | 73 | 19.18% | 24.66% | 19.205 | 5.000 |
| agenterrorbench | label_only | 171 | 22.22% | 36.26% | 6.427 | 3.000 |
| agenterrorbench | own_taxonomy | 171 | 23.98% | 37.43% | 5.696 | 3.000 |
| agenterrorbench | foreign_taxonomy | 171 | 19.88% | 31.58% | 6.661 | 3.000 |
| agenterrorbench | merged_taxonomy | 171 | 19.30% | 33.33% | 6.099 | 3.000 |

## Interpretation boundary

- Primary metric is exact visible-step accuracy; within-one accuracy and rank distance are secondary.
- All four conditions receive the identical complete visible candidate-step list.
- Taxonomy categories are diagnostic guidance only and are never prediction targets.
- Gold step is joined only after prediction and is never included in prompts.
- Label-Only includes native category IDs/names but no definitions.
- The same frozen GPT-5 merged taxonomy from the preceding attribution experiment is used for both datasets.
- Results are reported as observed without enforcing an expected condition ordering.
