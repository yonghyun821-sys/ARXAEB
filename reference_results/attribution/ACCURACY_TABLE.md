# Attribution Accuracy

| Dataset | N | Label-Only | Own Taxonomy | Foreign Taxonomy | Merged Taxonomy |
|---|---:|---:|---:|---:|---:|
| AgentRx | 73 | 23.29% | **26.03%** | 19.18% | 24.66% |
| AgentErrorBench | 171 | 14.04% | 15.79% | 14.62% | **16.96%** |

**굵은 값**은 각 dataset에서 가장 높은 accuracy다.

Model: `google/gemini-2.5-flash`  
Metric: native-label exact-match accuracy

