# Push-v3 Oracle Top-15 SmolVLA Evaluation Setup

- Generated: 2026-04-11 15:38:00 UTC
- Pipeline: SmolVLA Top-15 based on Oracle push-v3 baseline
- Oracle run used: `run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093`
- Task: `push-v3`
- Top-K: `15`
- Selection source: `optimal_report.json` rows joined with `run_manifest.json` reset_seed map.

## Top-15 target rows
| rank | episode_index | reset_seed | oracle_max_reward | oracle_sum_reward |
|---:|---:|---:|---:|---:|
| 1 | 3 | 1003 | 10.000000 | 675.818323 |
| 2 | 53 | 1053 | 10.000000 | 675.818323 |
| 3 | 16 | 1016 | 10.000000 | 661.689863 |
| 4 | 44 | 1044 | 10.000000 | 625.740145 |
| 5 | 39 | 1039 | 10.000000 | 621.677083 |
| 6 | 8 | 1008 | 10.000000 | 613.982350 |
| 7 | 58 | 1058 | 10.000000 | 613.982350 |
| 8 | 10 | 1010 | 10.000000 | 601.372515 |
| 9 | 42 | 1042 | 10.000000 | 600.088696 |
| 10 | 9 | 1009 | 10.000000 | 599.461609 |
| 11 | 59 | 1059 | 10.000000 | 599.461609 |
| 12 | 5 | 1005 | 10.000000 | 596.476320 |
| 13 | 55 | 1055 | 10.000000 | 596.476320 |
| 14 | 33 | 1033 | 10.000000 | 595.132527 |
| 15 | 28 | 1028 | 10.000000 | 594.740734 |

## Execution plan
- Step 1 (this run): pilot one episode on target rank #1 only.
- Step 2 (later): run 4 episodes per target and keep best episode by `sum_reward`, then `max_reward`, then earliest `episode_index`.
- Frame output: disabled for these runs (`SMOLVLA_SAVE_FRAMES=false`).

## Why this run exists
This run captures the exact top-15 Oracle-proven seeds for push-v3 to provide a deterministic target set before scaling out to 4x-per-environment SmolVLA evaluation.
