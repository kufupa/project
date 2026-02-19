# Push-v3 Oracle Top-15 SmolVLA Pilot Plan and Snapshot

- Generated: 2026-04-11 15:37:30 UTC
- Oracle run dir: `/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093`
- Task: `push-v3`
- Top-K: 15
- Selection source: `optimal_report.json` sorted by top-k, resolved with `run_manifest.json` reset seeds

## Oracle top-15 rows used for SmolVLA target selection

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

## Rationale
- Keep the exact top-15 mapping from the latest Oracle `push-v3` baseline run as the canonical target set.
- Run one SmolVLA episode first on the top-ranked environment only (pilot).
- After manual pilot verification, expand to 4 episodes per environment and pick best by deterministic reward ranking.
- Do not save raw frames; videos, actions, and reward curves remain generated for each run.
- Keep all logic under `/project` as requested.

## Pilot command

```bash
bash project/scripts/smolvla/run_oracle_topk_smolvla_pilot.sh
```

## Notes
- SmolVLA checkpoint configured as `jadechoghari/smolvla_metaworld` via `SMOLVLA_INIT_CHECKPOINT`.
- Runtime backend is `lerobot_metaworld`.
