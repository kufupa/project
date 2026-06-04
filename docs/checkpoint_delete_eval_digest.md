# top-3 ckpt runs — eval digest

> src: `eval_summary.json` + `eval_sweep_summary.json` under each run dir  
> **no 25ep / 100ep real-env eval on disk** — only **10ep** (wm smoke **4ep**)  
> `tail_85_100` = train **updates** 85–100, not eval episode count

## delete cheat sheet

| run | disk | train | best real-env eval | last eval done | uneval ckpts | call |
|-----|------|-------|-------------------|----------------|--------------|------|
| no_tanh_rngfix | 60.7G | push-v3, g6, u→**195** | **40%** u25/u50 (10ep) | u100 (10ep) | **u105–195** (~19 ckpt) | peak early → stale tail → **delete ok** |
| vector_async | 31.9G | push-v3, g6, u→**99** | **20%** u15/30/40/45 (10ep) | u65 (10ep) | **u70–100** (~7 ckpt) | weak + partial eval → **delete ok** |
| wm_u100 | 50.0G | wm GRPO 278u, g4 | **10%** u180/210 subprocess only | 0% @u300 vec | many ckpt | real env ~0% → **delete ok** |

**combined recover if delete all 3:** ~142.5 GiB

---

## no_tanh_rngfix

| field | value |
|-------|-------|
| path | `phase111_on_grpo_lerobot/push-v3/g6_u100_seed2000_vector_async_no_tanh_rngfix` |
| disk | 60.7 GiB (2026-05-15) |
| train | u100–190, g6, vector_async |
| task | push-v3 | label no_tanh_rngfix | action no_tanh |
| progress.jsonl | 196 rows, last update **195** |
| checkpoints/ | 39 × update_*.pt + latest |
| eval gap | ckpts thru **u195**; real-env eval only thru **u100** |

### eval sweeps (2)

#### `eval_sweep_so_far_20260514_000430`

| meta | value |
|------|-------|
| task | push-v3 |
| eval_ep | **10** |
| seed_start | 1000 |
| pc range | 0% – 40% |
| best | u25/u50 → **40.0%** |
| last ckpt in sweep | u80 → 0.0% |

| update | pc_success% | eval_ep | avg_sum_reward |
|--------|-------------|---------|----------------|
| 5 | 10.0 | 10 | 79.8 |
| 10 | 10.0 | 10 | 53.7 |
| 15 | 0.0 | 10 | 83.5 |
| 20 | 20.0 | 10 | 310.3 |
| 25 | 40.0 | 10 | 47.4 |
| 30 | 20.0 | 10 | 56.0 |
| 35 | 20.0 | 10 | 28.7 |
| 40 | 20.0 | 10 | 78.8 |
| 45 | 30.0 | 10 | 53.2 |
| 50 | 40.0 | 10 | 50.1 |
| 55 | 20.0 | 10 | 290.5 |
| 60 | 20.0 | 10 | 49.4 |
| 65 | 20.0 | 10 | 63.4 |
| 70 | 30.0 | 10 | 101.1 |
| 75 | 20.0 | 10 | 46.9 |
| 80 | 0.0 | 10 | 338.0 |

#### `eval_sweep_tail_85_100_20260514_234513`

| meta | value |
|------|-------|
| task | push-v3 |
| eval_ep | **10** |
| seed_start | 1000 |
| update range | 85–100 stride 5? |
| pc range | 0% – 10% |
| best | u100 → **10.0%** |
| last ckpt in sweep | u100 → 10.0% |

| update | pc_success% | eval_ep | avg_sum_reward |
|--------|-------------|---------|----------------|
| 85 | 0.0 | 10 | 209.8 |
| 90 | 0.0 | 10 | 34.2 |
| 95 | 0.0 | 10 | 262.9 |
| 100 | 10.0 | 10 | 89.5 |

---

## vector_async

| field | value |
|-------|-------|
| path | `phase111_on_grpo_lerobot/push-v3/g6_u100_seed2000_vector_async` |
| disk | 31.9 GiB (2026-05-13) |
| train | u0–100, g6, vector_async |
| task | push-v3 | label — | action default |
| progress.jsonl | 100 rows, last update **99** |
| checkpoints/ | 20 × update_*.pt + latest |
| eval gap | ckpts thru **u100**; eval sweep stops **u65** |

### eval sweeps (1)

#### `eval_sweep_current`

| meta | value |
|------|-------|
| task | push-v3 |
| eval_ep | **10** |
| seed_start | 1000 |
| pc range | 0% – 20% |
| best | u15 → **20.0%** |
| last ckpt in sweep | u65 → 0.0% |

| update | pc_success% | eval_ep | avg_sum_reward |
|--------|-------------|---------|----------------|
| 5 | 0.0 | 10 | 353.9 |
| 10 | 10.0 | 10 | 43.9 |
| 15 | 20.0 | 10 | 51.8 |
| 20 | 10.0 | 10 | 47.4 |
| 25 | 0.0 | 10 | 39.8 |
| 30 | 20.0 | 10 | 51.5 |
| 35 | 0.0 | 10 | 26.6 |
| 40 | 20.0 | 10 | 40.1 |
| 45 | 20.0 | 10 | 41.4 |
| 50 | 10.0 | 10 | 29.1 |
| 55 | 0.0 | 10 | 277.1 |
| 60 | 10.0 | 10 | 45.8 |
| 65 | 0.0 | 10 | 265.8 |

---

## wm_u100

| field | value |
|-------|-------|
| path | `phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000` |
| disk | 50.0 GiB (2026-05-15) |
| train | wm GRPO, 278 updates, g4, chunk 25 |
| task | push-v3 | backend official_lerobot_guarded |
| reward | wm_latent_progress | action bounded_executed |
| progress.jsonl | 302 events, max update_index **299** |
| checkpoints/ | 32 × update_*.pt + latest |
| eval gap | `u0_u300` only u0–u30 (4 pts); main sweeps u40–300 all 0–10% |

### eval sweeps (3)

#### `bounded_eval_every10_u100_u300_onegpu_vec4`

| meta | value |
|------|-------|
| task | push-v3 |
| eval_ep | **10** |
| seed_start | 1000 |
| update range | 100–300 stride 10 |
| exec | inprocess_vector n_envs=4 |
| pc range | 0% – 0% |
| best | u100 → **0.0%** |
| last ckpt in sweep | u300 → 0.0% |

| update | pc_success% | eval_ep | avg_sum_reward |
|--------|-------------|---------|----------------|
| 100 | 0.0 | 10 | 4.2 |
| 110 | 0.0 | 10 | 5.2 |
| 120 | 0.0 | 10 | 4.8 |
| 130 | 0.0 | 10 | 8.2 |
| 140 | 0.0 | 10 | 9.5 |
| 150 | 0.0 | 10 | 46.0 |
| 160 | 0.0 | 10 | 13.7 |
| 170 | 0.0 | 10 | 30.5 |
| 180 | 0.0 | 10 | 25.6 |
| 190 | 0.0 | 10 | 25.5 |
| 200 | 0.0 | 10 | 43.9 |
| 210 | 0.0 | 10 | 30.0 |
| 220 | 0.0 | 10 | 26.7 |
| 230 | 0.0 | 10 | 33.3 |
| 240 | 0.0 | 10 | 31.3 |
| 250 | 0.0 | 10 | 51.9 |
| 260 | 0.0 | 10 | 37.9 |
| 270 | 0.0 | 10 | 24.8 |
| 280 | 0.0 | 10 | 29.3 |
| 290 | 0.0 | 10 | 11.8 |
| 300 | 0.0 | 10 | 12.0 |

#### `bounded_eval_every10_u40_u300_subprocess`

| meta | value |
|------|-------|
| task | push-v3 |
| eval_ep | **10** |
| seed_start | 1000 |
| update range | 40–300 stride 10 |
| pc range | 0% – 10% |
| best | u180 → **10.0%** |
| last ckpt in sweep | u300 → 0.0% |

| update | pc_success% | eval_ep | avg_sum_reward |
|--------|-------------|---------|----------------|
| 40 | 0.0 | 10 | 29.0 |
| 50 | 0.0 | 10 | 23.5 |
| 60 | 0.0 | 10 | 26.3 |
| 70 | 0.0 | 10 | 15.9 |
| 80 | 0.0 | 10 | 11.2 |
| 90 | 0.0 | 10 | 9.6 |
| 100 | 0.0 | 10 | 8.0 |
| 110 | 0.0 | 10 | 9.3 |
| 120 | 0.0 | 10 | 11.9 |
| 130 | 0.0 | 10 | 9.8 |
| 140 | 0.0 | 10 | 19.4 |
| 150 | 0.0 | 10 | 299.7 |
| 160 | 0.0 | 10 | 88.1 |
| 170 | 0.0 | 10 | 46.9 |
| 180 | 10.0 | 10 | 65.6 |
| 190 | 0.0 | 10 | 122.0 |
| 200 | 0.0 | 10 | 101.7 |
| 210 | 10.0 | 10 | 92.1 |
| 220 | 0.0 | 10 | 69.7 |
| 230 | 0.0 | 10 | 60.0 |
| 240 | 0.0 | 10 | 85.6 |
| 250 | 0.0 | 10 | 82.3 |
| 260 | 0.0 | 10 | 90.8 |
| 270 | 0.0 | 10 | 74.2 |
| 280 | 0.0 | 10 | 59.3 |
| 290 | 0.0 | 10 | 78.2 |
| 300 | 0.0 | 10 | 162.7 |

#### `bounded_eval_vec_smoke_u100_vec4`

| meta | value |
|------|-------|
| task | push-v3 |
| eval_ep | **4** |
| seed_start | 1000 |
| update range | 100–100 stride 10 |
| exec | inprocess_vector n_envs=4 |
| pc range | 0% – 0% |
| best | u100 → **0.0%** |
| last ckpt in sweep | u100 → 0.0% |

| update | pc_success% | eval_ep | avg_sum_reward |
|--------|-------------|---------|----------------|
| 100 | 0.0 | 4 | 7.8 |

### `bounded_eval_every10_u0_u300` (no sweep summary)

| ckpt dir | pc% | ep |
|----------|-----|-----|
| update_0000 | 20.0 | 10 |
| update_0010 | 20.0 | 10 |
| update_0020 | 10.0 | 10 |
| update_0030 | 0.0 | 10 |

### other dirs (no pc_success json)

| dir | note |
|-----|------|
| `bounded_eval_every10_u0_u300_fast/` | empty / no eval_summary |
| `oracle/` | seed dirs only, no eval_summary |
| `rollouts/` | train rollouts, not held-out eval |

---

## paths (absolute)

```
/vol/bitbucket/aa6622/project/artifacts/phase111_on_grpo_lerobot/push-v3/g6_u100_seed2000_vector_async_no_tanh_rngfix
/vol/bitbucket/aa6622/project/artifacts/phase111_on_grpo_lerobot/push-v3/g6_u100_seed2000_vector_async
/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000
```

