# Phase46 overnight GRPO

**Train:** `submit_phase46_*_a30.slurm` on partition **a30** only.

**Eval:** RLinf fast — `RLinf/scripts/run_phase46_tiered_eval_rlinf.sh`  
(seeds **1000**, 150 steps, chunk 5). **Do not** use Phase111 vector 100ep or `max_steps=0` sweeps overnight.

**Launch:** `PHASE46_FOLLOW=0 bash scripts/grpo/run_overnight_phase46.sh --logprob-mode gaussian`

**Monitor:** `artifacts/phase46/latest/jobs_manifest.jsonl` + `autopilot.log`
