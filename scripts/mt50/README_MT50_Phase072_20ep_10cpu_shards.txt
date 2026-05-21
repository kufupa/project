MT50 Phase072 - official LeRobot eval, 50 tasks x 20 episodes, 10 CPU jobs
================================================================================

What this is
------------
- Same checkpoint and wrapper as Phase072 GPU runs: scripts/mt50/run_official_lerobot_mt50_eval.sh
- Ten independent PBS jobs, no array, no ngpus request.
- Each job runs 5 MetaWorld tasks x 20 episodes, seed 1030, no videos.
- Outputs: artifacts/MT50_Phase072_official_lerobot_20ep_s1030_10cpu/shard0 ... shard9/

Why this layout
---------------
- GPU queue is blocked by placement-set/Qlist scheduler behavior.
- CPU-only smoke ran, so CPU jobs route differently and can be used while GPU support is pending.
- Ten shards offset slower CPU inference by reducing each job to 5 tasks.

PBS resources per job
---------------------
  #PBS -l select=1:ncpus=4:mem=16gb
  #PBS -l walltime=12:00:00

Notes:
- 16gb is conservative: partial CPU smoke peaked around 4.2 GiB RSS / 9.8 GiB vmem.
- 12h is a runtime guess, not proven full-run timing. If jobs time out, rerun failed shard(s) with higher walltime.
- The script sets CUDA_VISIBLE_DEVICES="" and caps common BLAS/OpenMP thread pools to 4.

Queue all ten jobs
------------------
  cd /rds/general/user/aa6622/home/project
  bash scripts/mt50/submit_mt50_phase072_20ep_all_10cpu_pbs.sh

Or submit one shard manually
----------------------------
  cd /rds/general/user/aa6622/home/project
  qsub -N mt50p072c0 \
    -v MT50_PHASE072_SHARD_INDEX=0 \
    scripts/mt50/submit_mt50_phase072_20ep_10cpu_single_shard.pbs

Shard 0 - 5 tasks
-----------------
assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3

Shard 1 - 5 tasks
-----------------
button-press-topdown-wall-v3,button-press-v3,button-press-wall-v3,coffee-button-v3,coffee-pull-v3

Shard 2 - 5 tasks
-----------------
coffee-push-v3,dial-turn-v3,disassemble-v3,door-close-v3,door-lock-v3

Shard 3 - 5 tasks
-----------------
door-open-v3,door-unlock-v3,drawer-close-v3,drawer-open-v3,faucet-close-v3

Shard 4 - 5 tasks
-----------------
faucet-open-v3,hammer-v3,hand-insert-v3,handle-press-side-v3,handle-press-v3

Shard 5 - 5 tasks
-----------------
handle-pull-side-v3,handle-pull-v3,lever-pull-v3,peg-insert-side-v3,peg-unplug-side-v3

Shard 6 - 5 tasks
-----------------
pick-out-of-hole-v3,pick-place-v3,pick-place-wall-v3,plate-slide-back-side-v3,plate-slide-back-v3

Shard 7 - 5 tasks
-----------------
plate-slide-side-v3,plate-slide-v3,push-back-v3,push-v3,push-wall-v3

Shard 8 - 5 tasks
-----------------
reach-v3,reach-wall-v3,shelf-place-v3,soccer-v3,stick-pull-v3

Shard 9 - 5 tasks
-----------------
stick-push-v3,sweep-into-v3,sweep-v3,window-open-v3,window-close-v3

After all ten jobs finish - merge then summarize
------------------------------------------------
  cd /rds/general/user/aa6622/home/project
  python scripts/mt50/merge_official_lerobot_eval_shards.py \
    --parent artifacts/MT50_Phase072_official_lerobot_20ep_s1030_10cpu \
    --shards shard0 shard1 shard2 shard3 shard4 shard5 shard6 shard7 shard8 shard9 \
    --merged-eval-info artifacts/MT50_Phase072_official_lerobot_20ep_s1030_10cpu/eval_info.json

  python scripts/mt50/summarize_official_lerobot_eval.py \
    --eval-info artifacts/MT50_Phase072_official_lerobot_20ep_s1030_10cpu/eval_info.json \
    --output artifacts/MT50_Phase072_official_lerobot_20ep_s1030_10cpu/MT50_Phase072_official_index.json \
    --run-root artifacts/MT50_Phase072_official_lerobot_20ep_s1030_10cpu \
    --seed 1030 \
    --phase MT50_Phase072
