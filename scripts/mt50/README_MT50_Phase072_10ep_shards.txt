MT50 Phase072 — official LeRobot eval, 50 tasks × 10 episodes, 3 parallel GPU shards
==================================================================================

CX3 PBS — 50 tasks × 20 episodes, seed 1030, 4 parallel subjobs (one array)
---------------------------------------------------------------------------
- Script: scripts/mt50/submit_mt50_phase072_20ep_4job_array.pbs
- Resources per subjob: select=1:ncpus=2:mem=16gb:ngpus=1, walltime=24:00:00
- Forced: lerobot_mw_py312 venv, MUJOCO_GL=glfw, MT50_LEROBOT_MAX_EPISODES_RENDERED=0 (no videos)
- Outputs: artifacts/MT50_Phase072_official_lerobot_20ep_s1030_4job/shard0|shard1|shard2|shard3/
- Task split (train_config order): 13 + 13 + 12 + 12 tasks; lists are embedded in the PBS file (same comma-in-qsub -v caveat as Slurm).

Submit (run from project root so PBS_O_WORKDIR resolves correctly):

  cd .../project
  qsub scripts/mt50/submit_mt50_phase072_20ep_4job_array.pbs

After all four subjobs finish — merge then summarize (seed 1030):

  cd .../project
  python scripts/mt50/merge_official_lerobot_eval_shards.py \
    --parent artifacts/MT50_Phase072_official_lerobot_20ep_s1030_4job \
    --shards shard0 shard1 shard2 shard3 \
    --merged-eval-info artifacts/MT50_Phase072_official_lerobot_20ep_s1030_4job/eval_info.json

  python scripts/mt50/summarize_official_lerobot_eval.py \
    --eval-info artifacts/MT50_Phase072_official_lerobot_20ep_s1030_4job/eval_info.json \
    --output artifacts/MT50_Phase072_official_lerobot_20ep_s1030_4job/MT50_Phase072_official_index.json \
    --run-root artifacts/MT50_Phase072_official_lerobot_20ep_s1030_4job \
    --seed 1030 \
    --phase MT50_Phase072

Shard 0 — 13 tasks (20-ep PBS array)
------------------------------------
assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3,button-press-topdown-wall-v3,button-press-v3,button-press-wall-v3,coffee-button-v3,coffee-pull-v3,coffee-push-v3,dial-turn-v3,disassemble-v3

Shard 1 — 13 tasks
------------------
door-close-v3,door-lock-v3,door-open-v3,door-unlock-v3,drawer-close-v3,drawer-open-v3,faucet-close-v3,faucet-open-v3,hammer-v3,hand-insert-v3,handle-press-side-v3,handle-press-v3,handle-pull-side-v3

Shard 2 — 12 tasks
------------------
handle-pull-v3,lever-pull-v3,peg-insert-side-v3,peg-unplug-side-v3,pick-out-of-hole-v3,pick-place-v3,pick-place-wall-v3,plate-slide-back-side-v3,plate-slide-back-v3,plate-slide-side-v3,plate-slide-v3,push-back-v3

Shard 3 — 12 tasks
------------------
push-v3,push-wall-v3,reach-v3,reach-wall-v3,shelf-place-v3,soccer-v3,stick-pull-v3,stick-push-v3,sweep-into-v3,sweep-v3,window-open-v3,window-close-v3

What this is
------------
- Same checkpoint and wrapper as Phase071: scripts/mt50/run_official_lerobot_mt50_eval.sh
- Each shard runs lerobot-eval with a disjoint comma-separated --env.task list (17 + 17 + 16 tasks).
- Episodes per task: MT50_PHASE071_EPISODES=10 (set inside each shard Slurm file).
- Outputs: artifacts/MT50_Phase072_official_lerobot_10ep/shard0|shard1|shard2/
- Future shard reruns default to MT50_LEROBOT_MAX_EPISODES_RENDERED=0, so no videos are rendered.
- For 10 parallel GPUs (5 tasks each, same 50-task order), see README_MT50_Phase072_10ep_10gpu_shards.txt

Why shards are embedded in Slurm (do not use sbatch --export for task lists)
-----------------------------------------------------------------------------
Slurm splits --export on commas. A comma-separated task list in --export breaks.
Task lists are single-quoted inside: submit_mt50_phase072_10ep_shard{0,1,2}.slurm

Commands that were used to queue all three jobs
------------------------------------------------
  cd /vol/bitbucket/aa6622/project
  sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase072_10ep_shard0.slurm
  sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase072_10ep_shard1.slurm
  sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase072_10ep_shard2.slurm

Equivalent one-liner (see also submit_mt50_phase072_10ep_all_shards.sh):
  bash scripts/mt50/submit_mt50_phase072_10ep_all_shards.sh

Push-v3 only, 10 episodes, no videos
------------------------------------
  cd /vol/bitbucket/aa6622/project
  sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL \
    scripts/mt50/submit_mt50_phase072_official_push_10ep_no_video.slurm

Output:
  artifacts/MT50_Phase072_official_lerobot_push_10ep/

Shard 0 — 17 tasks (train_config.json order)
----------------------------------------------
assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3,button-press-topdown-wall-v3,button-press-v3,button-press-wall-v3,coffee-button-v3,coffee-pull-v3,coffee-push-v3,dial-turn-v3,disassemble-v3,door-close-v3,door-lock-v3,door-open-v3,door-unlock-v3

Shard 1 — 17 tasks
------------------
drawer-close-v3,drawer-open-v3,faucet-close-v3,faucet-open-v3,hammer-v3,hand-insert-v3,handle-press-side-v3,handle-press-v3,handle-pull-side-v3,handle-pull-v3,lever-pull-v3,peg-insert-side-v3,peg-unplug-side-v3,pick-out-of-hole-v3,pick-place-v3,pick-place-wall-v3,plate-slide-back-side-v3

Shard 2 — 16 tasks
------------------
plate-slide-back-v3,plate-slide-side-v3,plate-slide-v3,push-back-v3,push-v3,push-wall-v3,reach-v3,reach-wall-v3,shelf-place-v3,soccer-v3,stick-pull-v3,stick-push-v3,sweep-into-v3,sweep-v3,window-open-v3,window-close-v3

Slurm resources (per shard; edit in each .slurm if needed)
----------------------------------------------------------
  #SBATCH --gres=gpu:1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=64G
  #SBATCH --time=1-00:00:00

After all three jobs finish — merge eval_info.json then summarize
-------------------------------------------------------------------
  cd /vol/bitbucket/aa6622/project
  python scripts/mt50/merge_official_lerobot_eval_shards.py \
    --parent artifacts/MT50_Phase072_official_lerobot_10ep \
    --shards shard0 shard1 shard2 \
    --merged-eval-info artifacts/MT50_Phase072_official_lerobot_10ep/eval_info.json

  python scripts/mt50/summarize_official_lerobot_eval.py \
    --eval-info artifacts/MT50_Phase072_official_lerobot_10ep/eval_info.json \
    --output artifacts/MT50_Phase072_official_lerobot_10ep/MT50_Phase072_official_index.json \
    --run-root artifacts/MT50_Phase072_official_lerobot_10ep \
    --seed 1000 \
    --phase MT50_Phase072

Single-GPU alternative (no merge)
---------------------------------
  MT50_PHASE071_TASK=all MT50_PHASE071_EPISODES=10 \
  MT50_PHASE071_OUTPUT_ROOT=artifacts/MT50_Phase072_official_lerobot_10ep_single \
  bash scripts/mt50/run_official_lerobot_mt50_eval.sh

Notes
-----
- LeRobot eval_main hardcodes max_episodes_rendered=10, but the repo wrapper can route through
  scripts/mt50/lerobot_eval_configurable_rendering.py when MT50_LEROBOT_MAX_EPISODES_RENDERED is set.
- Set MT50_LEROBOT_MAX_EPISODES_RENDERED=0 for no videos. In that mode, progress is best checked from
  Slurm stdout/stderr; eval_info.json appears when that shard exits.
