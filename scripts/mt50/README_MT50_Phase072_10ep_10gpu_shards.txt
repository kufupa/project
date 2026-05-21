MT50 Phase072 — official LeRobot eval, 50 tasks × 10 episodes, 10 parallel GPU jobs (5 tasks each)
====================================================================================================

What this is
------------
- Same checkpoint and wrapper as Phase071: scripts/mt50/run_official_lerobot_mt50_eval.sh
- Ten jobs run at once; each job uses one GPU and a disjoint comma-separated --env.task list
  (5 tasks per job). Task order matches README_MT50_Phase072_10ep_shards.txt / train_config.json.
- Episodes per task: MT50_PHASE071_EPISODES=10 (default in each Slurm file).
- Outputs: artifacts/MT50_Phase072_official_lerobot_10ep_10gpu/shard0 … shard9/

Comparison to the 3-shard layout
---------------------------------
- 3-shard: 17 + 17 + 16 tasks per GPU (submit_mt50_phase072_10ep_shard{0,1,2}.slurm).
- 10-GPU: 5 tasks per GPU, shorter per-job wall time, needs 10 free GPUs at submit time.

Why task lists are embedded in Slurm (do not use sbatch --export for task lists)
--------------------------------------------------------------------------------
Slurm splits --export on commas. A comma-separated task list in --export breaks.
Task lists are single-quoted inside: submit_mt50_phase072_10ep_10gpu_shard{0..9}.slurm

Queue all ten jobs
------------------
  cd /vol/bitbucket/aa6622/project
  bash scripts/mt50/submit_mt50_phase072_10ep_all_10gpu.sh

Or individually:
  sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase072_10ep_10gpu_shard0.slurm
  … through …_shard9.slurm

Shard 0 — 5 tasks
-------------------
assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3

Shard 1 — 5 tasks
-----------------
button-press-topdown-wall-v3,button-press-v3,button-press-wall-v3,coffee-button-v3,coffee-pull-v3

Shard 2 — 5 tasks
-----------------
coffee-push-v3,dial-turn-v3,disassemble-v3,door-close-v3,door-lock-v3

Shard 3 — 5 tasks
-----------------
door-open-v3,door-unlock-v3,drawer-close-v3,drawer-open-v3,faucet-close-v3

Shard 4 — 5 tasks
-----------------
faucet-open-v3,hammer-v3,hand-insert-v3,handle-press-side-v3,handle-press-v3

Shard 5 — 5 tasks
-----------------
handle-pull-side-v3,handle-pull-v3,lever-pull-v3,peg-insert-side-v3,peg-unplug-side-v3

Shard 6 — 5 tasks
-----------------
pick-out-of-hole-v3,pick-place-v3,pick-place-wall-v3,plate-slide-back-side-v3,plate-slide-back-v3

Shard 7 — 5 tasks
-----------------
plate-slide-side-v3,plate-slide-v3,push-back-v3,push-v3,push-wall-v3

Shard 8 — 5 tasks
-----------------
reach-v3,reach-wall-v3,shelf-place-v3,soccer-v3,stick-pull-v3

Shard 9 — 5 tasks
-----------------
stick-push-v3,sweep-into-v3,sweep-v3,window-open-v3,window-close-v3

Slurm resources (per job; edit in each .slurm if needed)
---------------------------------------------------------
  #SBATCH --gres=gpu:1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=64G
  #SBATCH --time=1-00:00:00

PBS resources (committed CX3 scripts; no #PBS -q — scheduler routes GPU chunks)
--------------------------------------------------------------------------------
  #PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
  #PBS -l walltime=24:00:00
  Jobs often appear under queue v1_gpu72 because of ngpus=1, not because we pin -q.

After all ten jobs finish — merge eval_info.json then summarize
-----------------------------------------------------------------
  cd /vol/bitbucket/aa6622/project
  python scripts/mt50/merge_official_lerobot_eval_shards.py \
    --parent artifacts/MT50_Phase072_official_lerobot_10ep_10gpu \
    --shards shard0 shard1 shard2 shard3 shard4 shard5 shard6 shard7 shard8 shard9 \
    --merged-eval-info artifacts/MT50_Phase072_official_lerobot_10ep_10gpu/eval_info.json

  python scripts/mt50/summarize_official_lerobot_eval.py \
    --eval-info artifacts/MT50_Phase072_official_lerobot_10ep_10gpu/eval_info.json \
    --output artifacts/MT50_Phase072_official_lerobot_10ep_10gpu/MT50_Phase072_official_index.json \
    --run-root artifacts/MT50_Phase072_official_lerobot_10ep_10gpu \
    --seed 1000 \
    --phase MT50_Phase072

PBS / OpenPBS (no sbatch on login node)
---------------------------------------
Same bash body; replace Slurm directives with PBS. GPU resource names are site-specific.

Example directive mapping (adjust select/ngpus/queue for your site):
  #SBATCH --job-name=mt50-p072-p10-s0     ->  #PBS -N mt50-p072-p10-s0
  #SBATCH --gres=gpu:1                    ->  #PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
  #SBATCH --cpus-per-task=8               ->  (fold into select ncpus=)
  #SBATCH --mem=64G                       ->  (fold into select mem=)
  #SBATCH --time=1-00:00:00               ->  #PBS -l walltime=24:00:00
  #SBATCH --output=... / --error=...      ->  #PBS -j oe  and  #PBS -o mt50_phase072_10ep_10gpu_shard0.out

After the PBS headers, use PBS_O_WORKDIR for the repo root (PBS copies the script to spool):

  if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    PROJECT_ROOT="$(cd "${PBS_O_WORKDIR}" && pwd)"
    cd "${PROJECT_ROOT}"
  fi

Then keep the same exports and: bash scripts/mt50/run_official_lerobot_mt50_eval.sh

Single-GPU alternative (no merge)
---------------------------------
  MT50_PHASE071_TASK=all MT50_PHASE071_EPISODES=10 \
  MT50_PHASE071_OUTPUT_ROOT=artifacts/MT50_Phase072_official_lerobot_10ep_single \
  bash scripts/mt50/run_official_lerobot_mt50_eval.sh

Notes
-----
- LeRobot eval_main uses max_episodes_rendered=10; with n_episodes=10 you get up to 10 MP4s per task.
- Progress while running: count *.mp4 under each shard’s videos/; eval_info.json appears when that shard exits.
