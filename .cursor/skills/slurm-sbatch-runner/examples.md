# Examples: SmolVLA parity on Slurm

## Preflight

```bash
cd /vol/bitbucket/aa6622/project
bash -n scripts/slurm/common_env.sh
bash -n scripts/smolvla/submit_smolvla_parity_eval.slurm
sbatch --test-only --export=NIL scripts/smolvla/submit_smolvla_parity_eval.slurm \
  --task reach-v3 --episodes 1 --seed 1 --max-steps 10 --fps 30 \
  --overlay-mode reward_delta --save-frames false --camera-name corner2 --flip-corner2 true \
  --checkpoint jadechoghari/smolvla_metaworld
```

## Submit (clean env)

```bash
cd /vol/bitbucket/aa6622/project
sbatch -J smolvla-reach --export=NIL scripts/smolvla/submit_smolvla_parity_eval.slurm \
  --task reach-v3 --episodes 20 --seed 1000 --max-steps 120 --fps 30 \
  --overlay-mode reward_delta --save-frames true --camera-name corner2 --flip-corner2 true \
  --checkpoint jadechoghari/smolvla_metaworld
```

## Monitor

```bash
cd /vol/bitbucket/aa6622/project
python3 scripts/smolvla/monitor_smolvla_parity_jobs.py 236688 236689
```

## MT10-style: one job, many tasks

Submit `submit_phase8_mt10.slurm` only. Do not nest `sbatch` inside that allocation. For P6→P8→P9 chains, run `submit_mt10_p6_p8_p9_prefixed.sh` from the login node.
