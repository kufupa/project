# PI0.5 ManiSkill SFT Checkpoint Status

## Answer

The installed checkpoint already has the ManiSkill SFT stage applied. We do not need to run the paper's 1000 supervised fine-tuning steps from scratch for the current EGGROLL pipeline.

## Evidence

- The checkpoint installed in this repo is `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`.
- The local checkpoint metadata file, `metadata.pt`, records `global_step = 1000`.
- The same metadata identifies the training setup as `name = pi05_maniskill` and `exp_name = PutOnPlateInScene25Main-v3`.
- The metadata also shows the checkpoint was initialized from a base `pi05_base` path, then trained for the ManiSkill setup.
- RLinf's official PPO config starts from this exact SFT checkpoint path:
  `RLinf-Pi05-ManiSkill-25Main-SFT`.
- Hugging Face separately publishes RL-tuned checkpoints:
  `RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE` and
  `RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise`.
  Those are distinct from the SFT checkpoint and correspond to the RL stage.

## Sources And Re-Check Instructions

Primary local source:

- Checkpoint directory:
  `/vol/bitbucket/aa6622/eggroll/.cache/huggingface/rlinf/RLinf-Pi05-ManiSkill-25Main-SFT/`
- Manifest:
  `/vol/bitbucket/aa6622/eggroll/rlinf_pi05_maniskill_sft_manifest.json`
- Metadata file:
  `/vol/bitbucket/aa6622/eggroll/.cache/huggingface/rlinf/RLinf-Pi05-ManiSkill-25Main-SFT/metadata.pt`

To inspect the metadata:

```bash
cd /vol/bitbucket/aa6622/eggroll
.venv/bin/python - <<'PY'
from pathlib import Path
import torch

root = Path(".cache/huggingface/rlinf/RLinf-Pi05-ManiSkill-25Main-SFT")
metadata = torch.load(root / "metadata.pt", map_location="cpu", weights_only=False)

print("global_step:", metadata["global_step"])
config = metadata["config"]
print("name:", config["name"])
print("exp_name:", config["exp_name"])
print("weight_loader:", config["weight_loader"])
print("pytorch_weight_path:", config["pytorch_weight_path"])
print("model.openpi-like fields:", {
    "pi05": config["model"]["pi05"],
    "action_env_dim": config["model"]["action_env_dim"],
    "num_steps": config["model"]["num_steps"],
    "action_chunk": config["model"]["action_chunk"],
})
PY
```

Expected key output:

```text
global_step: 1000
name: pi05_maniskill
exp_name: PutOnPlateInScene25Main-v3
weight_loader: {'params_path': 'checkpoints/jax/pi05_base'}
```

RLinf config source:

- PPO config:
  `/vol/bitbucket/aa6622/eggroll/third_party/RLinf/examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`
- Relevant fields:
  `rollout.model.model_path` and `actor.model.model_path` both point to
  `/path/to/model/RLinf-Pi05-ManiSkill-25Main-SFT`.
- This means the official RLinf PPO path treats the SFT checkpoint as the starting model for RL.

Hugging Face sources:

- SFT checkpoint API:
  `https://huggingface.co/api/models/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`
- RL Flow-SDE checkpoint API:
  `https://huggingface.co/api/models/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE`
- RL Flow-Noise checkpoint API:
  `https://huggingface.co/api/models/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise`

Important distinction:

- `RLinf-Pi05-ManiSkill-25Main-SFT` is the released SFT baseline checkpoint.
- `RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE` and
  `RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise` are separately released RL-stage checkpoints.
- The RL checkpoint APIs expose actor artifacts under `global_step_150`, matching the paper's RL stage rather than the SFT stage.

## Practical Interpretation

For our work, `RLinf-Pi05-ManiSkill-25Main-SFT` is the correct warm-start checkpoint: post-SFT, pre-RL. The paper's `SFT train steps: 1000` describes how the authors produced this baseline checkpoint from `pi05_base`.

If we wanted to reproduce the full paper pipeline from the raw base model, we would need to run those 1000 SFT steps. For the current objective, where we start from the released RLinf SFT checkpoint and retarget EGGROLL, that SFT stage is already done.

Confidence: high. The Hugging Face model card is sparse, but the local checkpoint metadata directly records the 1000-step SFT state.
