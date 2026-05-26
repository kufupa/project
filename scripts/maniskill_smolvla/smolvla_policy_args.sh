#!/usr/bin/env bash
# SmolVLA contract for RL4VLA ManiSkill PutOnPlateInScene25Main-v3.

MSM_SMOLVLA_POLICY_ARGS=(
  '--policy.input_features={"observation.state":{"type":"STATE","shape":[7]},"observation.images.front":{"type":"VISUAL","shape":[3,480,640]}}'
  '--policy.output_features={"action":{"type":"ACTION","shape":[7]}}'
  --policy.chunk_size=50
  --policy.n_action_steps=1
  --policy.max_state_dim=32
  --policy.max_action_dim=32
  '--policy.resize_imgs_with_padding=[512,512]'
  --policy.freeze_vision_encoder=true
  --policy.train_expert_only=true
  --policy.train_state_proj=true
  --policy.use_amp=true
)
