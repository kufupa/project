# SmolVLA MetaWorld Resolution + Checkpoint Findings

## Main Conclusions

- Env/input resolution: `480x480`.
  - LeRobot MetaWorld env emits `pixels` as `480x480x3` uint8.
  - LeRobot preprocessing converts to `observation.image` as `B x 3 x 480 x 480`, float32 in `[0, 1]`.
- SmolVLA internal vision resolution: `512x512`.
  - Checkpoint config sets `resize_imgs_with_padding: [512, 512]`.
  - SmolVLA `prepare_images()` resizes/pads image tensor to `512x512` before vision backbone.
- Correct eval choice: keep MetaWorld env at `480x480`.
  - Training dataset `lerobot/metaworld_mt50` stores images as `3 x 480 x 480`.
  - Training path was `480x480 dataset image -> SmolVLA resize/pad -> 512x512 model image`.
  - Requesting `512` from MetaWorld would create different input distribution: `512x512 env render -> resize/pad 512x512`.
- Checkpoint used in Phase27:
  - Repo: `jadechoghari/smolvla_metaworld`.
  - Revision: `ef3089ecb84eeeb7d33fedab24f6c76180a68900`.
  - Dataset: `lerobot/metaworld_mt50`.
  - Base model: `lerobot/smolvla_base`.
  - VLM/tokenizer: `HuggingFaceTB/SmolVLM2-500M-Instruct`.
- Confidence: high, 95%+.
  - Exact runtime path traced through project scripts, LeRobot env config, observation preprocessing, checkpoint preprocessor/config, SmolVLA model code, and Hugging Face metadata.

## What We Did

- Phase27 eval wrapper calls official LeRobot eval with local checkpoint.
  - Source: `scripts/mt50/run_official_lerobot_mt50_eval.sh`.
  - Key args:
    - `--policy.path="${SNAPSHOT}"`
    - `--env.type=metaworld`
    - `--env.task="${TASK}"`
    - `--eval.n_episodes="${EPISODES}"`
    - `--eval.batch_size=1`
    - `--eval.use_async_envs=false`
    - `--seed="${SEED}"`
    - `--output_dir="${OUTPUT_ROOT}"`
  - No explicit image size override passed.
  - Default checkpoint path:
    - `${WORKSPACE_ROOT}/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900`
- Phase27 eval therefore relies on LeRobot MetaWorld default image size and checkpoint preprocessing.
  - LeRobot MetaWorld default image size: `480x480`.
  - Checkpoint image feature: `3 x 480 x 480`.
  - SmolVLA internal resize/pad target: `512 x 512`.

## Runtime Image Path

- Step 1: MetaWorld env creates pixel obs.
  - Source: `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/envs/metaworld.py`.
  - Constructor defaults:
    - `camera_name="corner2"`
    - `obs_type="pixels"`
    - `render_mode="rgb_array"`
    - `observation_width=480`
    - `observation_height=480`
  - Observation space for `pixels_agent_pos`:
    - `pixels`: shape `(observation_height, observation_width, 3)`.
    - With defaults: `(480, 480, 3)`.
    - dtype: `np.uint8`.
  - `_format_raw_obs()` calls `self._env.render()`, flips `corner2` camera image, returns:
    - `{"pixels": image.copy(), "agent_pos": agent_pos}` for `pixels_agent_pos`.

- Step 2: LeRobot env config declares MetaWorld features.
  - Source: `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/envs/configs.py`.
  - `MetaworldEnv` config:
    - `fps = 80`.
    - `episode_length = 400`.
    - `obs_type = "pixels_agent_pos"`.
    - `render_mode = "rgb_array"`.
  - For `pixels_agent_pos`, features are:
    - `agent_pos`: state shape `(4,)`.
    - `pixels/top`: visual shape `(480, 480, 3)`.
    - `action`: action shape `(4,)`.
  - `features_map` maps:
    - `agent_pos` -> `observation.state`.
    - `pixels/top` -> `observation.image`.

- Step 3: Observation preprocessing converts HWC uint8 to BCHW float.
  - Source: `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/envs/utils.py`.
  - `preprocess_observation()` behavior:
    - If key `pixels` exists, maps it to `observation.image`.
    - Converts numpy image to torch tensor.
    - Adds batch dim if image is 3D.
    - Checks channel-last layout: `b h w c`.
    - Checks dtype is `torch.uint8`.
    - Rearranges `b h w c -> b c h w`.
    - Converts to `float32`.
    - Divides by `255`.
  - Result:
    - Before preprocessing: `B x 480 x 480 x 3`, uint8.
    - After preprocessing: `B x 3 x 480 x 480`, float32 in `[0, 1]`.

- Step 4: Checkpoint preprocessor expects `3 x 480 x 480`.
  - Source: local HF cache blob:
    - `.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/blobs/16e91d61c969ab3b935bcaeb23fdcee633ba7788`
  - This is `policy_preprocessor.json`.
  - `normalizer_processor` features:
    - `observation.state`: shape `[4]`.
    - `observation.environment_state`: shape `[39]`.
    - `observation.image`: shape `[3, 480, 480]`.
    - `action`: shape `[4]`.
  - Visual normalization:
    - `VISUAL: IDENTITY`.
  - Meaning:
    - Preprocessor expects `480x480` channel-first image.
    - It does not resize visual features.

- Step 5: SmolVLA model resizes/pads internally to `512x512`.
  - Source: local HF cache blob:
    - `.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/blobs/32bff75f9785632dd46e93257688314864f340eb`
  - This is `config.json`.
  - Relevant config:
    - `input_features.observation.image.shape = [3, 480, 480]`.
    - `resize_imgs_with_padding = [512, 512]`.
  - Source: `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/policies/smolvla/configuration_smolvla.py`.
    - Default `resize_imgs_with_padding: tuple[int, int] = (512, 512)`.
  - Source: `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`.
  - `prepare_images()`:
    - Reads present image keys from batch.
    - Uses final obs image if sequence dim exists.
    - If `self.config.resize_imgs_with_padding is not None`, calls:
      - `resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)`.
    - Then normalizes image from `[0, 1]` to `[-1, 1]`.
  - Result:
    - Input to SmolVLA policy batch: `B x 3 x 480 x 480`.
    - Input after SmolVLA image prep: `B x 3 x 512 x 512`.
    - Vision backbone sees `512x512`.

## Training Dataset Image Size

- Dataset: `lerobot/metaworld_mt50`.
  - Source: Hugging Face dataset page:
    - `https://huggingface.co/datasets/lerobot/metaworld_mt50`
  - Dataset metadata:
    - `total_episodes`: `2500`.
    - `total_frames`: `204806`.
    - `fps`: `80`.
    - `observation.image`:
      - dtype: `image`.
      - shape: `[3, 480, 480]`.
    - `observation.state`:
      - dtype: `float32`.
      - shape: `[4]`.
    - `observation.environment_state`:
      - dtype: `float32`.
      - shape: `[39]`.
    - `action`:
      - dtype: `float32`.
      - shape: `[4]`.
  - Note:
    - Dataset page reports `total_tasks: 49`.
    - Train config task string contains 50 task names.
    - Keep this as source discrepancy, not blocker for resolution answer.

## Checkpoint Training Provenance

- Checkpoint repo:
  - `https://huggingface.co/jadechoghari/smolvla_metaworld`
- Revision used locally:
  - `ef3089ecb84eeeb7d33fedab24f6c76180a68900`.
  - Source: local cache `refs/main`.
- Local cache status:
  - Local snapshot contains model/config/preprocessor blobs.
  - Local snapshot does not contain full `README.md` or `train_config.json` in visible snapshot directory.
  - Therefore:
    - Runtime config and shape proof comes from local files.
    - Dataset/base-model/training recipe proof comes from live Hugging Face files for same repo/revision.
- Model card says:
  - `base_model: lerobot/smolvla_base`.
  - `datasets: lerobot/metaworld_mt50`.
  - `library_name: lerobot`.
  - `license: apache-2.0`.
  - `model_name: smolvla`.
  - `pipeline_tag: robotics`.
  - Source:
    - `https://huggingface.co/jadechoghari/smolvla_metaworld/resolve/main/README.md`
- Live `train_config.json` says:
  - Dataset:
    - `repo_id: lerobot/metaworld_mt50`.
    - `use_imagenet_stats: true`.
    - `video_backend: torchcodec`.
    - `streaming: false`.
    - image transforms configured but disabled: `image_transforms.enable: false`.
  - Env:
    - `type: metaworld`.
    - `fps: 80`.
    - `episode_length: 400`.
    - `obs_type: pixels_agent_pos`.
    - `render_mode: rgb_array`.
    - `multitask_eval: true`.
    - `max_parallel_tasks: 1`.
    - features:
      - `action`: `[4]`.
      - `agent_pos`: `[4]`.
      - `pixels/top`: `[480, 480, 3]`.
    - feature map:
      - `action` -> `action`.
      - `agent_pos` -> `observation.state`.
      - `top` -> `observation.image`.
      - `pixels/top` -> `observation.image`.
  - Policy:
    - `type: smolvla`.
    - `n_obs_steps: 1`.
    - `observation.state`: `[4]`.
    - `observation.environment_state`: `[39]`.
    - `observation.image`: `[3, 480, 480]`.
    - `action`: `[4]`.
    - `chunk_size: 50`.
    - `n_action_steps: 1`.
    - `resize_imgs_with_padding: [512, 512]`.
    - `tokenizer_max_length: 48`.
    - `num_steps: 10`.
    - `vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Instruct`.
    - `load_vlm_weights: true`.
    - `freeze_vision_encoder: true`.
    - `train_expert_only: true`.
    - `train_state_proj: true`.
  - Training:
    - `job_name: metaworld_smolvla`.
    - `seed: 1000`.
    - `num_workers: 4`.
    - `batch_size: 32`.
    - `steps: 100000`.
    - `eval_freq: 20000`.
    - `save_freq: 20000`.
    - optimizer: `adamw`.
    - learning rate: `0.0001`.
    - weight decay: `1e-10`.
    - grad clip: `10`.
    - betas: `[0.9, 0.95]`.
    - scheduler: `cosine_decay_with_warmup`.
    - warmup steps: `1000`.
    - decay steps: `30000`.
    - decay lr: `2.5e-06`.
  - Source:
    - `https://huggingface.co/jadechoghari/smolvla_metaworld/resolve/main/train_config.json`

## Why Not Request 512 From MetaWorld?

- Training dataset image size is `480x480`.
  - Dataset itself stores `observation.image` as `[3, 480, 480]`.
  - Policy config expects `observation.image` as `[3, 480, 480]`.
- SmolVLA resize target is not env render target.
  - `resize_imgs_with_padding=[512,512]` is model-side preprocessing.
  - It is applied after LeRobot observation preprocessing.
- If eval requests `512` from MetaWorld:
  - Env output changes from training distribution.
  - Image content changes because renderer produces different pixel grid, not same stored dataset image.
  - SmolVLA still applies resize/pad path, but input distribution no longer matches checkpoint config/data metadata.
- Best match:
  - Keep LeRobot env at `480x480`.
  - Let SmolVLA perform same internal `512x512` resize/pad as config says.

## What Matched Training

- MetaWorld family matched.
  - Eval uses `--env.type=metaworld`.
  - Checkpoint trained on `lerobot/metaworld_mt50`.
- Image size at env/dataset boundary matched.
  - Eval env image: `480x480`.
  - Dataset image: `480x480`.
- SmolVLA image preprocessing matched.
  - Eval checkpoint config: `resize_imgs_with_padding=[512,512]`.
  - Training config: `resize_imgs_with_padding=[512,512]`.
- Task text path matched LeRobot MetaWorld config.
  - Task descriptions come from installed `lerobot/envs/metaworld_config.json`.
  - Example:
    - `basketball-v3`: `Dunk the basketball into the basket`.

## Remaining Mismatch Risks

- Runtime env vs recorded demonstrations.
  - Training used recorded `lerobot/metaworld_mt50` dataset frames/actions.
  - Eval uses live LeRobot MetaWorld env.
  - Even with same nominal task/image size, reset states/render stack/package versions can differ.
- Camera/render stack.
  - LeRobot MetaWorld env uses `camera_name="corner2"` by default.
  - It flips `corner2` image with `np.flip(image, (0, 1))`.
  - Training dataset likely reflects same LeRobot pipeline if produced by matching LeRobot config, but exact package revision matters.
- Local cache incomplete.
  - Local snapshot visible in cache proves policy config and preprocessors.
  - Live HF proves model card and train config.
  - If HF main changes later, pin by SHA or download `train_config.json` for exact revision.
- Dataset metadata discrepancy.
  - Dataset page says `total_tasks: 49`.
  - Checkpoint train config task string lists 50 MetaWorld task names.
  - This does not affect image-resolution conclusion.

## Source List

- Project eval wrapper:
  - `scripts/mt50/run_official_lerobot_mt50_eval.sh`
    - Sets checkpoint path.
    - Sets offline/local HF flags.
    - Calls official LeRobot eval.
    - Does not set image resolution.
- Phase27 PBS wrappers:
  - `scripts/mt50/submit_mt50_phase27_smolvla_baseline_25ep_s1000_4gpu_rtx6000.pbs`
  - `scripts/mt50/submit_mt50_phase27_smolvla_baseline_recovery_25ep_s1000_3gpu_rtx6000.pbs`
    - Set task/episodes/seed/output root.
    - Call `run_official_lerobot_mt50_eval.sh`.
- Installed LeRobot env code:
  - `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/envs/metaworld.py`
    - Default `observation_width=480`, `observation_height=480`.
    - Pixel obs shape `(480,480,3)`.
    - `corner2` image flip.
  - `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/envs/configs.py`
    - `MetaworldEnv` features and feature map.
    - Visual feature shape `(480,480,3)`.
  - `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/envs/utils.py`
    - Converts `pixels` HWC uint8 to `observation.image` BCHW float `[0,1]`.
- Installed LeRobot SmolVLA code:
  - `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/policies/smolvla/configuration_smolvla.py`
    - Default `resize_imgs_with_padding=(512,512)`.
  - `.envs/lerobot_mw_py312/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`
    - `prepare_images()` calls `resize_with_pad(..., 512, 512)`.
    - Normalizes image `[0,1] -> [-1,1]`.
- Local checkpoint cache:
  - `.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/refs/main`
    - Revision: `ef3089ecb84eeeb7d33fedab24f6c76180a68900`.
  - `.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/blobs/32bff75f9785632dd46e93257688314864f340eb`
    - `config.json`.
    - `observation.image: [3,480,480]`.
    - `resize_imgs_with_padding: [512,512]`.
    - VLM: `HuggingFaceTB/SmolVLM2-500M-Instruct`.
  - `.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/blobs/16e91d61c969ab3b935bcaeb23fdcee633ba7788`
    - `policy_preprocessor.json`.
    - `observation.image: [3,480,480]`.
    - `VISUAL: IDENTITY`.
- Hugging Face live sources:
  - `https://huggingface.co/jadechoghari/smolvla_metaworld`
  - `https://huggingface.co/jadechoghari/smolvla_metaworld/tree/main`
  - `https://huggingface.co/jadechoghari/smolvla_metaworld/resolve/main/README.md`
  - `https://huggingface.co/jadechoghari/smolvla_metaworld/resolve/main/train_config.json`
  - `https://huggingface.co/datasets/lerobot/metaworld_mt50`

## Bottom Line

- Do not request `512x512` from MetaWorld.
- Use LeRobot MetaWorld default `480x480`.
- Let SmolVLA checkpoint config resize/pad internally to `512x512`.
- Current Phase27 eval path did this.
- Checkpoint was trained on `lerobot/metaworld_mt50`, whose images are `3 x 480 x 480`.
- Any big mismatch risk is likely live-env/reset/render distribution, not image resolution.
