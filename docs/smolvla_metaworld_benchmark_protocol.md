# SmolVLA Meta-World Benchmark Protocol Notes

Date: 2026-05-05

## Short Verdict

For reproducing the SmolVLA paper comparison, target the LeRobot/SmolVLA Meta-World MT50-style benchmark, not the strict original Farama RL MT50 evaluation loop.

The paper reports Meta-World results by difficulty bucket:

- Easy
- Medium
- Hard
- Very Hard
- Average

The SmolVLA 0.45B reported Meta-World scores are:

- Easy: 82.5
- Medium: 41.8
- Hard: 45.0
- Very Hard: 60.0
- Average: 57.3

## Protocol To Recreate

Use the LeRobot Meta-World evaluation framing:

- Dataset/checkpoint family: `lerobot/metaworld_mt50` / `HuggingFaceVLA/metaworld_mt50`
- Coverage: all 50 Meta-World tasks
- Evaluation grouping: easy, medium, hard, very_hard, and full MT50
- Recommended reproducible eval size: 10 episodes per task
- Full eval total: 50 tasks x 10 episodes = 500 episodes
- Horizon: 500 environment steps
- Success: episode succeeds if the environment success flag is true at any point in the episode
- SmolVLA action loop: feed new image/state/task prompt each step and execute one action per policy call

This matches the practical SmolVLA/LeRobot benchmark language and the reported paper table structure.

## Not Equivalent To Strict Farama MT50 RL Eval

Strict Farama Meta-World MT50 RL protocol is larger:

- `metaworld.MT50(seed=42)` creates 50 tasks.
- Each task has 50 train goal positions.
- Evaluation is one episode per goal position per task.
- Total: 50 tasks x 50 goals = 2500 episodes.
- Success is counted if `info["success"] == 1` at any point within a 500-step episode.

That is the original RL benchmark protocol. It is useful as a reference, but it is not the same as the SmolVLA paper-style 500-episode LeRobot evaluation.

## Current Implementation Implications

For our SmolVLA evaluator, paper-aligned changes should prioritize:

- Run 50 tasks x 10 episodes, not 1 episode per task.
- Use 500 max steps.
- Stop each rollout on `terminated or truncated or info["success"]`.
- Preserve success-any semantics in aggregate metrics.
- Keep `n_action_steps=1` / single action execution per fresh observation.
- Use the LeRobot Meta-World task list and difficulty grouping for reporting.
- Avoid claiming equivalence to strict Farama MT50 unless running all 2500 task-goal episodes.

## Sources

- SmolVLA paper: [SmolVLA: A vision-language-action model for affordable and efficient robotics](https://arxiv.org/pdf/2506.01844)
- LeRobot Meta-World docs: [Meta-World - Hugging Face LeRobot](https://huggingface.co/docs/lerobot/metaworld)
- Meta-World evaluation docs: [Evaluation - Meta-World Documentation](https://metaworld.farama.org/evaluation/evaluation/)
- Meta-World basic usage docs: [Basic Usage - Meta-World Documentation](https://metaworld.farama.org/introduction/basic_usage/)
- LeRobot dataset card mirror: [HuggingFaceVLA/metaworld_mt50](https://hugging-face.cn/datasets/HuggingFaceVLA/metaworld_mt50)
- SmolVLA/LeRobot discussion on one-action inference: [huggingface/lerobot issue 1316](https://github.com/huggingface/lerobot/issues/1316)
- Secondary extracted SmolVLA table values: [aimodels.fyi SmolVLA paper summary](https://aimodels.fyi/papers/arxiv/smolvla-vision-language-action-model-affordable-efficient)

