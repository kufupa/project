# Phase 50 EGGROLL Algorithm, Batching, and Parallelism Notes

## Scope

This note records the answers to two questions:

- What EGGROLL is doing, in ELI15 terms.
- How batching/parallelism works when each population member has different SmolVLA weight perturbations, plus measured slowdown versus Phase11.

## EGGROLL, ELI15

- EGGROLL is gradient-free fine-tuning for SmolVLA.
- It does not backpropagate through the model.
- It samples a population of low-rank random perturbations for selected SmolVLA `nn.Linear` layers.
- Each population member is the base policy plus one perturbation.
- Each member rolls out in real MetaWorld `push-v3`.
- Fitness is dense environment return, with success tracked separately.
- Fitness values are rank-normalized, so huge reward outliers do not dominate the update too much.
- The model weights move toward perturbations that scored well and away from perturbations that scored poorly.
- Checkpoints are saved every 2 updates in the overnight/next-run setup.

Pseudocode:

```python
for update in range(num_updates):
    seed_batch = [seed(update, k) for k in range(episodes_per_member)]

    member_scores = []
    for seed in seed_batch:
        for members in chunks(population, population_batch_size):
            rollouts = run_metaworld(
                members=members,
                reset_seed=seed,  # same seed for all members in this comparison
            )
            member_scores += dense_rewards(rollouts)

    fitness = average_scores_across_seed_batch(member_scores)
    shaped = rank_normalize(fitness)
    weights += alpha * weighted_sum(low_rank_perturbations, shaped)

    if update % 2 == 0:
        save_checkpoint()
```

## Batching We Use

There are two batching concepts:

- `population_batch_size`: how many perturbed population members run together in one forward/env wave.
- `episodes_per_member`: how many reset seeds each population member is evaluated on before averaging fitness.

Current jobs:

- Run C uses `population_batch_size=16`, `episodes_per_member=3`.
- Run D uses `population_batch_size=16`, `episodes_per_member=2`.

Fair seed mode:

- We use `seed_mode=shared_per_iteration`.
- Within an update and episode repeat, every population member sees the same MetaWorld reset seed.
- This makes the evolutionary comparison fair: better fitness should come from the perturbation, not from a luckier/easier initial state.

## How Batched Forward Works With Different Perturbed Weights

Naive EGGROLL would require one full SmolVLA copy per population member:

```python
for member in population:
    policy = copy_model(base_policy)
    policy.weights += perturbation[member]
    action = policy(obs)
```

That would be too slow and memory-heavy.

Our implementation keeps one SmolVLA model and patches selected `nn.Linear` layers. During one batched forward, each row/member receives a row-specific low-rank delta:

```python
y = x @ W.T + sigma * (x @ B_member @ A_member.T)
```

So the batch still flows through one `sample_actions()` call, but each row behaves like it has different perturbed weights.

Important files:

- `src/smolvla_grpo/eggroll_linear.py`: row-specific low-rank `nn.Linear` perturbation patch.
- `src/smolvla_grpo/eggroll_noise.py`: deterministic low-rank factor generation.
- `src/smolvla_grpo/eggroll_rollout.py`: population wave batching, vector env stepping, and fair reset seeds.
- `src/smolvla_grpo/eggroll_trainer.py`: fitness averaging, rank shaping, ES update, checkpointing.

## Parallelism Layers

- GPU/model batching:
  - `population_batch_size=16` means 16 active perturbed members can be passed through SmolVLA together.

- Environment batching:
  - `vector_async` uses vectorized MetaWorld envs and steps the wave together with `env.step_batch(...)`.

- Action chunking:
  - `action_chunk_size=5` means one model forward produces up to 5 environment actions.
  - This reduces model calls versus one forward per env step.

- Seed batching:
  - `episodes_per_member=2` or `3` repeats the population evaluation on multiple reset seeds.
  - This improves ranking robustness but multiplies rollout cost.

## Measured Slowdown Versus Phase11

Observed Phase11 reference:

- Phase11 `g32`, chunk5, real MetaWorld:
  - Around `231s/update`.
  - Includes rollout plus GRPO optimization/logprob recomputation.

Observed EGGROLL:

- Run A, `pop32`, `episodes_per_member=2`, full `max_steps=120`:
  - Mean update time around `574s/update`.
  - This is about `2.5x` slower per optimizer update than Phase11 `g32`.

- Run B, `pop64`, `episodes_per_member=2`, full `max_steps=120`:
  - Mean update time around `1135s/update`.
  - This is about `4.9x` slower per optimizer update than Phase11 `g32`, and roughly `2x` Run A.

Important interpretation:

- EGGROLL Run A does about `64` rollouts per update (`pop32 * seedbatch2`).
- Phase11 `g32` does about `32` rollouts per update.
- So EGGROLL Run A does roughly `2x` as many rollouts and costs `2.5x` walltime.
- Per rollout/env-step, the extra overhead is closer to `1.2x`, not `2.5x`.

## Why EGGROLL Is Slower

- Row-specific low-rank perturbations add work inside many patched Linear layers.
- Population members run in waves of 16, not all at once.
- Seed batching repeats rollout collection for fairer fitness estimates.
- Environment reset and step overhead is repeated for each seed batch.
- EGGROLL spends almost no time on the ES update itself; rollout dominates.

Measured examples from prior runs:

- Run A:
  - Total training reported time: `5:49:28.71`.
  - Mean iteration: `0:09:34`.
  - Rollout sum: `5:35:19`.
  - ES update time was tiny relative to rollout.

- Run B:
  - Total training reported time: `6:23:17.54`.
  - Mean iteration: `0:18:54`.
  - Rollout sum: `6:14:40`.
  - Runtime roughly doubled because population doubled from 32 to 64.

## Bottom Line

- Yes, EGGROLL still batches SmolVLA forward inference.
- It does this by applying row-specific low-rank deltas inside patched Linear layers.
- It is slower than Phase11 because it needs many policy perturbation rollouts per update, especially with fair seed batches.
- The batching keeps this practical: without row-wise perturbation batching, EGGROLL would require separate model forwards/copies per population member and would be much worse.
