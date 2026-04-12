# WM vs env chunk length (why 4D vs 20D)

Plain-language notes on how **simulator actions** relate to **world model `act_suffix`** and “chunk length.” For MetaWorld + JEPA-WM Metaworld hub.

---

## What you care about day-to-day

The policy and the simulator work in **small moves**: “this arm command, this gripper, …” — **one vector per time the sim actually steps**. For MetaWorld that vector has **4 numbers**. That is the usual mental model of “an action.”

## What the world model was trained on

The JEPA Metaworld checkpoint was not trained on “one small move → one brain tick.” It was trained on **chunks**: “here are **five** small moves in a row,” glued into **one** fat vector of **20 numbers** (5×4). So the model’s **one step forward in imagination** always consumes **that whole chunk**, not a single 4-number move.

## What `act_suffix` means (one sentence)

It is **the list of those fat vectors** you are asking the world model to “play out” after the current encoded image/state. Each row is **one chunk of five normalized 4D actions**, stored as **20 numbers**.

## Why `(T_wm, 1, 20)` shows up in code

- **20** — one chunk (five moves squashed together).
- **T_wm** — how many chunks you are rolling (e.g. eight sim steps → two chunks of five, with the last chunk possibly padded so it is still length five — a separate modeling choice).
- **1 in the middle** — batch size = one trajectory; common so stacks match what `unroll` expects.

## Batched vs iterative unroll

Same data, different scheduling:

- **Batched:** pass the **whole** sequence of chunks; advance the latent in one `unroll` call.
- **Iterative:** feed **chunk 1**, update latent, then **chunk 2**, and so on. Still **20 numbers per call**; just one chunk at a time.

## Memory hook

**Sim thinks in 4D steps; WM thinks in 20D megasteps (five 4D steps each); `act_suffix` is the sequence of megasteps.**

---

## Technical pointer (repo)

Implementation: [`segment_grpo_loop.py`](../../src/segment_grpo_loop.py) — `score_chunk_by_goal_latent` builds normalized env actions, **packs** them with `_pack_env_actions_for_wm`, then passes **`seq_actions`** shaped **`(T_wm, 1, wm_dim)`** as `act_suffix` to `model.unroll` (batched full tensor or iterative slices of shape `(1, 1, wm_dim)`).

**Factor `f`** is not a YAML `frameskip` in this repo; when `wm_dim` is divisible by env action dim, **`f = wm_dim // env_dim`** (Metaworld hub: 20/4 = 5).

For paper / upstream alignment, see the JEPA-WM paper appendix (Metaworld `action_dim` 20) and [`facebookresearch/jepa-wms`](https://github.com/facebookresearch/jepa-wms).

---

## Related plan

Semantic audit and code verification: Cursor plan `wm_action_semantics_78036ffe` (WM action semantics).
