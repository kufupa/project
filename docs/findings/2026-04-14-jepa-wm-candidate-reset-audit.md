# 2026-04-14 — JEPA-WM candidate reset audit

## Question under review
Is the segment GRPO scoring path reusing world-model state across candidate chunks, or does each candidate start from the current ground-truth environment state (`image`, `proprio`)?

## Findings summary
- No cross-candidate world-model state carryover exists in this repository’s scoring pipeline.
- Every candidate chunk scoring starts with a fresh `wm_bundle.model.encode(...)` call built from the same per-segment ground-truth state snapshot.
- Cross-step state continuation only happens inside one candidate’s iterative unroll (intended, within that candidate).
- Upstream JEPA-WM API used by this repo (`torch.hub` loading path) is explicit function-based and does not imply persistent internal state across calls.

## Evidence from repository code

### 1) Per-segment candidate loop reuses the same start observation for each candidate
In `rollout_with_chunks`, candidate scoring runs inside a `for candidate_idx in range(num_candidates)` loop and calls
`score_chunk_by_goal_latent(...)` with `current_image` and `current_proprio` as inputs.

Relevant code path:
- `project/src/segment_grpo_loop.py:2183` — candidate loop.
- `project/src/segment_grpo_loop.py:2204-2211` — every candidate passes `current_image`, `current_proprio` to scoring.

```python
for candidate_idx in range(num_candidates):
    ...
    distance, score_trace, decode_trace = score_chunk_by_goal_latent(
        wm_bundle,
        current_image,
        current_proprio,
        chunk,
        ...
    )
```

### 2) Fresh latent state creation per candidate via encode
`score_chunk_by_goal_latent` converts the passed image/proprio to WM tensors and calls:

- `project/src/segment_grpo_loop.py:1526-1530` — `_to_wm_visual`, `_to_wm_proprio`, then `wm_bundle.model.encode(...)`.

That encoded latent is the sole `latent_state` used for the candidate’s unroll.

```python
visual = _to_wm_visual(image, wm_bundle.device)
proprio_t = _to_wm_proprio(proprio, wm_bundle.proprio_dim, wm_bundle.device)
latent_state = _as_tensor_dict_if_available(wm_bundle.model.encode({"visual": visual, "proprio": proprio_t}))
```

### 3) Candidate-level unroll is rederived per candidate
- In batched mode: `unroll(latent_state, act_suffix=...)` is called directly on that fresh `latent_state` (`project/src/segment_grpo_loop.py:1545`).
- In iterative mode:
  - first unroll starts from `latent_state`.
  - then `_next_latent_state_after_unroll(...)` feeds the next input.
  - this loop is internal to the candidate and does not reference the next candidate’s state (`project/src/segment_grpo_loop.py:1591-1594`, `1361-1407`).

This means chaining is intra-candidate only (correct for iterative rollout), not cross-candidate.

### 4) State transition helpers are local/derived
`_next_latent_state_after_unroll`:
- converts the previous unroll output to the final timestep and returns only that slice.
- does not mutate shared globals or external buffers (`project/src/segment_grpo_loop.py:1361-1407`).

### 5) Upstream model contract does not indicate hidden internal continuation state
Repo loads WM from helper via `torch.hub` (`_try_load_wm`, `vendor/pi05/jepa_cem_paired_pushv3_export.py`) and passes explicit observations/latent inputs into:
- `model.encode(...)`
- `model.unroll(...)`

Upstream JEPA-WM source confirms these are explicit-input methods (`vit_enc_preds.py`) with no call-site stateful continuation:
- `facebookresearch/jepa-wms` `EncPredWM.unroll` uses input `z_ctxt` and `act_suffix`.
- `EncPredWM.encode` creates features from provided observation input.
- `VideoWM.encode`/`rollout` methods also operate from explicit inputs.

## Conclusion
From both call-path and upstream API structure, the current implementation does **not** continue candidate-to-candidate WM internal state.  
Each candidate is effectively evaluated from the same ground-truth segment-start state, then rolled forward internally only within that candidate.

## Residual risks
- If `model` implementation is replaced with a custom class that mutates internal hidden state inside `encode`/`unroll`, this could violate the assumption.
- Existing tests already confirm per-candidate encoding/packing behavior and selected decode-only-on-chosen-candidate behavior; adding an explicit “encode called once per candidate” test would make this invariant easier to preserve.
