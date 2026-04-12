# JEPA decoder contract and trace alignment — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align segment-GRPO decode and scoring inputs with upstream `EncPredWM.decode_unroll` (torch hub `facebookresearch/jepa-wms`), fix fragile visual-only dict calls, make iterative decode traces one step per action, and make scoring modality explicit (visual vs proprio vs fused).

**Architecture:** Keep `DecodeTrace` / `ScoreTrace` split. Adjust [`_decode_selected_trace`](file:///vol/bitbucket/aa6622/project/src/segment_grpo_loop.py) so visual-only uses the **tensor** branch of upstream decode. Refactor trace appends in [`score_chunk_by_goal_latent`](file:///vol/bitbucket/aa6622/project/src/segment_grpo_loop.py) to append **one** latent slice per outer action for iterative mode. Optionally add a small helper to extract “last predicted timestep” from `[T, B, V, H, W, D]` tensors. Tighten tests to use a **JEPA-like** stub that requires dicts to include both keys or accepts bare visual tensors.

**Tech stack:** Python 3, PyTorch, numpy, existing `segment_grpo_loop` tests under `python -m pytest`.

**External check:** Upstream `decode_unroll` in hub `vit_enc_preds.py` indexes `predicted_encs["visual"]` and `predicted_encs["proprio"]` for dict inputs; `proprio` is assigned but only **visual** feeds `image_head.decode` today. Visual-only API is **`decode_unroll(tensor, batch=...)`**. Paper ([arXiv:2512.24497](https://arxiv.org/abs/2512.24497)) discusses separate visual and proprio terms in the planning objective — product choice whether flat L2 in code should track **visual**, **proprio**, or a defined fusion.

---

## File map

| File | Role |
|------|------|
| [`/vol/bitbucket/aa6622/project/src/segment_grpo_loop.py`](/vol/bitbucket/aa6622/project/src/segment_grpo_loop.py) | `_decode_modal`, `_append_trace_steps` / iterative trace, `_extract_latent`, `_encode_state_to_latent`, `score_chunk_by_goal_latent` |
| [`/vol/bitbucket/aa6622/project/tests/test_segment_grpo_loop.py`](/vol/bitbucket/aa6622/project/tests/test_segment_grpo_loop.py) | Decoder contract tests; rollout decode tests |
| [`/vol/bitbucket/aa6622/project/pilot_wm_probe.py`](/vol/bitbucket/aa6622/project/pilot_wm_probe.py) | Any manual probe calling `score_chunk_by_goal_latent` / decode (only if signatures change) |

---

### Task 1: JEPA-like stub + failing test for visual-only dict bug

**Files:**
- Modify: `/vol/bitbucket/aa6622/project/tests/test_segment_grpo_loop.py`

- [ ] **Step 1: Add stub class matching hub `decode_unroll` dict behavior**

```python
class _JepaLikeDecodeUnroll:
    """Mimics EncPredWM.decode_unroll: dict must have both keys; tensor = visual only."""

    def __init__(self) -> None:
        self.calls: list[tuple[type, object]] = []

    def decode_unroll(self, predicted_encs, batch: bool = False):
        import torch
        import numpy as np

        if isinstance(predicted_encs, dict):
            _ = predicted_encs["visual"]
            _ = predicted_encs["proprio"]  # KeyError if missing — matches hub
            self.calls.append((dict, set(predicted_encs.keys())))
            return np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
        if isinstance(predicted_encs, torch.Tensor):
            self.calls.append((torch.Tensor, predicted_encs.shape))
            return np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
        raise TypeError(predicted_encs)
```

- [ ] **Step 2: Add test that current visual-only path triggers KeyError or wrong call**

```python
def test_decode_visual_only_must_not_use_single_key_dict() -> None:
    torch = pytest.importorskip("torch")
    dec = _JepaLikeDecodeUnroll()
    bundle = WMBundle(
        model=dec,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    # Only visual list populated; proprio empty -> today hits _decode_modal("visual", tensor)
    # which wraps {"visual": t} -> KeyError 'proprio' on stub
    trace = DecodeTrace(
        visual_latents=[np.zeros((1, 1, 1, 2, 2, 4), dtype=np.float32)],
        proprio_latents=[],
    )
    frames, failure = _decode_latent_trace_to_frames(bundle, trace)
    # Before fix: failure non-None containing KeyError / proprio
    # After fix: failure is None, and stub saw Tensor not dict with only visual
    assert failure is None
    assert len(frames) == 1
    assert dec.calls and dec.calls[0][0] is torch.Tensor
```

- [ ] **Step 3: Run test — expect FAIL before implementation**

Run:

```bash
cd /vol/bitbucket/aa6622/project && python -m pytest tests/test_segment_grpo_loop.py::test_decode_visual_only_must_not_use_single_key_dict -v
```

Expected: FAIL (KeyError or assertion on call type).

- [ ] **Step 4: Commit test (optional checkpoint)**

```bash
git add tests/test_segment_grpo_loop.py
git commit -m "test: expect tensor path for JEPA visual-only decode_unroll"
```

---

### Task 2: Fix `_decode_modal` to use tensor path for single-modality visual

**Files:**
- Modify: `/vol/bitbucket/aa6622/project/src/segment_grpo_loop.py` (`_decode_modal` inside `_decode_selected_trace`, approx. lines 430–473)

- [ ] **Step 1: Implement — when `decode_unroll` exists and `latent_obj` is `torch.Tensor` and modality is `visual`, call tensor API first**

Replace the block that always sets `decode_input = latent_obj if isinstance(latent_obj, dict) else {modality: latent_obj}` before `decode_unroll` with logic equivalent to:

```python
decode_unroll = getattr(model_bundle.model, "decode_unroll", None)
if decode_unroll is not None:
    tensor_payload: torch.Tensor | None = latent_obj if (modality == "visual" and isinstance(latent_obj, torch.Tensor)) else None
    if tensor_payload is not None:
        try:
            decoded = decode_unroll(tensor_payload, batch=True)
        except TypeError:
            decoded = decode_unroll(tensor_payload)
    else:
        decode_input = latent_obj if isinstance(latent_obj, dict) else {modality: latent_obj}
        try:
            decoded = decode_unroll(decode_input, batch=True)
        except TypeError:
            ...
```

Keep existing dict path for fused `{"visual": ..., "proprio": ...}` (modality label `visual+proprio` still passes a dict; do **not** use tensor shortcut for that).

- [ ] **Step 2: Drop or narrow proprio-only `decode_unroll` path**

Upstream does not decode RGB from proprio alone. For `modality == "proprio"` and tensor input, return clear failure: `"proprio decode_unroll: no image_head path for proprio-only in JEPA EncPredWM"` instead of calling `decode_unroll({"proprio": t})` (which KeyErrors on `"visual"`).

- [ ] **Step 3: Run Task 1 test — expect PASS**

```bash
cd /vol/bitbucket/aa6622/project && python -m pytest tests/test_segment_grpo_loop.py::test_decode_visual_only_must_not_use_single_key_dict -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/segment_grpo_loop.py
git commit -m "fix: JEPA decode_unroll visual-only uses tensor, not single-key dict"
```

---

### Task 3: Reconcile existing decoder unit test with fused dict contract

**Files:**
- Modify: `/vol/bitbucket/aa6622/project/tests/test_segment_grpo_loop.py` (`test_decode_selected_trace_prefers_visual_modal_and_accepts_structured_latents`)

- [ ] **Step 1: Update mock `decode_unroll` to accept both keys (hub-like)**

```python
def decode_unroll(self, payload: dict[str, object], batch: bool = False) -> np.ndarray:
    self.calls += 1
    assert "visual" in payload
    assert "proprio" in payload
    self.last_payload = payload
    return np.zeros((1, 2, 3, 2, 2), dtype=np.float32)
```

- [ ] **Step 2: Run test**

```bash
cd /vol/bitbucket/aa6622/project && python -m pytest tests/test_segment_grpo_loop.py::test_decode_selected_trace_prefers_visual_modal_and_accepts_structured_latents -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_segment_grpo_loop.py
git commit -m "test: align decode stub with JEPA dict decode_unroll contract"
```

---

### Task 4: Iterative mode — one decode timestep per chunk action

**Files:**
- Modify: `/vol/bitbucket/aa6622/project/src/segment_grpo_loop.py`

- [ ] **Step 1: Add helper (module level, near `_append_trace_steps`)**

```python
def _append_last_timestep_as_numpy_list(target: list[np.ndarray], raw: Any) -> None:
    """Append a single [B,V,H,W,D]-like slice from the last T index of unroll visual/proprio."""
    if raw is None:
        return
    if torch is not None and torch.is_tensor(raw):
        t = raw.detach().cpu()
        if t.ndim < 1 or t.shape[0] == 0:
            return
        last = t[-1]
        target.append(np.asarray(last, dtype=np.float32))
        return
    raw_np = np.asarray(raw)
    if raw_np.ndim == 0 or raw_np.shape[0] == 0:
        return
    target.append(np.asarray(raw_np[-1], dtype=np.float32))
```

- [ ] **Step 2: In `score_chunk_by_goal_latent` iterative branch, replace `_append_trace_steps(decode_visual_steps, ...)` with `_append_last_timestep_as_numpy_list`**

Same for `decode_proprio_steps`. Leave **batched** branch using full `_append_trace_steps` (one unroll → full sequence) or optionally only last — **batched**: keep full trace so `decode_unroll` receives full `T` matching one batched rollout (document in comment).

- [ ] **Step 3: Add unit test — iterative unroll mock returns 2 time steps; list length == chunk actions**

Use a fake `unroll` returning TensorDict with `visual` shape `[2, 1, 1, 2, 2, 2]` and `proprio` `[2, 1, 1]` per call. Assert `len(decode_trace.visual_latents) == chunk_len` (3) and last slice marks step index (e.g. fill `visual[-1]` with float `self.n`). **Goal latent:** after implementing the test, run `score_chunk_by_goal_latent` once in a REPL with the same mock and `return_latent_trace=False` to read `pred.shape`, or build `goal_latent = torch.zeros_like(pred)` inside the test by calling the scoring body with a tiny helper — minimal approach: patch `_extract_latent_with_fallback` to return `torch.ones(4)` so goal is `torch.zeros(4)` and distance is finite.

```python
def test_iterative_decode_trace_one_step_per_action(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    tensordict = pytest.importorskip("tensordict")
    TensorDict = tensordict.TensorDict

    def fixed_extract(z):
        return torch.ones(4, dtype=torch.float32)

    monkeypatch.setattr("segment_grpo_loop._extract_latent_with_fallback", fixed_extract)

    class _IterUnroll:
        def __init__(self) -> None:
            self.n = 0

        def encode(self, obs):
            return TensorDict(
                {"visual": torch.zeros(1, 1, 1, 2, 2, 2), "proprio": torch.zeros(1, 1, 1)},
                device=torch.device("cpu"),
            )

        def unroll(self, z, act_suffix, debug=False):
            self.n += 1
            v = torch.zeros(2, 1, 1, 2, 2, 2)
            v[-1].fill_(float(self.n))
            p = torch.zeros(2, 1, 1)
            return TensorDict({"visual": v, "proprio": p}, device=torch.device("cpu"))

    m = _IterUnroll()
    bundle = WMBundle(m, SimpleNamespace(), proprio_dim=1, planner_action_dim=4, device=torch.device("cpu"))
    _d, _st, dt = score_chunk_by_goal_latent(
        bundle,
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros(1, dtype=np.float32),
        np.zeros((3, 4), dtype=np.float32),
        torch.zeros(4, dtype=torch.float32),
        chunk_len=3,
        return_latent_trace=True,
        wm_rollout_mode="iterative",
    )
    assert len(dt.visual_latents) == 3
    assert float(dt.visual_latents[-1].reshape(-1)[0]) == 3.0
```

- [ ] **Step 4: Run new test + full segment_grpo_loop tests**

```bash
cd /vol/bitbucket/aa6622/project && python -m pytest tests/test_segment_grpo_loop.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/segment_grpo_loop.py tests/test_segment_grpo_loop.py
git commit -m "fix: iterative decode trace appends last unroll timestep only"
```

---

### Task 5: Stop silent swallow on trace append errors

**Files:**
- Modify: `/vol/bitbucket/aa6622/project/src/segment_grpo_loop.py`

- [ ] **Step 1: Replace `except Exception: pass` around `_append_trace_steps` with logging**

```python
import logging
_log = logging.getLogger(__name__)

# in iterative / batched branches:
try:
    _append_last_timestep_as_numpy_list(decode_visual_steps, unroll_out.get("visual"))
except Exception as exc:
    _log.warning("decode visual trace append failed: %s", exc)
```

- [ ] **Step 2: If `strict_wm_scoring` is plumbed into `score_chunk_by_goal_latent`, re-raise when strict** (optional; requires threading `strict` flag from `rollout_with_chunks` — only do if minimal signature change: add `*, strict_trace: bool = False` default False).

- [ ] **Step 3: Run pytest**

```bash
cd /vol/bitbucket/aa6622/project && python -m pytest tests/test_segment_grpo_loop.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/segment_grpo_loop.py
git commit -m "chore: log WM trace append failures instead of silent pass"
```

---

### Task 6 (product): Explicit scoring vector — visual-first vs proprio vs concat

**Files:**
- Modify: `/vol/bitbucket/aa6622/project/src/segment_grpo_loop.py` (`_extract_latent`, `_encode_state_to_latent`, `_extract_latent_with_fallback` usage for scoring only)

- [ ] **Step 1: Add `_extract_scoring_latent(z_pred, mode: str = "visual")`**

- `visual`: first try `z_pred["visual"]` (TensorDict/dict), flatten last step for distance (match current ` _latent_vector_from_unroll_step` pipeline but **fixed key**).
- `proprio`: use `z_pred["proprio"]` only.
- `concat`: `torch.cat([visual_flat, proprio_flat], dim=-1)` with documented order.

- [ ] **Step 2: Use same mode for `_encode_state_to_latent` goal encoding** so goal and pred live in the same space.

- [ ] **Step 3: Add CLI flag in `run_segment_grpo.py` e.g. `--wm-scoring-latent visual|proprio|concat` default `visual`**; pass into `rollout_with_chunks` → `score_chunk_by_goal_latent`.

- [ ] **Step 4: Tests** — three tiny tests: same fake unroll, different modes, assert `distance` differs or tensor slice used matches key.

- [ ] **Step 5: Run**

```bash
cd /vol/bitbucket/aa6622/project && python -m pytest tests/test_segment_grpo_loop.py tests/test_run_segment_grpo_main.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/segment_grpo_loop.py scripts/run_segment_grpo.py tests/
git commit -m "feat: configurable WM scoring latent (visual/proprio/concat)"
```

---

## Self-review

1. **Spec coverage:** Task 2 fixes dict KeyError; Task 4 fixes trace length/alignment; Task 5 observability; Task 6 addresses modality mismatch vs paper (L_vis / L_prop).
2. **Placeholder scan:** No TBD/TODO in tasks.
3. **Consistency:** Fused path still dict with two keys; visual-only uses tensor; tests updated so mocks match hub.

---

## Execution handoff

Plan complete and saved to [`/vol/bitbucket/aa6622/project/docs/superpowers/plans/2026-04-12-jepa-decoder-contract-fixes.md`](/vol/bitbucket/aa6622/project/docs/superpowers/plans/2026-04-12-jepa-decoder-contract-fixes.md).

**1. Subagent-Driven (recommended)** — dispatch per task, spec then quality review between tasks.

**2. Inline Execution** — run tasks 1→6 in this repo with `python -m pytest` after each chunk.

Which approach?
