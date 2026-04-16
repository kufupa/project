from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segment_grpo_loop import (
    DecodeTrace,
    ScoreTrace,
    _all_candidates_wm_goal_l2_rows,
    _build_real_vs_pred_strip,
    _comparison_ridx_for_column,
    _comparison_strip_overlay_lines,
    _comparison_strip_basename,
    _decode_latent_trace_to_frames,
    _derive_policy_rgb_for_smolvla,
    _ensure_action_matrix,
    _latent_overlay_distance_tables,
    _l2_goal_distance_np,
    _normalize_env_actions_for_wm,
    _pack_env_actions_for_wm,
    _sample_smolvla_chunk,
    _overlay_decode_panel_metadata,
    _select_comparison_frames,
    _to_wm_visual,
    _wm_action_block_factor,
    _stitch_comparison_strip,
    _write_comparison_segment_strip,
    rollout_with_chunks,
    score_chunk_by_goal_latent,
    WMBundle,
)



def test_replay_rollout_contracts_shape_and_scores() -> None:
    """Replay carry mode should advance by chunk and record selected metadata."""
    episode, _adapter = rollout_with_chunks(
        smolvla_bundle=None,
        wm_bundle=None,
        task="push-v3",
        episode_index=0,
        chunk_len=4,
        num_candidates=3,
        max_steps=10,
        carry_mode="replay",
        replay_root=None,
        goal_latent_source=None,
        seed=123,
        train_steps=0,
        dry_run=True,
    )

    assert episode.steps == 10
    assert episode.done is True
    assert episode.metadata.get("smolvla_task_text") == "Push the puck to a goal"
    assert len(episode.segments) == 3
    assert episode.steps == len(episode.actions)
    assert len(episode.latent_scores) == len(episode.segments)
    assert len(episode.selected_scores) == len(episode.segments)
    assert len(episode.selected_indices) == len(episode.segments)

    for segment in episode.segments:
        assert len(segment.candidates) == 3
        assert len(segment.executed_actions) <= 4
        assert segment.selected_index in {c.index for c in segment.candidates}


def test_ensure_action_matrix_padding_and_truncation_regression() -> None:
    """Action matrix helper must preserve leading 4-D action values and pad/truncate safely."""
    short = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    padded = _ensure_action_matrix(short, action_dim=20, length=2)
    assert padded.shape == (2, 20)
    np.testing.assert_allclose(padded[:, :4], short)
    np.testing.assert_allclose(padded[:, 4:], 0.0)

    long = np.arange(12, dtype=np.float32).reshape(3, 4)
    truncated = _ensure_action_matrix(long, action_dim=2, length=2)
    assert truncated.shape == (2, 2)
    expected = np.array([[0.0, 1.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(truncated, expected)



def test_to_wm_visual_feeds_jepa_hub_encode_range() -> None:
    """EncPredWM.encode divides by 255 once; WM input must stay in ~[0, 255] float."""
    torch = pytest.importorskip("torch")
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    t = _to_wm_visual(white, torch.device("cpu"))
    assert t.shape == (1, 1, 3, 256, 256)
    assert float(t.max()) > 200.0
    assert float(t.min()) >= 0.0


def test_wm_encode_unroll_smoke() -> None:
    torch = pytest.importorskip("torch")

    class _FakeModel:
        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[0.25, -0.10, 0.40]]], dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            assert z.shape == (1, 1, 3)
            assert act_suffix.shape == (1, 1, 4)
            return torch.tensor([[[0.50, 0.70, 0.90]]], dtype=torch.float32)

    bundle = WMBundle(
        model=_FakeModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    distance = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=np.array([[0.1, -0.2, 0.3, 0.4]], dtype=np.float32),
        goal_latent=goal_latent,
        chunk_len=1,
        wm_rollout_mode="batched",
    )

    assert isinstance(distance, float)
    assert distance > 0.0


def test_score_chunk_by_goal_latent_uses_scoring_latent_mode_visual() -> None:
    torch = pytest.importorskip("torch")

    class _ScoringModel:
        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.zeros((1, 1, 2), dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> dict[str, object]:
            return {
                "visual": torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
                "proprio": torch.tensor([[[5.0]]], dtype=torch.float32),
            }

    bundle = WMBundle(
        model=_ScoringModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([1.0, 2.0], dtype=torch.float32)
    chunk = np.zeros((1, 4), dtype=np.float32)

    distance = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=1,
        wm_rollout_mode="batched",
        wm_scoring_latent="visual",
    )
    assert distance == 0.0


def test_batched_proprio_scoring_omits_visual_latents_for_decode_when_no_visual_trace() -> None:
    """Do not feed proprio rollout slices into image decode as fake visual latents."""
    torch = pytest.importorskip("torch")

    class _ProprioOnlyDictUnroll:
        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            del obs
            return torch.zeros((1, 1, 3), dtype=torch.float32)

        def unroll(self, z: object, act_suffix: torch.Tensor, debug: bool = False) -> dict[str, object]:
            del z, debug
            return {"proprio": torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)}

    bundle = WMBundle(
        model=_ProprioOnlyDictUnroll(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([1.0, 2.0], dtype=torch.float32)
    chunk = np.zeros((1, 4), dtype=np.float32)

    _distance, _score_trace, decode_trace = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=1,
        return_latent_trace=True,
        wm_rollout_mode="batched",
        wm_scoring_latent="proprio",
    )
    assert decode_trace.visual_latents == []
    assert len(decode_trace.proprio_latents) == 1


def test_score_chunk_by_goal_latent_uses_scoring_latent_mode_proprio() -> None:
    torch = pytest.importorskip("torch")

    class _ScoringModel:
        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.zeros((1, 1, 2), dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> dict[str, object]:
            return {
                "visual": torch.tensor([[[9.0, 9.0]]], dtype=torch.float32),
                "proprio": torch.tensor([[[4.0]]], dtype=torch.float32),
            }

    bundle = WMBundle(
        model=_ScoringModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([4.0], dtype=torch.float32)
    chunk = np.zeros((1, 4), dtype=np.float32)

    distance = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=1,
        wm_rollout_mode="batched",
        wm_scoring_latent="proprio",
    )
    assert distance == 0.0


def test_score_chunk_by_goal_latent_uses_scoring_latent_mode_concat() -> None:
    torch = pytest.importorskip("torch")

    class _ScoringModel:
        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.zeros((1, 1, 2), dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> dict[str, object]:
            return {
                "visual": torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
                "proprio": torch.tensor([[[3.0]]], dtype=torch.float32),
            }

    bundle = WMBundle(
        model=_ScoringModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    chunk = np.zeros((1, 4), dtype=np.float32)

    distance = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=1,
        wm_rollout_mode="batched",
        wm_scoring_latent="concat",
    )
    assert distance == 0.0


def test_wm_iterative_rollout_calls_unroll_per_action() -> None:
    torch = pytest.importorskip("torch")

    class _CountingModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            self.calls += 1
            return torch.tensor([[[float(self.calls), 0.0, 0.0]]], dtype=torch.float32)

    model = _CountingModel()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.zeros(3, dtype=torch.float32)
    chunk_len = 4
    chunk = np.zeros((chunk_len, 4), dtype=np.float32)

    score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=chunk_len,
        wm_rollout_mode="iterative",
    )
    assert model.calls == chunk_len


def test_iterative_rollout_keeps_structured_score_decode_traces_and_scores_from_final_latent() -> None:
    torch = pytest.importorskip("torch")

    class _TracingModel:
        def __init__(self) -> None:
            self.unroll_calls = 0

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

        def unroll(self, z: object, act_suffix: torch.Tensor, debug: bool = False) -> dict[str, object]:
            step = self.unroll_calls
            self.unroll_calls += 1
            latent = torch.tensor(
                [[[float(step + 1), float(step + 1), float(step + 1)]]], dtype=torch.float32
            )
            visual = torch.tensor(
                [[[float(step + 1), float(step + 2), float(step + 3)]]], dtype=torch.float32
            )
            proprio = torch.tensor([[[float(step + 1), -float(step + 1)]]], dtype=torch.float32)
            return {"latent": latent, "visual": visual, "proprio": proprio}

    model = _TracingModel()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
    chunk_len = 3
    chunk = np.zeros((chunk_len, 4), dtype=np.float32)

    distance, score_trace, decode_trace = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=chunk_len,
        return_latent_trace=True,
        wm_rollout_mode="iterative",
    )

    assert model.unroll_calls == chunk_len
    assert len(score_trace.step_vectors) == chunk_len
    np.testing.assert_allclose(score_trace.final_vector, np.array([3.0, 3.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(score_trace.step_vectors[-1], score_trace.final_vector)
    assert np.array_equal(decode_trace.visual_latents[0], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert np.array_equal(decode_trace.visual_latents[-1], np.array([[3.0, 4.0, 5.0]], dtype=np.float32))
    assert np.array_equal(decode_trace.proprio_latents[0], np.array([[1.0, -1.0]], dtype=np.float32))
    assert np.array_equal(decode_trace.proprio_latents[-1], np.array([[3.0, -3.0]], dtype=np.float32))
    assert distance == 0.0


def test_iterative_decode_trace_one_step_per_action(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    tensordict = pytest.importorskip("tensordict")
    TensorDict = tensordict.TensorDict

    def fixed_extract(_z: object) -> torch.Tensor:
        return torch.ones(4, dtype=torch.float32)

    monkeypatch.setattr("segment_grpo_loop._extract_latent_with_fallback", fixed_extract)

    class _IterUnroll:
        def __init__(self) -> None:
            self.n = 0

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            del obs
            return TensorDict(
                {"visual": torch.zeros(1, 1, 1, 2, 2, 2), "proprio": torch.zeros(1, 1, 1)},
                device=torch.device("cpu"),
            )

        def unroll(self, z: object, act_suffix: torch.Tensor, debug: bool = False) -> TensorDict:
            del z, debug
            self.n += 1
            v = torch.zeros(2, 1, 1, 2, 2, 2)
            v[-1].fill_(float(self.n))
            p = torch.zeros(2, 1, 1)
            return TensorDict({"visual": v, "proprio": p}, device=torch.device("cpu"))

    model = _IterUnroll()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=1,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    chunk_len = 3
    chunk = np.zeros((chunk_len, 4), dtype=np.float32)
    goal_latent = torch.zeros(4, dtype=torch.float32)

    _distance, score_trace, decode_trace = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(1, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=chunk_len,
        return_latent_trace=True,
        wm_rollout_mode="iterative",
    )

    assert len(decode_trace.visual_latents) == chunk_len
    assert float(decode_trace.visual_latents[-1].reshape(-1)[0]) == 3.0
    assert score_trace.final_vector.size == 8


def test_wm_batched_rollout_single_unroll_call() -> None:
    torch = pytest.importorskip("torch")

    class _CountingModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            self.calls += 1
            t = int(act_suffix.shape[0])
            return torch.tensor([[[float(t), 0.0, 0.0]]], dtype=torch.float32).expand(t, 1, 3)

    model = _CountingModel()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.zeros(3, dtype=torch.float32)
    chunk_len = 4
    chunk = np.zeros((chunk_len, 4), dtype=np.float32)

    score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=chunk_len,
        wm_rollout_mode="batched",
    )
    assert model.calls == 1


def test_decode_selected_trace_prefers_fused_modal_and_accepts_structured_latents() -> None:
    torch = pytest.importorskip("torch")

    class _Decoder:
        def __init__(self) -> None:
            self.calls = 0
            self.last_payload: dict[str, object] | None = None
            self.last_mode_visual = False

        def decode_unroll(self, payload: dict[str, object], batch: bool = False) -> np.ndarray:
            self.calls += 1
            assert set(payload.keys()) == {"visual", "proprio"}
            self.last_payload = payload
            return np.zeros((1, 2, 3, 2, 2), dtype=np.float32)

    model = _Decoder()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    trace = DecodeTrace(
        visual_latents=[
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        ],
        proprio_latents=[
            np.array([[7.0, 8.0]], dtype=np.float32),
            np.array([[9.0, 10.0]], dtype=np.float32),
        ],
        selected_candidate_index=1,
    )

    frames, failure = _decode_latent_trace_to_frames(bundle, trace)
    assert failure is None
    assert model.calls == 1
    assert model.last_payload is not None
    assert set(model.last_payload.keys()) == {"visual", "proprio"}
    assert len(frames) == 2
    assert frames[0].shape == (2, 2, 3)


def test_decode_visual_only_must_not_use_single_key_dict() -> None:
    torch = pytest.importorskip("torch")

    class _JepaLikeDecodeUnroll:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def decode_unroll(self, predicted_encs, batch: bool = False) -> np.ndarray:
            if isinstance(predicted_encs, dict):
                _ = predicted_encs["visual"]
                _ = predicted_encs["proprio"]
                self.calls.append((dict, set(predicted_encs.keys())))
            elif torch.is_tensor(predicted_encs):
                self.calls.append((torch.Tensor, predicted_encs.shape))
            else:
                raise TypeError(predicted_encs)
            return np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)

    model = _JepaLikeDecodeUnroll()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    trace = DecodeTrace(
        visual_latents=[np.zeros((1, 1, 1, 2, 2, 4), dtype=np.float32)],
        proprio_latents=[],
    )

    frames, failure = _decode_latent_trace_to_frames(bundle, trace)
    assert failure is None
    assert len(frames) == 1
    assert model.calls
    assert model.calls[0][0] is torch.Tensor


def test_rollout_only_decodes_selected_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class _SelectiveDecodeModel:
        def __init__(self) -> None:
            self.unroll_calls = 0
            self.decode_calls = 0

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

        def unroll(self, z: object, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            self.unroll_calls += 1
            action_value = float(act_suffix.reshape(-1)[0]) if act_suffix.ndim > 0 else 0.0
            return torch.tensor([[[action_value, 0.0, 0.0]]], dtype=torch.float32)

        def decode_unroll(self, payload: dict[str, object], batch: bool = False) -> np.ndarray:
            self.decode_calls += 1
            return np.zeros((1, 2, 3, 2, 2), dtype=np.float32)

    def _fixed_synthetic_chunk(
        plan_dim: int, chunk_len: int, candidate_idx: int, rng: np.random.Generator
    ) -> np.ndarray:
        return np.full((chunk_len, plan_dim), float(candidate_idx), dtype=np.float32)

    monkeypatch.setattr("segment_grpo_loop._synthetic_chunk", _fixed_synthetic_chunk)
    model = _SelectiveDecodeModel()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_path = tmp_path / "goal.json"
    goal_path.write_text('{"latent": [0, 0, 0]}', encoding="utf-8")

    episode, _ = rollout_with_chunks(
        smolvla_bundle=None,
        wm_bundle=bundle,
        task="push-v3",
        episode_index=0,
        chunk_len=4,
        num_candidates=2,
        max_steps=2,
        carry_mode="replay",
        replay_root=None,
        goal_latent_source=str(goal_path),
        seed=7,
        dry_run=True,
        strict_wm_scoring=False,
        strict_decode=False,
        wm_rollout_mode="iterative",
    )

    assert len(episode.segments) == 1
    segment = episode.segments[0]
    selected_idx = segment.selected_index
    assert 0 <= selected_idx < len(segment.candidates)
    assert episode.selected_indices == [selected_idx]
    assert segment.candidates[selected_idx].meta.get("decode_status") == "ok"
    assert sum(1 for candidate in segment.candidates if candidate.meta.get("decode_status") == "ok") == 1
    assert model.decode_calls == 1
    assert episode.metadata["wm_scoring_statuses"] == ["ok"]
    assert episode.metadata["decode_statuses"] == ["ok"]


def test_normalize_and_pack_env_actions_for_wm_factor5() -> None:
    torch = pytest.importorskip("torch")

    class _Preproc:
        action_mean = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        action_std = torch.ones(4, dtype=torch.float32)

        def normalize_actions(self, a: torch.Tensor) -> torch.Tensor:
            return (a - self.action_mean) / self.action_std

    actions = np.array(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    env_d, wm_d = 4, 20
    f = _wm_action_block_factor(env_d, wm_d)
    assert f == 5
    norm = _normalize_env_actions_for_wm(_Preproc(), actions, env_d, torch.device("cpu"))
    assert norm.shape == (6, 4)
    assert abs(float(norm[0, 0]) - 9.0) < 1e-5
    packed = _pack_env_actions_for_wm(norm, f, env_d, wm_d)
    assert packed.shape == (2, 20)
    assert abs(float(packed[1, 0]) - 1.0) < 1e-5


def test_score_chunk_normalizes_and_packs_before_unroll() -> None:
    torch = pytest.importorskip("torch")

    class _Recorder:
        def __init__(self) -> None:
            self.last_act: torch.Tensor | None = None

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            del obs
            return torch.zeros((1, 1, 2), dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            del z, debug
            self.last_act = act_suffix.detach().cpu().clone()
            t = int(act_suffix.shape[0])
            return torch.zeros((t, 1, 2), dtype=torch.float32)

    class _Preproc:
        action_mean = torch.ones(4, dtype=torch.float32)
        action_std = torch.ones(4, dtype=torch.float32)

        def normalize_actions(self, a: torch.Tensor) -> torch.Tensor:
            return (a - self.action_mean) / self.action_std

    model = _Recorder()
    bundle = WMBundle(
        model=model,
        preprocessor=_Preproc(),
        proprio_dim=4,
        planner_action_dim=20,
        device=torch.device("cpu"),
    )
    chunk = np.ones((5, 4), dtype=np.float32)
    goal = torch.zeros(2, dtype=torch.float32)

    score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal,
        chunk_len=5,
        wm_rollout_mode="batched",
    )
    assert model.last_act is not None
    assert tuple(model.last_act.shape) == (1, 1, 20)
    np.testing.assert_allclose(model.last_act.numpy()[0, 0, :4], 0.0, atol=1e-6)


def test_wm_iterative_unroll_once_per_packed_wm_step() -> None:
    torch = pytest.importorskip("torch")

    class _CountPacked:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            del obs
            return torch.zeros((1, 1, 3), dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            del z, debug
            self.calls += 1
            assert act_suffix.shape == (1, 1, 20)
            return torch.zeros((1, 1, 3), dtype=torch.float32)

    class _Preproc:
        action_mean = torch.zeros(4, dtype=torch.float32)
        action_std = torch.ones(4, dtype=torch.float32)

        def normalize_actions(self, a: torch.Tensor) -> torch.Tensor:
            return a

    model = _CountPacked()
    bundle = WMBundle(
        model=model,
        preprocessor=_Preproc(),
        proprio_dim=4,
        planner_action_dim=20,
        device=torch.device("cpu"),
    )
    chunk_len = 8
    chunk = np.zeros((chunk_len, 4), dtype=np.float32)
    score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=torch.zeros(3, dtype=torch.float32),
        chunk_len=chunk_len,
        wm_rollout_mode="iterative",
    )
    assert model.calls == 2


def test_derive_policy_rgb_hflip_only_under_jepa_parity() -> None:
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    out_parity = _derive_policy_rgb_for_smolvla(
        img, jepa_parity_sim=True, policy_hflip_corner2=True
    )
    np.testing.assert_array_equal(out_parity, np.flip(img, axis=1))
    out_no = _derive_policy_rgb_for_smolvla(
        img, jepa_parity_sim=False, policy_hflip_corner2=True
    )
    np.testing.assert_array_equal(out_no, img)
    out_flag_off = _derive_policy_rgb_for_smolvla(
        img, jepa_parity_sim=True, policy_hflip_corner2=False
    )
    np.testing.assert_array_equal(out_flag_off, img)


def test_sample_smolvla_chunk_zero_noise_repeats_base(monkeypatch: pytest.MonkeyPatch) -> None:
    def _exec(*_a: object, **_k: object) -> np.ndarray:
        return np.array([0.5, -0.25, 0.0, 0.1], dtype=np.float32)

    def _chunk_fail(*_a: object, **_k: object) -> np.ndarray:
        raise RuntimeError("chunk api unavailable in test")

    fake_helper = SimpleNamespace()
    fake_helper._smolvla_exec_action_chunk = _chunk_fail
    fake_helper._smolvla_exec_action = _exec
    monkeypatch.setattr("segment_grpo_loop._load_jepa_helper_module", lambda: fake_helper)
    rng = np.random.default_rng(999)
    chunk, meta = _sample_smolvla_chunk(
        smolvla_bundle=object(),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_len=4,
        env_action_dim=4,
        task_text="t",
        rng=rng,
        noise_std=0.0,
    )
    assert chunk.shape == (4, 4)
    assert meta.get("chunk_generation_mode") == "single_step_fallback"
    for i in range(1, 4):
        np.testing.assert_array_equal(chunk[i], chunk[0])


def test_sample_smolvla_chunk_positive_noise_varies_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def _exec(*_a: object, **_k: object) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    def _chunk_fail(*_a: object, **_k: object) -> np.ndarray:
        raise RuntimeError("chunk api unavailable in test")

    fake_helper = SimpleNamespace()
    fake_helper._smolvla_exec_action_chunk = _chunk_fail
    fake_helper._smolvla_exec_action = _exec
    monkeypatch.setattr("segment_grpo_loop._load_jepa_helper_module", lambda: fake_helper)
    rng = np.random.default_rng(42)
    chunk, _meta = _sample_smolvla_chunk(
        smolvla_bundle=object(),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_len=5,
        env_action_dim=4,
        task_text="t",
        rng=rng,
        noise_std=0.5,
    )
    assert not np.allclose(chunk[0], chunk[1])


def test_sampled_chunk_keeps_env_action_dim_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_helper = SimpleNamespace()

    def _exec(*_a: object, **_k: object) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def _chunk_fail(*_a: object, **_k: object) -> np.ndarray:
        raise RuntimeError("chunk api unavailable in test")

    fake_helper._smolvla_exec_action_chunk = _chunk_fail
    fake_helper._smolvla_exec_action = _exec
    monkeypatch.setattr("segment_grpo_loop._load_jepa_helper_module", lambda: fake_helper)
    rng = np.random.default_rng(0)
    chunk, _meta = _sample_smolvla_chunk(
        smolvla_bundle=object(),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_len=3,
        env_action_dim=4,
        task_text="t",
        rng=rng,
        noise_std=0.1,
    )
    assert chunk.shape == (3, 4)


def test_sample_smolvla_chunk_sequence_head_uses_distinct_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # First column must stay in [-1, 1] so np.clip in _sample_smolvla_chunk does not merge rows.
    seq = np.stack(
        [np.array([v, 0.0, 0.0, 0.0], dtype=np.float32) for v in (-1.0, -0.5, 0.0, 0.5, 1.0)],
        axis=0,
    )

    def _chunk_ok(*_a: object, **_k: object) -> np.ndarray:
        return seq

    fake_helper = SimpleNamespace()
    fake_helper._smolvla_exec_action_chunk = _chunk_ok
    monkeypatch.setattr("segment_grpo_loop._load_jepa_helper_module", lambda: fake_helper)
    rng = np.random.default_rng(1)
    chunk, meta = _sample_smolvla_chunk(
        smolvla_bundle=object(),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_len=5,
        env_action_dim=4,
        task_text="t",
        rng=rng,
        noise_std=0.0,
    )
    assert meta.get("chunk_generation_mode") == "sequence_head"
    assert chunk.shape == (5, 4)
    assert meta.get("unique_action_rows") == 5
    np.testing.assert_array_equal(chunk, seq)


def test_sample_smolvla_chunk_sequence_head_pads_short_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = np.stack(
        [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-0.25, 0.0, 0.0, 0.0], dtype=np.float32),
        ],
        axis=0,
    )

    def _chunk_ok(*_a: object, **_k: object) -> np.ndarray:
        return seq

    fake_helper = SimpleNamespace()
    fake_helper._smolvla_exec_action_chunk = _chunk_ok
    monkeypatch.setattr("segment_grpo_loop._load_jepa_helper_module", lambda: fake_helper)
    rng = np.random.default_rng(2)
    chunk, meta = _sample_smolvla_chunk(
        smolvla_bundle=object(),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_len=5,
        env_action_dim=4,
        task_text="t",
        rng=rng,
        noise_std=0.0,
    )
    assert meta.get("chunk_generation_mode") == "sequence_head"
    assert chunk.shape == (5, 4)
    np.testing.assert_array_equal(chunk[0], seq[0])
    np.testing.assert_array_equal(chunk[1], seq[1])
    for i in range(2, 5):
        np.testing.assert_array_equal(chunk[i], seq[1])


def test_select_comparison_frames_with_wm_step_factor() -> None:
    real_frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(9)]
    pred_frames = [np.full((2, 2, 3), 100 + k, dtype=np.uint8) for k in range(2)]
    selected_real, selected_pred, pred_indices = _select_comparison_frames(
        real_frames,
        pred_frames,
        carried_steps=8,
        env_steps_per_wm_step=5,
    )
    assert len(selected_real) == len(selected_pred) == len(pred_indices) == 2
    assert pred_indices == [0, 1]
    np.testing.assert_array_equal(selected_real[0], real_frames[5])
    np.testing.assert_array_equal(selected_real[1], real_frames[8])
    np.testing.assert_array_equal(selected_pred[0], pred_frames[0])


def test_select_comparison_frames_keeps_t0_as_context() -> None:
    real_frames = [
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 20, dtype=np.uint8),
        np.full((2, 2, 3), 30, dtype=np.uint8),
    ]
    pred_frames = [
        np.full((2, 2, 3), 40, dtype=np.uint8),
        np.full((2, 2, 3), 50, dtype=np.uint8),
        np.full((2, 2, 3), 60, dtype=np.uint8),
    ]
    selected_real, selected_pred, pred_indices = _select_comparison_frames(
        real_frames, pred_frames, carried_steps=2
    )
    assert len(selected_real) == 2
    assert len(selected_pred) == 2
    assert pred_indices == [0, 1]
    np.testing.assert_array_equal(selected_real[0], real_frames[0])
    np.testing.assert_array_equal(selected_pred[0], pred_frames[0])


def test_comparison_ridx_for_column_factor5() -> None:
    assert _comparison_ridx_for_column(0, factor=5, carried_steps=8) == 5
    assert _comparison_ridx_for_column(1, factor=5, carried_steps=8) == 8
    assert _comparison_ridx_for_column(0, factor=1, carried_steps=8) == 0


def test_comparison_strip_overlay_lines_includes_segment_when_set() -> None:
    lines = _comparison_strip_overlay_lines(
        column_idx=0,
        total_columns=4,
        factor=1,
        carried_steps=8,
        overlay_env_step_start=10,
        overlay_selected_candidate_index=2,
        wm_step_index=0,
        d_full=[1.0, 2.0],
        delta_full=[0.5, 0.25],
        overlay_segment_index=12,
    )
    assert "seg 0012" in lines
    assert any(line.startswith("d ") for line in lines)


def test_comparison_strip_overlay_lines_omits_segment_when_none() -> None:
    lines = _comparison_strip_overlay_lines(
        column_idx=0,
        total_columns=2,
        factor=1,
        carried_steps=2,
        overlay_env_step_start=0,
        overlay_selected_candidate_index=0,
        wm_step_index=0,
        d_full=None,
        delta_full=None,
        overlay_segment_index=None,
    )
    assert not any(line.startswith("seg ") for line in lines)


def test_stitch_comparison_strip_inserts_gutter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p0 = tmp_path / "s0.png"
    p1 = tmp_path / "s1.png"
    p0.touch()
    p1.touch()
    a = np.zeros((8, 4, 3), dtype=np.uint8)
    b = np.full((8, 6, 3), 200, dtype=np.uint8)

    def _imread(path: object) -> np.ndarray:
        p = Path(path)
        if p == p0:
            return a
        if p == p1:
            return b
        raise AssertionError(f"unexpected path {path!r}")

    captured: dict[str, np.ndarray] = {}

    def _imwrite(path: object, arr: np.ndarray) -> None:
        captured[str(Path(path))] = np.asarray(arr)

    dummy_v2 = ModuleType("imageio.v2")
    dummy_v2.imread = _imread
    dummy_v2.imwrite = _imwrite
    dummy_pkg = ModuleType("imageio")
    dummy_pkg.__path__ = []  # type: ignore[attr-defined]
    dummy_pkg.v2 = dummy_v2

    monkeypatch.setitem(sys.modules, "imageio", dummy_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", dummy_v2)

    out0 = tmp_path / "st0.png"
    out1 = tmp_path / "st1.png"
    path0, err0 = _stitch_comparison_strip([p0, p1], out0, gutter_pixels=0)
    path1, err1 = _stitch_comparison_strip([p0, p1], out1, gutter_pixels=3, gutter_rgb=(10, 20, 30))
    assert err0 is None and err1 is None
    assert path0 == out0 and path1 == out1
    arr0 = captured[str(out0)]
    arr1 = captured[str(out1)]
    assert arr0.shape[1] == 4 + 6
    assert arr1.shape[1] == 4 + 3 + 6
    gutter_slice = arr1[:, 4 : 4 + 3, :]
    assert np.all(gutter_slice == np.array([10, 20, 30], dtype=np.uint8))


def test_l2_goal_distance_np_prefix_matches_torch_slice() -> None:
    v = np.array([3.0, 4.0], dtype=np.float32)
    g = np.array([0.0, 0.0, 99.0], dtype=np.float32)
    assert abs(_l2_goal_distance_np(v, g) - 5.0) < 1e-6


def test_latent_overlay_distance_tables_delta_vs_initial() -> None:
    init = np.array([0.0, 0.0], dtype=np.float32)
    s0 = np.array([1.0, 0.0], dtype=np.float32)
    s1 = np.array([1.0, 1.0], dtype=np.float32)
    goal = np.array([0.0, 0.0], dtype=np.float32)
    d_full, delta_full = _latent_overlay_distance_tables(init, [s0, s1], goal)
    assert len(d_full) == 2 and len(delta_full) == 2
    assert abs(d_full[0] - 1.0) < 1e-6
    d_init = _l2_goal_distance_np(init, goal)
    assert abs(delta_full[0] - (d_full[0] - d_init)) < 1e-6
    assert abs(delta_full[1] - (d_full[1] - d_full[0])) < 1e-6


def test_overlay_decode_panel_metadata_appends_footer_row() -> None:
    pred = np.full((40, 60, 3), 200, dtype=np.uint8)
    out = _overlay_decode_panel_metadata(pred, ["line1", "line2"])
    assert out.shape[0] > pred.shape[0]
    assert out.shape[1] == pred.shape[1]
    assert np.array_equal(out[: pred.shape[0]], pred)
    footer = out[pred.shape[0] :]
    assert footer.shape[0] > 0
    assert np.any(np.any(footer != 0, axis=2))


def test_build_real_vs_pred_strip_overlay_on_changes_array() -> None:
    real_frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    pred_frames = [np.full((8, 8, 3), 200, dtype=np.uint8) for _ in range(3)]
    goal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    trace = ScoreTrace(
        step_vectors=[np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in range(3)],
        final_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        initial_vector=np.zeros(3, dtype=np.float32),
    )
    base = _build_real_vs_pred_strip(
        real_frames,
        pred_frames,
        carried_steps=3,
        env_steps_per_wm_step=1,
        overlay_decode_meta=False,
    )
    with_overlay = _build_real_vs_pred_strip(
        real_frames,
        pred_frames,
        carried_steps=3,
        env_steps_per_wm_step=1,
        overlay_decode_meta=True,
        overlay_env_step_start=10,
        overlay_selected_candidate_index=0,
        overlay_wm_env_steps_per_wm_step=1,
        overlay_goal_latent_np=goal,
        overlay_score_trace=trace,
        overlay_segment_index=5,
    )
    assert with_overlay.shape[1] == base.shape[1]
    assert with_overlay.shape[0] > base.shape[0]
    overlay = with_overlay.shape[0] - base.shape[0]
    assert overlay > 0
    frame_h = real_frames[0].shape[0]
    frame_w = real_frames[0].shape[1]
    assert np.array_equal(with_overlay[:frame_h, :frame_w], real_frames[0])
    assert np.array_equal(with_overlay[frame_h : frame_h * 2, frame_w : frame_w * 2], pred_frames[0])
    footer = with_overlay[(frame_h * 2) : (frame_h * 2 + overlay), :frame_w]
    assert footer.shape[0] > 0
    assert not np.array_equal(footer, np.zeros_like(footer))


def test_comparison_strip_basename_step_range_and_wmf_prefix() -> None:
    assert (
        _comparison_strip_basename(
            segment_index=4,
            env_step_start=12,
            carried_steps=6,
            selected_candidate_index=1,
            wm_env_steps_per_wm_step=1,
        )
        == "comparison_strip_steps_0012_to_0018_seg0004_cand001.png"
    )
    assert (
        _comparison_strip_basename(
            segment_index=3,
            env_step_start=3,
            carried_steps=2,
            selected_candidate_index=12,
            wm_env_steps_per_wm_step=5,
        )
        == "wmf05_comparison_strip_steps_0003_to_0005_seg0003_cand012.png"
    )


def test_write_comparison_segment_strip_returns_deterministic_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Path] = {}

    def _fake_imwrite(path: Path, _image: np.ndarray) -> None:
        captured["path"] = path

    dummy_v2 = ModuleType("imageio.v2")
    dummy_v2.imwrite = _fake_imwrite
    dummy_pkg = ModuleType("imageio")
    dummy_pkg.__path__ = []  # type: ignore[attr-defined]
    dummy_pkg.v2 = dummy_v2

    monkeypatch.setitem(sys.modules, "imageio", dummy_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", dummy_v2)

    real_frames = [np.full((2, 2, 3), 10, dtype=np.uint8), np.full((2, 2, 3), 20, dtype=np.uint8)]
    pred_frames = [np.full((2, 2, 3), 30, dtype=np.uint8), np.full((2, 2, 3), 40, dtype=np.uint8)]
    path, failure = _write_comparison_segment_strip(
        out_dir=tmp_path,
        episode_index=7,
        segment_index=3,
        real_frames=real_frames,
        pred_frames=pred_frames,
        env_step_start=3,
        selected_candidate_index=12,
        carried_steps=2,
    )

    assert failure is None
    expected_name = _comparison_strip_basename(
        segment_index=3,
        env_step_start=3,
        carried_steps=2,
        selected_candidate_index=12,
        wm_env_steps_per_wm_step=1,
    )
    assert path == tmp_path / "episode_0007" / expected_name
    assert captured["path"] == path


def test_write_comparison_segment_strip_appends_wm_megastep_footer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: list[np.ndarray] = []

    def _fake_imwrite(path: Path, image: np.ndarray) -> None:
        written.append(np.asarray(image))

    dummy_v2 = ModuleType("imageio.v2")
    dummy_v2.imwrite = _fake_imwrite
    dummy_pkg = ModuleType("imageio")
    dummy_pkg.__path__ = []  # type: ignore[attr-defined]
    dummy_pkg.v2 = dummy_v2
    monkeypatch.setitem(sys.modules, "imageio", dummy_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", dummy_v2)

    real_frames = [np.full((4, 4, 3), 10, dtype=np.uint8), np.full((4, 4, 3), 11, dtype=np.uint8)]
    pred_frames = [np.full((4, 4, 3), 20, dtype=np.uint8), np.full((4, 4, 3), 21, dtype=np.uint8)]
    footer = ["title", "k0 x", "k1 y"]
    path, err = _write_comparison_segment_strip(
        out_dir=tmp_path,
        episode_index=0,
        segment_index=0,
        real_frames=real_frames,
        pred_frames=pred_frames,
        env_step_start=0,
        selected_candidate_index=0,
        carried_steps=1,
        wm_megastep_footer_lines=footer,
        wm_megastep_footer_min_lines=6,
    )
    assert err is None
    assert path is not None
    assert len(written) == 1
    base_h = real_frames[0].shape[0] + pred_frames[0].shape[0]
    assert written[0].shape[0] > base_h


def test_all_candidates_wm_goal_l2_rows_two_candidates_ints() -> None:
    """Per-candidate integer L2 list; no delta/blk strings; cand1 has no WM trace."""
    goal = np.zeros(3, dtype=np.float32)
    st0 = ScoreTrace(
        step_vectors=[
            np.array([12.3, 0.0, 0.0], dtype=np.float32),
            np.array([45.6, 0.0, 0.0], dtype=np.float32),
        ],
        final_vector=np.zeros(3, dtype=np.float32),
        initial_vector=np.zeros(3, dtype=np.float32),
    )
    lines, meta, max_k = _all_candidates_wm_goal_l2_rows(
        goal,
        {0: st0, 1: None},
        num_candidates=2,
    )
    assert max_k == 2
    assert len(lines) == 1 + 2
    assert meta[0]["candidate_index"] == 0
    assert meta[0]["d_goal_l2_wm_int"] == [12, 46]
    assert meta[1]["candidate_index"] == 1
    assert meta[1]["d_goal_l2_wm_int"] == [None, None]
    blob = "\n".join(lines)
    assert "dd=" not in blob and "|blk|" not in blob
    assert "cand00" in lines[1]
    assert "    12" in lines[1] and "    46" in lines[1]


def test_wm_megastep_action_range_footer_line() -> None:
    row = _wm_megastep_action_range_footer_line(
        carried_steps=13,
        wm_stride=5,
        wm_step_count=3,
    )
    assert "action range" in row
    assert "  0:5" in row
    assert "  5:10" in row
    assert "10:13" in row


def test_iterative_rollout_final_latent_matches_last_step() -> None:
    torch = pytest.importorskip("torch")

    class _CountingModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            self.calls += 1
            return torch.tensor([[[float(self.calls), 0.0, 0.0]]], dtype=torch.float32)

    model = _CountingModel()
    bundle = WMBundle(
        model=model,
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    chunk_len = 3
    chunk = np.zeros((chunk_len, 4), dtype=np.float32)

    distance, score_trace, decode_trace = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=chunk,
        goal_latent=goal_latent,
        chunk_len=chunk_len,
        return_latent_trace=True,
        wm_rollout_mode="iterative",
    )
    assert len(score_trace.step_vectors) == chunk_len
    assert model.calls == chunk_len
    assert abs(float(distance) - float(chunk_len)) < 1e-5


def test_rollout_with_chunks_records_strict_and_status_lists_with_fallback_decoding(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class _ScoringSuccessDecodeFailModel:
        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

        def unroll(self, z: object, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            return torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

    bundle = WMBundle(
        model=_ScoringSuccessDecodeFailModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_path = tmp_path / "goal.json"
    goal_path.write_text('{"latent": [1.0, 0.0, 0.0]}', encoding="utf-8")

    episode, _ = rollout_with_chunks(
        smolvla_bundle=None,
        wm_bundle=bundle,
        task="push-v3",
        episode_index=0,
        chunk_len=2,
        num_candidates=2,
        max_steps=2,
        carry_mode="replay",
        replay_root=None,
        goal_latent_source=str(goal_path),
        seed=123,
        dry_run=True,
        strict_wm_scoring=False,
        strict_decode=False,
        wm_rollout_mode="iterative",
    )

    assert episode.metadata["strict_wm_scoring"] is False
    assert episode.metadata["strict_decode"] is False
    assert len(episode.metadata["wm_scoring_statuses"]) == len(episode.segments) == 1
    assert len(episode.metadata["decode_statuses"]) == len(episode.segments) == 1
    assert episode.metadata["wm_scoring_statuses"] == ["ok"]
    assert episode.metadata["decode_statuses"] == ["failed"]
    assert episode.segments[0].metadata["wm_scoring_status"] == "ok"
    assert episode.segments[0].metadata["decode_status"] == "failed"


def test_rollout_with_chunks_records_wm_rollout_mode_metadata() -> None:
    episode, _adapter = rollout_with_chunks(
        smolvla_bundle=None,
        wm_bundle=None,
        task="push-v3",
        episode_index=0,
        chunk_len=4,
        num_candidates=2,
        max_steps=8,
        carry_mode="replay",
        replay_root=None,
        goal_latent_source=None,
        seed=123,
        train_steps=0,
        dry_run=True,
        wm_rollout_mode="batched",
    )
    assert episode.metadata.get("wm_rollout_mode") == "batched"


def test_rollout_with_chunks_records_wm_scoring_latent_metadata() -> None:
    episode, _adapter = rollout_with_chunks(
        smolvla_bundle=None,
        wm_bundle=None,
        task="push-v3",
        episode_index=0,
        chunk_len=4,
        num_candidates=2,
        max_steps=8,
        carry_mode="replay",
        replay_root=None,
        goal_latent_source=None,
        seed=123,
        train_steps=0,
        dry_run=True,
        wm_rollout_mode="iterative",
        wm_scoring_latent="proprio",
    )
    assert episode.metadata.get("wm_scoring_latent") == "proprio"
