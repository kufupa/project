import importlib.util
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _build_torch_import_stub() -> types.ModuleType:
    torch_stub = types.ModuleType("torch")
    torch_stub.is_tensor = lambda _: False
    torch_stub.Tensor = object
    torch_stub.device = lambda *_args, **_kwargs: "cpu"
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)

    def _save(obj, path):
        Path(path).write_bytes(pickle.dumps(obj))

    def _load(path):
        return pickle.loads(Path(path).read_bytes())

    torch_stub.save = _save
    torch_stub.load = _load
    return torch_stub


MODULE = Path(__file__).resolve().parents[1] / "jepa_cem_paired_pushv3_export.py"
SPEC = importlib.util.spec_from_file_location("jepa_export_memory_bounded", MODULE)
jepa_export = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader

_original_torch_module = sys.modules.get("torch")
if _original_torch_module is None:
    sys.modules["torch"] = _build_torch_import_stub()
try:
    sys.modules[SPEC.name] = jepa_export
    SPEC.loader.exec_module(jepa_export)
finally:
    if _original_torch_module is None:
        del sys.modules["torch"]
    else:
        sys.modules["torch"] = _original_torch_module

if not hasattr(jepa_export.torch, "device"):
    jepa_export.torch.device = lambda *_args, **_kwargs: "cpu"


def test_encode_image_payload_returns_uint8_array():
    frame = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)
    out = jepa_export._encode_image_payload(frame)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_encode_latent_payload_not_python_list():
    latent = np.arange(512, dtype=np.float32)
    out = jepa_export._encode_latent_payload(latent, full_latents_export=True)
    assert not isinstance(out, list)
    if jepa_export.torch.is_tensor(out):
        assert int(out.numel()) == 512
    else:
        arr = np.asarray(out)
        assert int(arr.size) == 512


def test_encode_latent_payload_truncates_when_full_export_disabled():
    latent = np.arange(512, dtype=np.float32)
    out = jepa_export._encode_latent_payload(latent, full_latents_export=False)
    assert not isinstance(out, list)
    arr = np.asarray(out)
    assert int(arr.size) == 256
    np.testing.assert_array_equal(arr, latent[:256])


def test_encode_latent_payload_torch_branch_not_python_list():
    class _FakeTensor:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *_shape):
            self._values = self._values.reshape(-1)
            return self

        def numel(self):
            return int(self._values.size)

        def __getitem__(self, item):
            return _FakeTensor(self._values[item])

    original_torch = jepa_export.torch
    jepa_export.torch = types.SimpleNamespace(is_tensor=lambda value: isinstance(value, _FakeTensor))
    try:
        out = jepa_export._encode_latent_payload(_FakeTensor(np.arange(300, dtype=np.float32)), full_latents_export=False)
    finally:
        jepa_export.torch = original_torch

    assert not isinstance(out, list)
    assert isinstance(out, _FakeTensor)
    assert int(out.numel()) == 256


def test_episode_shard_writer_writes_episode_file():
    with tempfile.TemporaryDirectory() as td:
        writer = jepa_export.EpisodeShardWriter(Path(td), episodes_per_shard=1)
        episode = {
            "meta": {"episode_index": 0},
            "state": [[0.0, 1.0, 2.0]],
            "actions": [[0.0, 0.0, 0.0, 0.0]],
            "images": [],
        }
        file_path = writer.write_episode(episode)
        assert isinstance(file_path, Path)
        assert file_path.is_file()
        written_files = writer.finalize()
        assert written_files == [file_path]
        restored = jepa_export.torch.load(file_path)
        assert isinstance(restored, dict)
        assert restored["meta"]["episode_index"] == 0


def test_episode_shard_writer_flush_writes_per_episode_files():
    with tempfile.TemporaryDirectory() as td:
        writer = jepa_export.EpisodeShardWriter(Path(td), episodes_per_shard=2)
        assert writer.write_episode({"meta": {"episode_index": 0}}) is None
        flush_path = writer.write_episode({"meta": {"episode_index": 1}})
        assert isinstance(flush_path, Path)
        files = writer.finalize()
        assert len(files) == 2
        restored = [jepa_export.torch.load(path) for path in files]
        indices = sorted(int(item["meta"]["episode_index"]) for item in restored)
        assert indices == [0, 1]


def test_promote_episode_shards_replaces_existing_destination():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging_dir = root / ".episodes_staging"
        final_dir = root / "episodes"
        staging_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "old.txt").write_text("stale", encoding="utf-8")
        (staging_dir / "episode_000000.pt").write_bytes(b"fresh")

        jepa_export._promote_episode_shards(staging_dir, final_dir)

        assert not staging_dir.exists()
        assert final_dir.is_dir()
        assert (final_dir / "episode_000000.pt").is_file()
        assert not (final_dir / "old.txt").exists()


def test_cleanup_episode_shards_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging_dir = root / ".episodes_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        (staging_dir / "episode_000000.pt").write_bytes(b"fresh")

        jepa_export._cleanup_episode_shards(staging_dir)
        assert not staging_dir.exists()
        jepa_export._cleanup_episode_shards(staging_dir)
        assert not staging_dir.exists()


def test_rss_guard_raises_when_limit_exceeded(monkeypatch):
    monkeypatch.setattr(jepa_export, "_current_rss_gb", lambda: 12.5)
    with pytest.raises(RuntimeError):
        jepa_export._enforce_rss_limit(max_rss_gb=10.0, context="ep=0 step=0")


def test_rss_guard_noop_when_limit_disabled(monkeypatch):
    def _unexpected_rss_read() -> float:
        raise AssertionError("rss sampler should not be called when max_rss_gb <= 0")

    monkeypatch.setattr(jepa_export, "_current_rss_gb", _unexpected_rss_read)
    jepa_export._enforce_rss_limit(max_rss_gb=0.0, context="ep=0 step=0")
    jepa_export._enforce_rss_limit(max_rss_gb=-1.0, context="ep=0 step=0")


def test_rss_guard_no_raise_when_under_limit(monkeypatch):
    monkeypatch.setattr(jepa_export, "_current_rss_gb", lambda: 9.99)
    jepa_export._enforce_rss_limit(max_rss_gb=10.0, context="ep=0 step=0")


def test_rollout_enforces_rss_pre_and_post_step(monkeypatch):
    class _FakeEnv:
        def __init__(self):
            self.action_space = types.SimpleNamespace(shape=(4,))

        def reset(self, seed=None):
            del seed
            return np.zeros(12, dtype=np.float32)

        def step(self, action):
            del action
            return np.zeros(12, dtype=np.float32), 0.0, True, {"success": True}

    calls: list[str] = []

    def _record_enforce(*, max_rss_gb: float, context: str) -> None:
        assert max_rss_gb == 10.0
        calls.append(context)

    monkeypatch.setattr(jepa_export, "_enforce_rss_limit", _record_enforce)
    monkeypatch.setattr(
        jepa_export,
        "_select_executed_action",
        lambda **_kwargs: {"action_executed": [0.0, 0.0, 0.0, 0.0], "policy_source": "heuristic"},
    )

    jepa_export.rollout_episode(
        env=_FakeEnv(),
        max_steps=1,
        pair_key="pair-key",
        wm_bundle=None,
        smolvla_bundle=None,
        task_text="push the puck to the goal",
        cem_horizon=4,
        cem_pop=8,
        cem_iters=2,
        execution_policy="cem_primary",
        store_cem_plan_seq=True,
        store_smolvla_action=True,
        full_latents_export=False,
        rng=np.random.default_rng(0),
        max_rss_gb=10.0,
        rss_log_interval_steps=0,
        episode_index=7,
    )

    assert calls == ["ep=7 step=0 pre_step", "ep=7 step=0 post_step"]


def test_rollout_serialization_uses_compact_image_encoder(monkeypatch):
    class _FakeEnv:
        def __init__(self):
            self.action_space = types.SimpleNamespace(shape=(4,))

        def reset(self, seed=None):
            del seed
            return np.zeros(12, dtype=np.float32)

        def render(self):
            return np.ones((4, 4, 3), dtype=np.uint8) * 255

        def step(self, action):
            del action
            return np.zeros(12, dtype=np.float32), 0.0, True, {"success": True}

    sentinel_image = np.full((2, 2, 3), 17, dtype=np.uint8)
    encode_calls: list[tuple[int, ...]] = []

    def _encode_image_spy(image: Any) -> np.ndarray:
        encode_calls.append(tuple(np.asarray(image).shape))
        return sentinel_image

    monkeypatch.setattr(jepa_export, "_encode_image_payload", _encode_image_spy)
    monkeypatch.setattr(
        jepa_export,
        "_select_executed_action",
        lambda **_kwargs: {"action_executed": [0.0, 0.0, 0.0, 0.0], "policy_source": "heuristic"},
    )

    episode = jepa_export.rollout_episode(
        env=_FakeEnv(),
        max_steps=1,
        pair_key="pair-key",
        wm_bundle=None,
        smolvla_bundle=None,
        task_text="push the puck to the goal",
        cem_horizon=4,
        cem_pop=8,
        cem_iters=2,
        execution_policy="cem_primary",
        store_cem_plan_seq=True,
        store_smolvla_action=True,
        full_latents_export=False,
        rng=np.random.default_rng(0),
        max_rss_gb=0.0,
        rss_log_interval_steps=0,
        episode_index=0,
    )

    assert encode_calls
    assert isinstance(episode["images"][0], np.ndarray)
    np.testing.assert_array_equal(episode["images"][0], sentinel_image)


def test_rollout_serialization_uses_compact_latent_encoder(monkeypatch):
    class _FakeEnv:
        def __init__(self):
            self.action_space = types.SimpleNamespace(shape=(4,))

        def reset(self, seed=None):
            del seed
            return np.zeros(12, dtype=np.float32)

        def step(self, action):
            del action
            return np.zeros(12, dtype=np.float32), 0.0, True, {"success": True}

    class _FakeLatent:
        def to(self, _device):
            return self

    class _FakeModel:
        def encode(self, _obs):
            return _FakeLatent()

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    monkeypatch.setattr(jepa_export.torch, "no_grad", lambda: _NoGrad(), raising=False)
    monkeypatch.setattr(jepa_export, "_render_to_wm_visual", lambda *_args, **_kwargs: np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(jepa_export, "_build_proprio", lambda *_args, **_kwargs: np.zeros((4,), dtype=np.float32))
    monkeypatch.setattr(
        jepa_export,
        "cem_first_action",
        lambda *_args, **_kwargs: (
            np.zeros((4,), dtype=np.float32),
            {
                "meta": {"cem_iterations": 1, "cem_cost": 0.0, "cem_seed": 123},
                "latent_pred": [float(i) for i in range(300)],
                "latent_pred_dim": 300,
            },
        ),
    )

    latent_encode_calls: list[bool] = []

    def _encode_latent_spy(latent_vec: Any, full_latents_export: bool) -> np.ndarray:
        latent_encode_calls.append(bool(full_latents_export))
        return np.asarray(latent_vec, dtype=np.float32)[:16]

    monkeypatch.setattr(jepa_export, "_encode_latent_payload", _encode_latent_spy)

    episode = jepa_export.rollout_episode(
        env=_FakeEnv(),
        max_steps=1,
        pair_key="pair-key",
        wm_bundle=(_FakeModel(), object(), 4, 4, "cpu"),
        smolvla_bundle=None,
        task_text="push the puck to the goal",
        cem_horizon=4,
        cem_pop=8,
        cem_iters=2,
        execution_policy="cem_primary",
        store_cem_plan_seq=True,
        store_smolvla_action=True,
        full_latents_export=False,
        rng=np.random.default_rng(0),
        max_rss_gb=0.0,
        rss_log_interval_steps=0,
        episode_index=0,
    )

    per_step = episode["cem_plan"]["per_step"]
    assert latent_encode_calls == [False]
    assert per_step
    assert not isinstance(per_step[0]["latent_pred"], list)
    assert int(np.asarray(per_step[0]["latent_pred"]).size) == 16
    assert int(per_step[0]["latent_pred_dim"]) == 300


def test_main_cleans_up_and_closes_env_on_guard_failure(monkeypatch):
    created_envs: list[Any] = []
    cleanup_paths: list[Path] = []

    class _FakeEnv:
        def __init__(self):
            self.action_space = types.SimpleNamespace(shape=(4,))
            self.render_mode = None
            self.closed = False
            created_envs.append(self)

        def set_task(self, _task):
            return None

        def close(self):
            self.closed = True

    class _FakeML1:
        def __init__(self, task: str, seed: int):
            del seed
            self.train_classes = {task: _FakeEnv}
            self.train_tasks = [object()]

    monkeypatch.setitem(sys.modules, "metaworld", types.SimpleNamespace(ML1=_FakeML1))
    monkeypatch.setattr(jepa_export, "_try_load_smolvla_exec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_try_load_wm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        jepa_export,
        "rollout_episode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rss guard tripped")),
    )
    original_cleanup = jepa_export._cleanup_episode_shards

    def _cleanup_spy(path: Path) -> None:
        cleanup_paths.append(Path(path))
        original_cleanup(path)

    monkeypatch.setattr(jepa_export, "_cleanup_episode_shards", _cleanup_spy)
    monkeypatch.setenv("SMOLVLA_JEPA_EXPORT_SKIP_WM", "1")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "jepa_cem_paired_pushv3_export.py",
                "--out",
                str(out_dir),
                "--episodes",
                "1",
                "--max-steps",
                "1",
                "--device",
                "cpu",
            ],
        )
        rc = jepa_export.main()

    assert rc == 1
    assert cleanup_paths
    assert any(path.name.startswith(".episodes_staging_") for path in cleanup_paths)
    assert created_envs
    assert created_envs[0].closed


def test_main_accepts_cli_episodes_per_shard_and_wires_writer(monkeypatch):
    captured: dict[str, int] = {}

    class _FakeWriter:
        def __init__(self, out_dir: Path, episodes_per_shard: int = 1):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            captured["episodes_per_shard"] = int(episodes_per_shard)

        def write_episode(self, _episode: dict[str, Any]):
            return None

        def finalize(self):
            return []

    class _FakeEnv:
        def __init__(self):
            self.action_space = types.SimpleNamespace(shape=(4,))
            self.render_mode = None

        def set_task(self, _task):
            return None

        def close(self):
            return None

    class _FakeML1:
        def __init__(self, task: str, seed: int):
            del seed
            self.train_classes = {task: _FakeEnv}
            self.train_tasks = [object()]

    monkeypatch.setitem(sys.modules, "metaworld", types.SimpleNamespace(ML1=_FakeML1))
    monkeypatch.setattr(jepa_export, "_try_load_smolvla_exec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_try_load_wm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_enforce_export_quality_gates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_promote_episode_shards", lambda _src, dst: Path(dst).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(jepa_export, "EpisodeShardWriter", _FakeWriter)
    monkeypatch.setattr(
        jepa_export,
        "rollout_episode",
        lambda *_args, **_kwargs: {
            "meta": {"episode_index": 0},
            "images": [np.zeros((4, 4, 3), dtype=np.uint8)],
            "cem_plan": {"per_step": [{"latent_pred_dim": 256, "policy_source": "cem_mpc_wm", "planner_metadata": {}}]},
        },
    )
    monkeypatch.setenv("SMOLVLA_JEPA_EXPORT_SKIP_WM", "1")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "jepa_cem_paired_pushv3_export.py",
                "--out",
                str(out_dir),
                "--episodes",
                "1",
                "--max-steps",
                "1",
                "--device",
                "cpu",
                "--episodes-per-shard",
                "3",
            ],
        )
        rc = jepa_export.main()

        assert rc == 0
        assert captured["episodes_per_shard"] == 3
        manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
        assert manifest["episodes_per_shard"] == 3


def test_main_uses_env_default_episodes_per_shard_when_flag_absent(monkeypatch):
    captured: dict[str, int] = {}

    class _FakeWriter:
        def __init__(self, out_dir: Path, episodes_per_shard: int = 1):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            captured["episodes_per_shard"] = int(episodes_per_shard)

        def write_episode(self, _episode: dict[str, Any]):
            return None

        def finalize(self):
            return []

    class _FakeEnv:
        def __init__(self):
            self.action_space = types.SimpleNamespace(shape=(4,))
            self.render_mode = None

        def set_task(self, _task):
            return None

        def close(self):
            return None

    class _FakeML1:
        def __init__(self, task: str, seed: int):
            del seed
            self.train_classes = {task: _FakeEnv}
            self.train_tasks = [object()]

    monkeypatch.setitem(sys.modules, "metaworld", types.SimpleNamespace(ML1=_FakeML1))
    monkeypatch.setattr(jepa_export, "_try_load_smolvla_exec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_try_load_wm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_enforce_export_quality_gates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_promote_episode_shards", lambda _src, dst: Path(dst).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(jepa_export, "EpisodeShardWriter", _FakeWriter)
    monkeypatch.setattr(
        jepa_export,
        "rollout_episode",
        lambda *_args, **_kwargs: {
            "meta": {"episode_index": 0},
            "images": [np.zeros((4, 4, 3), dtype=np.uint8)],
            "cem_plan": {"per_step": [{"latent_pred_dim": 256, "policy_source": "cem_mpc_wm", "planner_metadata": {}}]},
        },
    )
    monkeypatch.setenv("SMOLVLA_JEPA_EXPORT_SKIP_WM", "1")
    monkeypatch.setenv("SMOLVLA_JEPA_EXPORT_EPISODES_PER_SHARD", "5")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "jepa_cem_paired_pushv3_export.py",
                "--out",
                str(out_dir),
                "--episodes",
                "1",
                "--max-steps",
                "1",
                "--device",
                "cpu",
            ],
        )
        rc = jepa_export.main()

        assert rc == 0
        assert captured["episodes_per_shard"] == 5
        manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
        assert manifest["episodes_per_shard"] == 5


def test_main_manifest_shard_metadata_uses_promoted_episode_paths(monkeypatch):
    class _FakeEnv:
        def __init__(self):
            self.action_space = types.SimpleNamespace(shape=(4,))
            self.render_mode = None

        def set_task(self, _task):
            return None

        def close(self):
            return None

    class _FakeML1:
        def __init__(self, task: str, seed: int):
            del seed
            self.train_classes = {task: _FakeEnv}
            self.train_tasks = [object()]

    monkeypatch.setitem(sys.modules, "metaworld", types.SimpleNamespace(ML1=_FakeML1))
    monkeypatch.setattr(jepa_export, "_try_load_smolvla_exec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_try_load_wm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jepa_export, "_enforce_export_quality_gates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        jepa_export,
        "rollout_episode",
        lambda *_args, **_kwargs: {
            "meta": {"episode_index": 0},
            "images": [np.zeros((4, 4, 3), dtype=np.uint8)],
            "cem_plan": {"per_step": [{"latent_pred_dim": 256, "policy_source": "cem_mpc_wm", "planner_metadata": {}}]},
        },
    )
    monkeypatch.setenv("SMOLVLA_JEPA_EXPORT_SKIP_WM", "1")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "jepa_cem_paired_pushv3_export.py",
                "--out",
                str(out_dir),
                "--episodes",
                "2",
                "--episodes-per-shard",
                "2",
                "--max-steps",
                "1",
                "--device",
                "cpu",
            ],
        )
        rc = jepa_export.main()

        assert rc == 0
        manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
        assert manifest["shard_count"] == 2
        assert manifest["complete_episodes"] == 2
        assert len(manifest["shard_files"]) == 2
        for rel in manifest["shard_files"]:
            assert rel.startswith("episodes/episode_")
            assert (out_dir / rel).is_file()


class ExporterMemoryBoundedTests(unittest.TestCase):
    def test_incremental_metrics_match_legacy_metrics(self):
        episodes = [
            {
                "meta": {"policy": "cem_primary"},
                "images": [np.zeros((4, 4, 3), dtype=np.uint8)],
                "cem_plan": {
                    "per_step": [
                        {
                            "policy_source": "cem_mpc_wm",
                            "planner_metadata": {"wm_step_error": False, "policy_exec_error": False},
                            "latent_pred_dim": 256,
                        },
                        {
                            "policy_source": "heuristic_fallback",
                            "planner_metadata": {"wm_step_error": True, "policy_exec_error": False},
                        },
                    ]
                },
            },
            {
                "meta": {"policy": "heuristic"},
                "images": [],
                "cem_plan": {
                    "per_step": [
                        {
                            "policy_source": "smolvla",
                            "planner_metadata": {"wm_step_error": False, "policy_exec_error": True},
                        }
                    ]
                },
            },
            {
                "meta": {"policy": "cem_primary"},
                "images": [np.ones((2, 2, 3), dtype=np.uint8)],
                "cem_plan": {"per_step": ["invalid-row", {"policy_source": "cem_mpc_wm", "planner_metadata": {}}]},
            },
            {
                "meta": {"policy": "heuristic_fallback"},
                "images": None,
                "cem_plan": {},
            },
        ]

        legacy_metrics = jepa_export._compute_export_quality_metrics(episodes)
        acc = jepa_export.ExportQualityAccumulator()
        for episode in episodes:
            acc.update(episode)
        incremental_metrics = acc.to_metrics()

        assert set(legacy_metrics.keys()) == set(incremental_metrics.keys())
        for key, legacy_value in legacy_metrics.items():
            assert abs(float(legacy_value) - float(incremental_metrics[key])) < 1e-12
