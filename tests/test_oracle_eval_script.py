import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_oracle_eval_module():
    fake_imageio_v2 = types.ModuleType("imageio.v2")
    fake_imageio_v2.imwrite = lambda *_args, **_kwargs: None
    fake_imageio_v2.mimsave = lambda *_args, **_kwargs: None
    fake_imageio = types.ModuleType("imageio")
    fake_imageio.v2 = fake_imageio_v2

    fake_metaworld = types.ModuleType("metaworld")
    fake_policies = types.ModuleType("metaworld.policies")
    fake_policies.ENV_POLICY_MAP = {}
    fake_metaworld.policies = fake_policies

    sys.modules.setdefault("imageio", fake_imageio)
    sys.modules.setdefault("imageio.v2", fake_imageio_v2)
    sys.modules.setdefault("metaworld", fake_metaworld)
    sys.modules.setdefault("metaworld.policies", fake_policies)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "oracle"
        / "run_metaworld_oracle_eval.py"
    )
    spec = importlib.util.spec_from_file_location("oracle_eval_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_match_smolvla_parity(monkeypatch):
    module = _load_oracle_eval_module()

    monkeypatch.delenv("ORACLE_METAWORLD_CAMERA_NAME", raising=False)
    monkeypatch.delenv("ORACLE_FLIP_CORNER2", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_metaworld_oracle_eval.py", "--output-dir", "/tmp/oracle_eval_test"],
    )
    args = module.parse_args()

    assert args.max_steps == 120
    assert args.camera_name == "corner2"
    assert args.flip_corner2 == "true"


def test_render_rgb_frame_flips_corner2():
    module = _load_oracle_eval_module()

    class _Env:
        def render(self, camera_name=None):
            _ = camera_name
            return np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            )

    frame = module._render_rgb_frame(_Env(), camera_name="corner2", flip_corner2=True)
    assert frame is not None
    expected = np.array(
        [
            [[10, 11, 12], [7, 8, 9]],
            [[4, 5, 6], [1, 2, 3]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(frame, expected)


def _load_oracle_parity_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "oracle"
        / "run_oracle_parity_1ep.py"
    )
    spec = importlib.util.spec_from_file_location("run_oracle_parity_1ep", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_parity_task_slug():
    mod = _load_oracle_parity_module()
    assert mod._task_slug("push-v3") == "push_v3"


def test_oracle_parity_default_seed_matches_phase06_campaign(monkeypatch):
    mod = _load_oracle_parity_module()
    monkeypatch.delenv("ORACLE_PARITY_SEED", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_oracle_parity_1ep.py", "--output-dir", "/tmp/oracle_parity_out"],
    )
    args = mod.parse_args()
    assert args.seed == 1000
    assert args.save_frames == "true"
    assert args.video == "true"
    assert args.max_steps == 120


def test_oracle_parity_seed_from_env(monkeypatch):
    mod = _load_oracle_parity_module()
    monkeypatch.setenv("ORACLE_PARITY_SEED", "1044")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_oracle_parity_1ep.py", "--output-dir", "/tmp/oracle_parity_out"],
    )
    args = mod.parse_args()
    assert args.seed == 1044


def test_oracle_parity_unique_run_dir_naming(tmp_path: Path):
    mod = _load_oracle_parity_module()
    run_dir = mod._unique_run_dir(
        tmp_path, episodes=1, task="push-v3", seed=1000
    )
    assert run_dir.is_dir()
    name = run_dir.name
    assert name.startswith("run_")
    assert "_ep1_voracle_tpush_v3_s1000_r" in name
