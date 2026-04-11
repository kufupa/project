import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_pushv3_eval.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_pushv3_eval", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_episode_sort_key_prefers_higher_sum_reward_for_ties():
    mod = _load_module()
    low = {"episode_index": 0, "max_reward": 10.0, "sum_reward": 550.0}
    high = {"episode_index": 7, "max_reward": 10.0, "sum_reward": 650.0}

    ranked = sorted([low, high], key=mod._episode_sort_key)
    assert ranked[0]["episode_index"] == 7


def test_episode_sort_key_handles_missing_sum_reward():
    mod = _load_module()
    missing = {"episode_index": 1, "max_reward": 10.0, "sum_reward": None}
    present = {"episode_index": 2, "max_reward": 10.0, "sum_reward": 1.0}

    ranked = sorted([missing, present], key=mod._episode_sort_key)
    assert ranked[0]["episode_index"] == 2
