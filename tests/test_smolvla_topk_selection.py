from src.smolvla_pipeline.topk_selection import pick_best_episode


def test_pick_best_episode_prefers_sum_then_max_then_earliest_index():
    episodes = [
        {"episode_index": 2, "sum_reward": 5.0, "max_reward": 1.0},
        {"episode_index": 1, "sum_reward": 6.0, "max_reward": 0.9},
        {"episode_index": 3, "sum_reward": 6.0, "max_reward": 1.2},
    ]
    winner = pick_best_episode(episodes)
    assert winner["episode_index"] == 3


def test_pick_best_episode_prefers_earlier_when_tied():
    episodes = [
        {"episode_index": 5, "sum_reward": 1.0, "max_reward": 0.0},
        {"episode_index": 2, "sum_reward": 1.0, "max_reward": 0.0},
        {"episode_index": 1, "sum_reward": 1.0, "max_reward": 0.0},
    ]
    winner = pick_best_episode(episodes)
    assert winner["episode_index"] == 1


def test_pick_best_episode_raises_on_missing_required_field():
    episodes = [
        {"episode_index": 1, "sum_reward": 1.0},
    ]
    try:
        pick_best_episode(episodes)
    except ValueError as exc:
        assert "missing required field" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pick_best_episode_raises_on_invalid_value_type():
    episodes = [
        {"episode_index": 1, "sum_reward": "bad", "max_reward": 0.0},
    ]
    try:
        pick_best_episode(episodes)
    except ValueError as exc:
        assert "invalid reward/episode_index type" in str(exc)
    else:
        raise AssertionError("expected ValueError")
