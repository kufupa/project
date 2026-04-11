from __future__ import annotations

from typing import Any


def pick_best_episode(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return best SmolVLA episode row.

    Priority:
    1) highest sum_reward
    2) highest max_reward
    3) earliest episode_index for determinism
    """
    if not episodes:
        raise ValueError("No episodes to rank")

    normalized: list[tuple[dict[str, Any], float, float, int]] = []
    required_fields = ("episode_index", "sum_reward", "max_reward")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"episode row at index {index} must be an object.")

        missing = [field for field in required_fields if field not in episode]
        if missing:
            raise ValueError(
                f"episode row at index {index} missing required field(s): {', '.join(missing)}"
            )

        try:
            episode_index = int(episode["episode_index"])
            sum_reward = float(episode["sum_reward"])
            max_reward = float(episode["max_reward"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"episode row at index {index} has invalid reward/episode_index type.") from exc

        normalized.append((episode, sum_reward, max_reward, episode_index))

    return max(
        normalized,
        key=lambda packed: (packed[1], packed[2], -packed[3]),
    )[0]
