from __future__ import annotations

from smolvla_grpo.eggroll_rollout import build_eggroll_reset_seeds


def test_shared_per_iteration_seed_mode_uses_same_seed_for_all_members() -> None:
    seeds = build_eggroll_reset_seeds(
        train_seed_base=2000,
        iteration=3,
        rollout_seed_offset=1,
        member_ids=[0, 1, 2, 3],
        seed_mode="shared_per_iteration",
    )

    assert seeds == [2007, 2007, 2007, 2007]


def test_member_seed_mode_preserves_existing_member_specific_behavior() -> None:
    seeds = build_eggroll_reset_seeds(
        train_seed_base=2000,
        iteration=3,
        rollout_seed_offset=1,
        member_ids=[0, 1, 2, 3],
        seed_mode="member_offset",
    )

    assert seeds == [
        2000 + 3 * 100003 + 1 * 1009 + 0,
        2000 + 3 * 100003 + 1 * 1009 + 1,
        2000 + 3 * 100003 + 1 * 1009 + 2,
        2000 + 3 * 100003 + 1 * 1009 + 3,
    ]


def test_shared_seed_batch_rotates_between_episode_repeats_and_iterations() -> None:
    first = build_eggroll_reset_seeds(
        train_seed_base=2000,
        iteration=0,
        rollout_seed_offset=0,
        member_ids=[0, 1],
        seed_mode="shared_per_iteration",
    )
    second_episode = build_eggroll_reset_seeds(
        train_seed_base=2000,
        iteration=0,
        rollout_seed_offset=1,
        member_ids=[0, 1],
        seed_mode="shared_per_iteration",
    )
    next_iter = build_eggroll_reset_seeds(
        train_seed_base=2000,
        iteration=1,
        rollout_seed_offset=0,
        member_ids=[0, 1],
        seed_mode="shared_per_iteration",
    )

    assert first == [2000, 2000]
    assert second_episode == [2001, 2001]
    assert next_iter == [2002, 2002]
