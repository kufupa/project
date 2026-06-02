from __future__ import annotations

from scripts.grpo.eval_phase12_checkpoint_sweep import _parse_update_list


def test_parse_update_list_accepts_colons_commas_and_dedupes() -> None:
    assert _parse_update_list("2:4,5;5:10") == [2, 4, 5, 10]
    assert _parse_update_list("") == []

