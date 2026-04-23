from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "segment_grpo"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def campaign_module():
    return importlib.import_module("run_all60_frame50_k3")


def _make_oracle_run(tmp_path: Path, have_frames_for: list[int], goal_frame_idx: int) -> Path:
    run = tmp_path / "oracle_run"
    for ep in have_frames_for:
        ep_dir = run / "frames" / f"episode_{ep:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / f"frame_{goal_frame_idx - 1:06d}.png").write_bytes(b"\x89PNG\r\n")
    return run


def _install_fake_oracle(monkeypatch: pytest.MonkeyPatch, campaign_module, oracle_run: Path) -> None:
    monkeypatch.setattr(
        campaign_module,
        "resolve_latest_oracle_pushv3_run",
        lambda *a, **k: oracle_run,
    )


class _FakeCompleted:
    def __init__(self, rc: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


def test_default_run_dir_respects_run_name_prefix_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign_module
):
    monkeypatch.setenv("RUN_NAME_PREFIX", "mt10")
    oracle_run = tmp_path / "oracle_run"
    oracle_run.mkdir()
    args = campaign_module._parse_args(
        [
            "--seed-base",
            "1000",
            "--episodes",
            "1",
            "--goal-frame-index",
            "50",
            "--output-root",
            str(tmp_path / "out"),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--oracle-run-root",
            str(oracle_run),
        ]
    )
    run_dir = campaign_module._resolve_run_dir(args)
    assert run_dir.name.startswith("mt10_")
    assert "_tpush_v3" in run_dir.name


def test_seed_derivation_and_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign_module):
    oracle_run = _make_oracle_run(tmp_path, have_frames_for=[0, 1, 2], goal_frame_idx=50)
    _install_fake_oracle(monkeypatch, campaign_module, oracle_run)

    calls: list[tuple[int, int]] = []

    def fake_run(argv, capture_output=True, text=True, timeout=None, **kwargs):  # noqa: ARG001
        out_idx = argv.index("--output-json") + 1
        ep_idx = argv.index("--episode-index") + 1
        seed_idx = argv.index("--reset-seed") + 1
        Path(argv[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[out_idx]).write_text(
            json.dumps(
                {
                    "latent_scores": [0.1],
                    "selected_scores": [0.1],
                    "candidate_distances": [[0.1, 0.2, 0.3]],
                    "selected_indices": [0],
                    "steps": 50,
                    "done": True,
                    "goal_source": str(argv[argv.index("--oracle-run-root") + 1]),
                }
            )
        )
        calls.append((int(argv[ep_idx]), int(argv[seed_idx])))
        return _FakeCompleted(rc=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)

    run_dir = tmp_path / "out"
    rc = campaign_module.main(
        [
            "--seed-base",
            "1000",
            "--episode-start",
            "0",
            "--episodes",
            "3",
            "--goal-frame-index",
            "50",
            "--output-root",
            str(run_dir),
            "--run-name",
            "test",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--oracle-run-root",
            str(oracle_run),
            "--child-script",
            str(Path("/tmp/noop.py")),
            "--no-comparison-strip-overlay",
        ]
    )
    assert rc == 0
    assert calls == [(0, 1000), (1, 1001), (2, 1002)]
    manifest = json.loads((run_dir / "test" / "segment_grpo_manifest.json").read_text())
    assert manifest["counts"] == {
        "ok": 3,
        "resume_skip": 0,
        "missing_goal": 0,
        "missing_prefetch": 0,
        "failed": 0,
    }
    assert len((run_dir / "test" / "episodes.jsonl").read_text().splitlines()) == 3


def test_missing_goal_frame_skips_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign_module
):
    oracle_run = _make_oracle_run(tmp_path, have_frames_for=[0, 2], goal_frame_idx=50)
    _install_fake_oracle(monkeypatch, campaign_module, oracle_run)

    calls: list[int] = []

    def fake_run(argv, capture_output=True, text=True, timeout=None, **kwargs):  # noqa: ARG001
        ep_idx = int(argv[argv.index("--episode-index") + 1])
        calls.append(ep_idx)
        out_idx = argv.index("--output-json") + 1
        Path(argv[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[out_idx]).write_text("{}")
        return _FakeCompleted(rc=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)

    run_dir = tmp_path / "out"
    rc = campaign_module.main(
        [
            "--seed-base",
            "1000",
            "--episode-start",
            "0",
            "--episodes",
            "3",
            "--goal-frame-index",
            "50",
            "--output-root",
            str(run_dir),
            "--run-name",
            "test",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--oracle-run-root",
            str(oracle_run),
            "--child-script",
            str(Path("/tmp/noop.py")),
            "--no-comparison-strip-overlay",
        ]
    )
    assert rc == 0
    assert calls == [0, 2], "episode 1 has no goal frame, must not be spawned"
    skipped = (run_dir / "test" / "skipped_episodes.jsonl").read_text().splitlines()
    assert len(skipped) == 1
    rec = json.loads(skipped[0])
    assert rec["target_episode_index"] == 1
    assert rec["reason"] == "goal_frame_missing"


def test_resume_skips_existing_episode_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign_module
):
    oracle_run = _make_oracle_run(tmp_path, have_frames_for=[0, 1], goal_frame_idx=50)
    _install_fake_oracle(monkeypatch, campaign_module, oracle_run)

    run_dir = tmp_path / "out" / "test"
    run_dir.mkdir(parents=True)
    (run_dir / "out_episode_0000.json").write_text("{}")

    calls: list[int] = []

    def fake_run(argv, capture_output=True, text=True, timeout=None, **kwargs):  # noqa: ARG001
        calls.append(int(argv[argv.index("--episode-index") + 1]))
        Path(argv[argv.index("--output-json") + 1]).write_text("{}")
        return _FakeCompleted(rc=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)

    rc = campaign_module.main(
        [
            "--seed-base",
            "1000",
            "--episode-start",
            "0",
            "--episodes",
            "2",
            "--goal-frame-index",
            "50",
            "--output-root",
            str(tmp_path / "out"),
            "--run-name",
            "test",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--oracle-run-root",
            str(oracle_run),
            "--child-script",
            str(Path("/tmp/noop.py")),
            "--no-comparison-strip-overlay",
        ]
    )
    assert rc == 0
    assert calls == [1], "pre-existing ep0 JSON must prevent subprocess spawn"


def test_failure_continues_by_default_and_aborts_on_stop_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign_module
):
    oracle_run = _make_oracle_run(tmp_path, have_frames_for=[0, 1, 2], goal_frame_idx=50)
    _install_fake_oracle(monkeypatch, campaign_module, oracle_run)

    def fake_run(argv, capture_output=True, text=True, timeout=None, **kwargs):  # noqa: ARG001
        ep = int(argv[argv.index("--episode-index") + 1])
        if ep == 1:
            return _FakeCompleted(rc=2, stderr="boom")
        Path(argv[argv.index("--output-json") + 1]).write_text("{}")
        return _FakeCompleted(rc=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)

    base = [
        "--seed-base",
        "1000",
        "--episode-start",
        "0",
        "--episodes",
        "3",
        "--goal-frame-index",
        "50",
        "--artifacts-root",
        str(tmp_path / "artifacts"),
        "--oracle-run-root",
        str(oracle_run),
        "--child-script",
        str(Path("/tmp/noop.py")),
        "--no-comparison-strip-overlay",
    ]

    rc = campaign_module.main(base + ["--output-root", str(tmp_path / "out_a"), "--run-name", "a"])
    assert rc == 0
    manifest_a = json.loads((tmp_path / "out_a" / "a" / "segment_grpo_manifest.json").read_text())
    assert manifest_a["counts"]["failed"] == 1
    assert manifest_a["counts"]["ok"] == 2

    rc2 = campaign_module.main(
        base + ["--output-root", str(tmp_path / "out_b"), "--run-name", "b", "--stop-on-error"]
    )
    assert rc2 == 1
    manifest_b = json.loads((tmp_path / "out_b" / "b" / "segment_grpo_manifest.json").read_text())
    assert manifest_b["counts"]["failed"] == 1
    assert manifest_b["counts"]["ok"] == 1


def test_manifest_counts_and_preserves_latent_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, campaign_module
):
    oracle_run = _make_oracle_run(tmp_path, have_frames_for=[0, 1], goal_frame_idx=50)
    _install_fake_oracle(monkeypatch, campaign_module, oracle_run)

    def fake_run(argv, capture_output=True, text=True, timeout=None, **kwargs):  # noqa: ARG001
        out_path = Path(argv[argv.index("--output-json") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "latent_scores": [0.42],
                    "selected_scores": [0.42],
                    "candidate_distances": [[0.4, 0.5, 0.6]],
                    "selected_indices": [2],
                    "steps": 50,
                    "done": True,
                    "goal_source": "x",
                }
            )
        )
        return _FakeCompleted(rc=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)

    rc = campaign_module.main(
        [
            "--seed-base",
            "1000",
            "--episode-start",
            "0",
            "--episodes",
            "2",
            "--goal-frame-index",
            "50",
            "--output-root",
            str(tmp_path / "out"),
            "--run-name",
            "test",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--oracle-run-root",
            str(oracle_run),
            "--child-script",
            str(Path("/tmp/noop.py")),
            "--no-comparison-strip-overlay",
        ]
    )
    assert rc == 0
    lines = (tmp_path / "out" / "test" / "episodes.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    for r in records:
        assert r["latent_scores"] == [0.42]
        assert r["selected_scores"] == [0.42]
        assert r["candidate_distances"] == [[0.4, 0.5, 0.6]]
