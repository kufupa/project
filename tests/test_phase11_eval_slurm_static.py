from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVAL25 = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_eval25_a30.slurm"


def test_flow_sde_chunk_eval25_absolutizes_paths_before_rlinf_chdir() -> None:
    text = _EVAL25.read_text(encoding="utf-8")
    assert 'CKPT_DIR="$(realpath "${1:?checkpoint dir}")"' in text
    assert 'OUT_DIR_RAW="${2:?eval output dir}"' in text
    assert 'mkdir -p "${OUT_DIR_RAW}"' in text
    assert 'OUT_DIR="$(realpath "${OUT_DIR_RAW}")"' in text
    assert '--checkpoint-dir "${CKPT_DIR}"' in text
    subprocess.run(["bash", "-n", str(_EVAL25)], check=True, cwd=str(_REPO_ROOT))
