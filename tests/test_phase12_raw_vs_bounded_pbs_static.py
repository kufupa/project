from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_raw_vs_bounded_decode_pbs_contract() -> None:
    text = (PROJECT / "scripts" / "grpo" / "submit_phase12_raw_vs_bounded_decode_6ep.pbs").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1:gpu_type=RTX6000" in text
    assert "module load tools/prod" in text
    assert "module load Python/3.12.3-GCCcore-13.3.0" in text
    assert "module load Mesa/24.1.3-GCCcore-13.3.0" in text
    assert 'slurm_resolve_project_root "scripts/grpo/compare_phase12_raw_vs_bounded_decode.py"' in text
    assert "slurm_export_pythonpath" in text
    assert 'export MUJOCO_GL="${MUJOCO_GL:-osmesa}"' in text
    assert 'export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"' in text
    assert 'export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"' in text
    assert 'export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-0}"' in text
    assert 'NUM_EPISODES="${PHASE12_NUM_EPISODES:-6}"' in text
    assert 'CHUNK_LEN="${PHASE12_CHUNK_LEN:-50}"' in text
    assert "--num-episodes \"${NUM_EPISODES}\"" in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert "compare_summary.json" in text
    assert "real_raw_bounded_decode_strip.png" in text
    assert "actions_raw_bounded_env.npz" in text
    assert "PHASE12_RAW_VS_BOUNDED_DECODE_DONE" in text
