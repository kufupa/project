from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts" / "eggroll"


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_eggroll_pbs_scripts_use_cx3_contracts() -> None:
    for name in (
        "submit_phase50_eggroll_smoke.pbs",
        "submit_phase50_eggroll_calibrate.pbs",
        "submit_phase50_eggroll_train_from_calibration.pbs",
        "submit_phase50_eggroll_eval_sweep.pbs",
    ):
        text = _read(name)
        assert "#SBATCH" not in text
        assert "PBS_O_WORKDIR" in text
        assert "module load tools/prod" in text
        assert "module load Python/3.12.3-GCCcore-13.3.0" in text
        assert "module load Mesa/24.1.3-GCCcore-13.3.0" in text
        assert "MUJOCO_GL" in text and "osmesa" in text
        assert "PYOPENGL_PLATFORM" in text and "osmesa" in text
        assert "LIBGL_ALWAYS_SOFTWARE" in text
        assert "gpu_type=RTX6000" in text
        assert "/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python" in text


def test_resource_requests_match_measured_usage() -> None:
    train = _read("submit_phase50_eggroll_train_from_calibration.pbs")
    eval_text = _read("submit_phase50_eggroll_eval_sweep.pbs")
    calibrate = _read("submit_phase50_eggroll_calibrate.pbs")

    assert "#PBS -l select=1:ncpus=16:mem=48gb:ngpus=1:gpu_type=RTX6000" in train
    assert "#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000" in eval_text
    assert "#PBS -l walltime=00:35:00" in eval_text
    assert "#PBS -l select=1:ncpus=16:mem=48gb:ngpus=1:gpu_type=RTX6000" in calibrate


def test_train_and_eval_contracts() -> None:
    calib = _read("submit_phase50_eggroll_calibrate.pbs")
    train = _read("submit_phase50_eggroll_train_from_calibration.pbs")
    eval_text = _read("submit_phase50_eggroll_eval_sweep.pbs")

    assert 'rm -rf "${out}"' in calib
    assert 'mkdir -p "${out}"' in calib
    assert "calibration_summary.json" in train
    assert "no usable EGGROLL_PYTHON" in calib
    assert "no usable EGGROLL_PYTHON" in train
    assert "no usable EGGROLL_PYTHON" in eval_text
    assert "selected_population_batch_size" in train
    assert "ITER_SECONDS" in train
    assert "budget_s=10*3600" in train
    assert "PHASE50_EGGROLL_TRAIN_DONE" in train
    assert "--save-every" in train
    assert "--video-every" in train
    assert "PHASE50_FITNESS_SHAPING" in train
    assert "--fitness-shaping" in train

    assert "scripts/grpo/eval_phase12_checkpoint_sweep.py" in eval_text
    assert "inprocess_vector" in eval_text
    assert "vector_async" in eval_text
    for text in (calib, train, eval_text, _read("submit_phase50_eggroll_smoke.pbs")):
        assert 'SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"' in text
    assert '--n-envs "${PHASE50_EVAL_N_ENVS:-3}"' in eval_text
    assert "PHASE50_EGGROLL_SEEDED_EVAL_SWEEP_DONE" in eval_text
    assert "MAX_UPDATE" in eval_text
    assert "PHASE50_ALLOW_BASE_EVAL" in eval_text
