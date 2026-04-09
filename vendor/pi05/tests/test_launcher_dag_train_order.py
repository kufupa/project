import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "smolvla_workflow_launcher.py"
SPEC = importlib.util.spec_from_file_location("smolvla_workflow_launcher", MODULE)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class LauncherDagTrainOrderTests(unittest.TestCase):
    def test_stage_scripts_tail_matches_target_order(self):
        self.assertEqual(
            launcher.STAGE_SCRIPTS[5:11],
            [
                "stage04_bridge_dataset_build.slurm",
                "stage06_train_stageB_jepa_mix.slurm",
                "stage07_vgg_gatecheck.slurm",
                "stage08_train_stageC_vgg_aux.slurm",
                "stage05_train_stageA_real_only.slurm",
                "stage09_final_eval_and_bundle.slurm",
            ],
        )

    def test_branch_parallel_dependencies_match_train_order_contract(self):
        calls = []

        def _fake_submit(path: Path, dependency=None, *, requires_gpu=False):
            job_id = f"{path.stem}_jid"
            calls.append((path.name, dependency, job_id, requires_gpu))
            return job_id

        with mock.patch.object(launcher, "submit_stage", side_effect=_fake_submit):
            ids_order = launcher.submit_workflow_branch_parallel(map_out=None)

        stage_to_dep = {name: dep for name, dep, _jid, _gpu in calls}
        stage_to_jid = {name: jid for name, _dep, jid, _gpu in calls}

        self.assertEqual(
            stage_to_dep["stage06_train_stageB_jepa_mix.slurm"],
            "stage02_baseline_pushv3_eval_jid:stage04_bridge_dataset_build_jid",
        )
        self.assertEqual(
            stage_to_dep["stage07_vgg_gatecheck.slurm"],
            "stage01b_install_metaworld_jid",
        )
        self.assertEqual(
            stage_to_dep["stage08_train_stageC_vgg_aux.slurm"],
            "stage07_vgg_gatecheck_jid:stage04_bridge_dataset_build_jid",
        )
        self.assertEqual(
            stage_to_dep["stage05_train_stageA_real_only.slurm"],
            "stage06_train_stageB_jepa_mix_jid:stage08_train_stageC_vgg_aux_jid",
        )
        self.assertEqual(
            stage_to_dep["stage09_final_eval_and_bundle.slurm"],
            "stage05_train_stageA_real_only_jid:stage06_train_stageB_jepa_mix_jid:stage08_train_stageC_vgg_aux_jid",
        )

        expected_ids_order = [stage_to_jid[stage] for stage in launcher.STAGE_SCRIPTS]
        self.assertEqual(ids_order, expected_ids_order)


if __name__ == "__main__":
    unittest.main()
