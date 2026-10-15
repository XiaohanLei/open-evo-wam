import sys
import tempfile
import unittest
from pathlib import Path
from evo_wam.pipeline import run_stage
from evo_wam.io import read


class NativeTrainingReceiptTests(unittest.TestCase):
    def config(self, root, mutate=False):
        data = root / "data.bin"
        data.write_bytes(b"original input")
        recipe = root / "recipe.json"
        recipe.write_text("{}")
        source = root / "source"
        source.mkdir()
        script = source / "fixture.py"
        script.write_text(
            "from pathlib import Path\nimport os\np=Path(os.environ['EVO_WAM_STAGE_OUTPUT'])/'checkpoint'\np.mkdir()\n(p/'weights.bin').write_bytes(b'fixture checkpoint')\n"
            + (f"Path({str(data)!r}).write_bytes(b'changed input')\n" if mutate else "")
        )
        return dict(
            run_dir=str(root / "run"),
            stages={
                "train": dict(
                    adapter="evo_wam.adapters.training:run",
                    requires_previous=False,
                    source_root=str(source),
                    train_argv=[sys.executable, str(script)],
                    checkpoint_output="${STAGE_DIR}/checkpoint",
                    dependency_paths=[str(data)],
                    recipe_artifacts=[str(recipe)],
                )
            },
        )

    def test_receipt_binds_recipe_and_actual_training_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = self.config(root)
            self.assertEqual(run_stage(config, "train")["status"], "complete")
            result = read(root / "run/R01/train/result.json")
            self.assertIn(str(root / "recipe.json"), result["artifacts"])
            self.assertEqual(result["dependency_paths"], [str(root / "data.bin")])
            self.assertEqual(run_stage(config, "train")["status"], "reused")
            (root / "data.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                run_stage(config, "train")

    def test_input_mutation_during_command_cannot_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = self.config(root, mutate=True)
            with self.assertRaisesRegex(ValueError, "input changed during"):
                run_stage(config, "train")
            self.assertFalse((root / "run/R01/train/result.json").exists())
            self.assertEqual(
                read(root / "run/R01/train/stage.json")["status"], "failed"
            )
