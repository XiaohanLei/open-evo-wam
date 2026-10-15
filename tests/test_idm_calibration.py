import tempfile
import unittest
from pathlib import Path
from evo_wam.io import read, write, sha
from evo_wam.adapters.idm_calibration import calibrate


class IDMCalibrationTests(unittest.TestCase):
    def test_heldout_population_and_exact_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            population = []
            scores = []
            for i, value in enumerate((1.0, 3.0, 99.0)):
                row = dict(
                    uid=str(i),
                    task="task",
                    config="Clean",
                    outcome="success" if i < 2 else "fail",
                    length=64,
                )
                population.append(dict(row, partition="calibration"))
                source = root / f"source-{i}.json"
                write(source, dict(mse=value))
                scores.append(
                    dict(
                        row,
                        mse=value,
                        status="complete",
                        checkpoint_export_sha256="model",
                        source=str(source),
                        source_sha256=sha(source),
                    )
                )
            manifest = root / "population.json"
            scorefile = root / "scores.json"
            write(manifest, population)
            write(scorefile, scores)
            context = dict(
                stage="idm-calibrate",
                round=1,
                output=str(root / "out"),
                spec=dict(
                    interface="dense",
                    manifest=str(manifest),
                    scores=str(scorefile),
                    checkpoint_export_sha256="model",
                ),
            )
            calibrate(context)
            result = read(root / "out/THRESHOLDS.json")
            self.assertEqual(result["successes"], 2)
            self.assertAlmostEqual(result["schedule"][0]["threshold"], 2.8)
            self.assertEqual(result["schedule"][-1]["threshold"], 1.0)
            write(scorefile, scores[:-1])
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                calibrate(context)
            write(scorefile, scores)
            population[0]["partition"] = "final"
            write(manifest, population)
            with self.assertRaisesRegex(ValueError, "held-out"):
                calibrate(context)
            population[0]["partition"] = "calibration"
            write(manifest, population)
            scores[0]["checkpoint_export_sha256"] = "other"
            write(scorefile, scores)
            with self.assertRaisesRegex(ValueError, "mixed-checkpoint"):
                calibrate(context)
            scores[0]["checkpoint_export_sha256"] = "model"
            write(scorefile, scores)
            (root / "source-0.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                calibrate(context)
