import unittest
from evo_wam.adapters.cosmos_training import relocate, override, load_callable


class TrainingConfigTests(unittest.TestCase):
    def test_longest_directory_binding(self):
        x = {
            "path": "/old/data/a",
            "other": "/old/database",
            "loss": 10,
            "layers": [1, 2],
        }
        y = relocate(x, {"/old": "/new", "/old/data": "/dataset"})
        self.assertEqual(y["path"], "/dataset/a")
        self.assertEqual(y["other"], "/new/database")
        self.assertEqual(y["loss"], 10)
        self.assertEqual(x["path"], "/old/data/a")

    def test_missing_override_rejected(self):
        with self.assertRaises(KeyError):
            override({"trainer": {"steps": 4}}, {"trainer.step": 8})

    def test_nested_callable_resolution(self):
        from pathlib import Path

        self.assertEqual(load_callable("pathlib.Path.cwd")(), Path.cwd())
