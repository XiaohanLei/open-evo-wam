import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evo_wam.native_idm import native_module, package_name, remap_environment


class NativeIDMTests(unittest.TestCase):
    def test_package_binding_and_source_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("first", "second"):
                package = root / name / "sample_idm"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("")
                (package / "data.py").write_text("MARKER = '" + name + "'\n")
            try:
                with patch.dict(os.environ, IDM_PACKAGE="sample_idm"):
                    self.assertEqual(native_module("data", root / "first").MARKER, "first")
                    with self.assertRaises(RuntimeError):
                        native_module("data", root / "second")
            finally:
                for key in list(sys.modules):
                    if key == "sample_idm" or key.startswith("sample_idm."):
                        del sys.modules[key]
                sys.path[:] = [p for p in sys.path if p != str(root / "first")]

    def test_invalid_or_missing_package_rejected(self):
        with patch.dict(os.environ, IDM_PACKAGE="../invalid"):
            with self.assertRaises(ValueError):
                package_name()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, IDM_PACKAGE="missing_idm"):
                with self.assertRaises(FileNotFoundError):
                    native_module("data", directory)

    def test_environment_mapping_preserves_values(self):
        env = {"DATA_ROOT": "/data", "SLOTS": "32"}
        self.assertEqual(remap_environment(env, {"DATA_ROOT": "NATIVE_DATA"}),
                         {"NATIVE_DATA": "/data", "SLOTS": "32"})
        with self.assertRaises(ValueError):
            remap_environment(env, {"DATA_ROOT": "SLOTS"})
        with self.assertRaises(ValueError):
            remap_environment(env, {"UNKNOWN": "NATIVE_DATA"})
