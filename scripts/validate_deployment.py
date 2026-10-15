"""Verify the deployed commit and run asset-free CPU contract checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "DEPLOYMENT.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise RuntimeError("Unsafe deployment path: " + name)
        data = path.read_bytes()
        if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise RuntimeError("Deployment hash mismatch: " + name)
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    doctor = subprocess.run([sys.executable, "-m", "evo_wam", "doctor", "--config",
                             str(root / "configs/cosmos3-robotwin-improve.yaml")],
                            check=True, capture_output=True, text=True, cwd=root)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "commit": manifest["commit"], "platform": platform.platform(),
              "python": platform.python_version(), "verified_files": len(manifest["files"]),
              "cpu_tests": {"run": result.testsRun, "passed": result.wasSuccessful()},
              "doctor": json.loads(doctor.stdout), "gpu_weight_load_verified": False,
              "real_inference_verified": False, "full_reproduction_verified": False}
    private = root / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    output = private / "deployment-validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.chmod(0o600)
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
