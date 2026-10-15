"""Local execution controls for the two native RoboTwin IDM trainers."""

from pathlib import Path
from evo_wam.io import read, sha


def validate_portable_report(report, output, historical_gate_sha256=None):
    report, output = Path(report).resolve(), Path(output).resolve()
    if output == report or output.is_relative_to(report):
        raise ValueError("Portable output must be separate from the frozen report")
    data = report / "data"
    gate = read(data / "GPU_READY.json")
    if gate.get("passed") is not True and not (
        gate.get("user_authorized_direct_launch") is True
        and bool(gate.get("authorization"))
        and historical_gate_sha256 == sha(data / "GPU_READY.json")
    ):
        raise ValueError(
            "Portable training requires a passed gate or an explicitly hash-bound historical authorization"
        )
    required = {"train.jsonl", "statistics.json", "select_fixtures.json"}
    if not required.issubset(gate["files"]):
        raise ValueError(
            "Data gate must bind training, statistics and selection fixtures"
        )
    for name, digest in gate["files"].items():
        path = data / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Data gate contains an invalid relative path")
        if sha(path) != digest:
            raise ValueError("Frozen data changed: " + name)
    if not (report / "model_config.json").is_file():
        raise FileNotFoundError(report / "model_config.json")
    if (output / "STOP").exists():
        raise ValueError(
            "STOP already exists; inspect the previous run before continuing"
        )


def stop_requested(output):
    return (Path(output) / "STOP").exists()
