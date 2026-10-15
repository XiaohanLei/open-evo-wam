"""Standalone native RoboTwin IDM training with explicit data and completion proofs."""

import math
import os
import subprocess
import sys
from pathlib import Path
from evo_wam.io import read, write, sha, identity
from evo_wam.stages import json_rows, complete
from evo_wam.native_idm import package_name
from .idm_control import validate_portable_report
from .idm_bindings import configure, asset_path, data_rows, model_assets


def prepare(context):
    spec = context["spec"]
    report = Path(spec["report"])
    out = Path(context["output"])
    training = out / "training"
    if spec["interface"] != "dense":
        raise ValueError("This training adapter supports the dense interface only")
    if any(
        type(spec[k]) is not int or spec[k] < 1
        for k in ("steps", "world_size", "batch", "accumulation")
    ):
        raise ValueError("Positive integer training dimensions required")
    if spec["world_size"] * spec["batch"] * spec["accumulation"] != 256:
        raise ValueError("Native formal global batch must remain 256")
    binding_sha = configure(spec.get("bindings_file"))
    validate_portable_report(report, training, spec.get("historical_gate_sha256"))
    data = report / "data"
    fixture = read(data / "select_fixtures.json")
    if fixture["partition"] != "select":
        raise ValueError("Selection cannot use final or calibration fixtures")
    selected = Path(asset_path(fixture["manifest"]))
    if sha(selected) != fixture["manifest_sha256"]:
        raise ValueError("Selection manifest changed")
    dependencies = {
        str(report / "model_config.json"),
        str(data / "GPU_READY.json"),
        str(data / "statistics.json"),
        str(data / "train.jsonl"),
        str(data / "select_fixtures.json"),
        str(selected),
    }
    if spec.get("bindings_file"):
        dependencies.add(spec["bindings_file"])
    counts = {}
    for path in (data / "train.jsonl", selected):
        rows = data_rows(json_rows(path))
        counts[path.name] = len(rows)
        for row in rows:
            dependencies.add(row["cache_path"])
            dependencies.add(
                row["video_path"] if row["source"] == "base" else row["source_path"]
            )
    model = model_assets(read(report / "model_config.json"))

    def assets(value):
        if isinstance(value, dict):
            for v in value.values():
                yield from assets(v)
        elif isinstance(value, list):
            for v in value:
                yield from assets(v)
        elif isinstance(value, str) and value.startswith("/"):
            yield value

    dependencies.update(assets(model))
    missing = sorted(p for p in dependencies if not Path(p).exists())
    if missing:
        raise FileNotFoundError("Missing IDM assets: " + ", ".join(missing[:20]))
    out.mkdir(parents=True, exist_ok=True)
    evidence = dict(
        interface=spec["interface"],
        steps=spec["steps"],
        world_size=spec["world_size"],
        global_batch=256,
        path_bindings_sha256=binding_sha,
        manifest_rows=counts,
        dependencies=sorted(dependencies),
        historical_gate=read(data / "GPU_READY.json"),
        model_config_sha256=sha(report / "model_config.json"),
    )
    write(out / "TRAINING_INPUTS.json", evidence)
    return training, sorted(dependencies)


def checked_checkpoint(path, root, world):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint is outside this training output")
    marker = read(path / "COMPLETE.json")
    if marker["world_size"] != world:
        raise ValueError("Checkpoint world size changed")
    for name, record in marker["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid checkpoint member")
        member = path / name
        if member.stat().st_size != record["bytes"] or sha(member) != record["sha256"]:
            raise ValueError("Checkpoint file changed: " + name)
    return path, marker


def finish(context, training):
    spec = context["spec"]
    exit_record = read(training / "TRAINING_EXIT.json")
    if exit_record["phase"] != "train" or exit_record["step"] != spec["steps"]:
        raise ValueError("IDM training stopped before all requested updates completed")
    best = read(training / "BEST.json")
    if best != exit_record["best"] or not math.isfinite(best["metric"]):
        raise ValueError("Selection and training exit disagree")
    final, final_marker = checked_checkpoint(
        exit_record["final"], training, spec["world_size"]
    )
    selected, marker = checked_checkpoint(best["path"], training, spec["world_size"])
    if final_marker["step"] != spec["steps"] or marker["step"] != best["step"]:
        raise ValueError("Checkpoint steps disagree with training receipts")
    export = selected / "export"
    proof = read(export / "EXPORT_VERIFIED.json")
    if proof["tensors"] != 1670 or not proof["files"]:
        raise ValueError("Unexpected native export tensor count")
    for name, digest in proof["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid export member")
        if sha(export / name) != digest:
            raise ValueError("Selected export changed")
    return export


def train(context):
    spec = context["spec"]
    training, deps = prepare(context)
    before = {p: identity(p) for p in deps}
    argv = [
        *spec["launcher"],
        "-m",
        package_name() + ".train",
        "--portable-run",
        "--report",
        spec["report"],
        "--output",
        str(training),
        "--phase",
        "train",
        "--steps",
        str(spec["steps"]),
        "--batch",
        str(spec["batch"]),
        "--accumulation",
        str(spec["accumulation"]),
        "--workers",
        str(spec.get("workers", 4)),
    ]
    for field, flag in (
        ("bindings_file", "--path-bindings"),
        ("historical_gate_sha256", "--historical-gate-sha256"),
    ):
        if spec.get(field):
            argv.extend([flag, spec[field]])
    exit_path = training / "TRAINING_EXIT.json"
    already_finished = (
        exit_path.exists() and read(exit_path).get("step") == spec["steps"]
    )
    if not already_finished:
        latest = training / "LATEST.json"
        if latest.exists():
            checkpoint, _ = checked_checkpoint(
                read(latest)["path"], training, spec["world_size"]
            )
            argv.extend(["--resume", str(checkpoint)])
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                spec["source_root"],
                str(Path(__file__).resolve().parents[2]),
                env.get("PYTHONPATH", ""),
            ]
        )
        subprocess.run(argv, cwd=spec["source_root"], env=env, check=True)
    export = finish(context, training)
    if any(identity(p) != expected for p, expected in before.items()):
        raise ValueError("IDM input changed during training")
    complete(
        context,
        [training, Path(context["output"]) / "TRAINING_INPUTS.json"],
        checkpoint=str(export),
        dependency_paths=deps,
        strict_reload_verified=False,
        export_file_hashes_verified=True,
    )
