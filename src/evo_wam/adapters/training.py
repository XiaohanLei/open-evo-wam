"""Standard launcher bridge for native training and export, without resource scheduling."""

import os, subprocess
from pathlib import Path
from evo_wam.io import write, identity
from evo_wam.stages import complete


def run(context):
    spec = context["spec"]
    root = Path(context["output"])
    dependencies = sorted(set(spec.get("dependency_paths", [])))
    dependency_identities = {p: identity(p) for p in dependencies}
    env = dict(os.environ, **{k: str(v) for k, v in spec.get("native_env", {}).items()})
    env["PYTHONPATH"] = os.pathsep.join(
        [
            spec["source_root"],
            *spec.get("extra_python_paths", []),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["EVO_WAM_PARENT_CHECKPOINT"] = context["parent_checkpoint"]
    env["EVO_WAM_STAGE_OUTPUT"] = str(root)
    if context.get("previous_stage") and spec.get("requires_previous", True):
        from evo_wam.io import read

        previous = read(Path(context["previous_stage"]) / "result.json")
        if previous.get("manifest"):
            env["EVO_WAM_TRAINING_MANIFEST"] = previous["manifest"]
    checkpoint = Path(spec["checkpoint_output"])
    before = identity(checkpoint) if checkpoint.exists() else None
    for name in ("train_argv", "export_argv"):
        argv = spec.get(name)
        if argv is None and name == "export_argv":
            continue
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(x, str) for x in argv)
        ):
            raise ValueError(name + " must be an argv list")
        subprocess.run(argv, cwd=spec["source_root"], env=env, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(
            "Native training/export did not produce " + str(checkpoint)
        )
    after = identity(checkpoint)
    for dependency, expected in dependency_identities.items():
        if identity(dependency) != expected:
            raise ValueError("Training input changed during execution: " + dependency)
    if before == after:
        raise ValueError(
            "Training returned without producing a changed checkpoint; inspect native receipts"
        )
    evidence = dict(
        training_returncode=0,
        export_requested="export_argv" in spec,
        checkpoint=str(checkpoint),
        checkpoint_identity=after,
        strict_reload_verified=False,
    )
    path = root / "native-training.json"
    write(path, evidence)
    complete(
        context,
        [path, checkpoint, *spec.get("recipe_artifacts", [])],
        checkpoint=str(checkpoint),
        dependency_paths=dependencies,
    )
