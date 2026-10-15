"""Dense quantile calibration from a complete, frozen held-out population."""

import math
from pathlib import Path
import numpy as np
from evo_wam.io import sha, write
from evo_wam.stages import json_rows, complete
from evo_wam.path_bindings import PathBindings


def calibrate(context):
    spec = context["spec"]
    if spec["interface"] != "dense":
        raise ValueError("This is the dense success-quantile protocol")
    population = json_rows(spec["manifest"])
    scores = json_rows(spec["scores"])
    by_id = {row["uid"]: row for row in population}
    if (
        not population
        or len(by_id) != len(population)
        or any(row["partition"] != "calibration" for row in population)
    ):
        raise ValueError(
            "Calibration requires unique held-out calibration trajectories"
        )
    if len(scores) != len(population) or {row["uid"] for row in scores} != set(by_id):
        raise ValueError("Calibration scores have missing or duplicate trajectories")
    bindings = PathBindings(spec.get("path_bindings", []))
    dependencies = [spec["manifest"], spec["scores"]]
    success = []
    for row in scores:
        source = by_id[row["uid"]]
        if any(
            row[key] != source[key] for key in ("task", "config", "outcome", "length")
        ):
            raise ValueError("Score and calibration manifest disagree")
        if (
            row["status"] != "complete"
            or row["checkpoint_export_sha256"] != spec["checkpoint_export_sha256"]
        ):
            raise ValueError("Incomplete or mixed-checkpoint calibration scores")
        if (
            source["outcome"] not in ("success", "fail")
            or not math.isfinite(row["mse"])
            or row["mse"] < 0
        ):
            raise ValueError("Invalid calibration outcome or MSE")
        score_source = bindings.path(row["source"])
        if sha(score_source) != row["source_sha256"]:
            raise ValueError("Per-trajectory score evidence changed")
        dependencies.append(score_source)
        if source["outcome"] == "success":
            success.append(row["mse"])
    if not success:
        raise ValueError("No real-success calibration trajectories")
    schedule = [
        dict(
            round=r,
            percentile=90 - 2.5 * (r - 1),
            threshold=float(
                np.quantile(success, (90 - 2.5 * (r - 1)) / 100, method="linear")
            ),
        )
        for r in range(1, 38)
    ]
    target = Path(context["output"]) / "THRESHOLDS.json"
    write(
        target,
        dict(
            protocol="dense_success_quantile_v1",
            quantile_method='numpy.quantile(method="linear")',
            population="one equal weight per real-success trajectory",
            total=len(scores),
            successes=len(success),
            failures=len(scores) - len(success),
            tasks=len({r["task"] for r in population}),
            schedule=schedule,
            snapshot_sha256=sha(spec["scores"]),
            calibration_manifest_sha256=sha(spec["manifest"]),
            checkpoint_export_sha256=spec["checkpoint_export_sha256"],
            validation_scope="Frozen complete trajectory scores and source hashes; no new model inference",
        ),
    )
    complete(
        context,
        [target],
        threshold_schedule=str(target),
        dependency_paths=sorted(set(dependencies)),
    )
