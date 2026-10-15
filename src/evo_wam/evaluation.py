"""Frozen-registry accounting with explicit task and infrastructure outcomes."""

from pathlib import Path
from .io import write
from .stages import json_rows, complete

IDENTITY_FIELDS = (
    "registry_id",
    "episode_key",
    "episode_spec_sha256",
    "fixed_seed",
    "instruction",
    "exact_instruction",
    "task_name",
    "task_config",
    "inference_noise_seed",
)
INFRASTRUCTURE_STATUSES = {
    "infrastructure_error",
    "init_fail",
    "expert_check_fail",
    "simulator_crash",
    "timeout",
    "physics_invalid",
}


def summarize(registry, results):
    expected = {r["id"]: r for r in registry}
    if len(expected) != len(registry):
        raise ValueError("Duplicate registry episode")
    got = {}
    for row in results:
        uid = row["id"]
        if uid not in expected:
            raise ValueError("Unexpected episode " + uid)
        if uid in got:
            raise ValueError("Duplicate episode result " + uid)
        if (
            row["status"]
            not in {"success", "failure", "model_task_failure"}
            | INFRASTRUCTURE_STATUSES
        ):
            raise ValueError("Unknown episode status")
        for key in IDENTITY_FIELDS:
            if key in expected[uid] and row.get(key) != expected[uid][key]:
                raise ValueError("Frozen registry mismatch: " + uid + " " + key)
        got[uid] = row
    missing = sorted(set(expected) - set(got))
    if missing:
        raise ValueError("Missing episodes: " + ",".join(missing))
    success = sum(r["status"] == "success" for r in got.values())
    failure = sum(
        r["status"] in ("failure", "model_task_failure") for r in got.values()
    )
    infrastructure = len(got) - success - failure
    detail = {
        s: sum(r["status"] == s for r in got.values())
        for s in sorted({r["status"] for r in got.values()})
    }
    groups = {}
    for config in sorted(
        {r.get("task_config") for r in registry if r.get("task_config")}
    ):
        subset = [r for r in registry if r.get("task_config") == config]
        ids = {r["id"] for r in subset}
        s = sum(got[i]["status"] == "success" for i in ids)
        valid = sum(
            got[i]["status"] in ("success", "failure", "model_task_failure")
            for i in ids
        )
        groups[config] = dict(
            planned=len(ids),
            success=s,
            valid=valid,
            conservative_success=s / len(ids),
            valid_success=s / valid if valid else None,
        )
    return dict(
        total=len(registry),
        success=success,
        failure=failure,
        infrastructure_error=infrastructure,
        success_rate=success / len(registry) if registry else None,
        conservative_success=success / len(registry) if registry else None,
        valid_success=success / (success + failure) if success + failure else None,
        outcome_counts=detail,
        by_task_config=groups,
        complete_without_infrastructure_errors=infrastructure == 0,
    )


def replay(context):
    spec = context["spec"]
    registry = json_rows(spec["registry"])
    results = json_rows(spec["episode_results"])
    if spec.get("checkpoint_identity"):
        for row in results:
            if row.get("checkpoint_identity") != spec["checkpoint_identity"]:
                raise ValueError("Evaluation checkpoint identity mismatch")
    result = summarize(registry, results)
    out = Path(context["output"]) / "evaluation.json"
    write(out, dict(result, mode="frozen_episode_result_replay"))
    complete(context, [out], summary=result)
