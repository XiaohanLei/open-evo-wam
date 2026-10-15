"""Portable stage contracts. Native adapters must produce explicit artifacts."""

import importlib, math
from pathlib import Path
from .io import read, write, sha
from .full_prefix import summary


def json_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            __import__("json").loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    obj = read(path)
    return obj if isinstance(obj, list) else obj["episodes"]


def complete(context, artifacts, **fields):
    write(
        Path(context["output"]) / "result.json",
        dict(
            status="complete",
            stage=context["stage"],
            round=context["round"],
            artifacts=[str(Path(p).resolve()) for p in artifacts],
            **fields,
        ),
    )


def verify(context):
    """Replay frozen endpoint votes and exact scores. No VLM response is synthesized."""
    spec = context["spec"]
    rows = json_rows(spec["decisions"])
    out = Path(context["output"]) / "decisions.json"
    threshold = float(spec["threshold"])
    minimum = int(spec.get("minimum_votes", 1))
    result = []
    if not math.isfinite(threshold) or threshold < 0 or minimum < 1:
        raise ValueError("Invalid threshold/vote count")
    seen = set()
    for row in rows:
        uid = row["id"]
        if uid in seen:
            raise ValueError("Duplicate candidate ID")
        seen.add(uid)
        endpoint = row["visual_endpoint"]
        groups = row.get(
            "vote_groups",
            [
                dict(
                    goal="visual_success", endpoint=endpoint, votes=row.get("votes", [])
                )
            ],
        )
        required = spec.get("required_goals", ["visual_success"])
        if len({g["goal"] for g in groups}) != len(groups) or {
            g["goal"] for g in groups
        } != set(required):
            raise ValueError("Missing or duplicate subgoal votes")
        counts = {}
        for group in groups:
            votes = group["votes"]
            point = group["endpoint"]
            if type(point) is not int or not 1 <= point <= endpoint:
                raise ValueError("Invalid subgoal endpoint")
            if len({v["id"] for v in votes}) != len(votes):
                raise ValueError("Duplicate vote identity")
            if any(
                type(v["accepted"]) is not bool or v["endpoint"] != point for v in votes
            ):
                raise ValueError("Votes must refer to the selected endpoint")
            counts[str(group["goal"])] = sum(v["accepted"] for v in votes)
        score = summary(row["windows"], endpoint)
        decision = dict(
            row,
            **score,
            accepted=all(v >= minimum for v in counts.values())
            and score["normalized_mse"] <= threshold,
            votes_for=counts,
            threshold=threshold,
        )
        if (
            "expected_accepted" in row
            and decision["accepted"] != row["expected_accepted"]
        ):
            raise ValueError("Historical decision replay mismatch: " + uid)
        result.append(decision)
    write(out, dict(schema="evo-wam.decisions.v1", episodes=result))
    complete(
        context,
        [out],
        decisions=str(out),
        accepted=sum(r["accepted"] for r in result),
        mode="frozen_response_replay",
    )


def export_prefix(context):
    """Boundary-aligned NPZ interface. Each adapter retains its native schema."""
    import numpy as np

    spec = context["spec"]
    decision_path = spec.get("decisions")
    if not decision_path:
        decision_path = read(Path(context["previous_stage"]) / "result.json")[
            "decisions"
        ]
    decisions = json_rows(decision_path)
    episodes = []
    artifacts = []
    horizon = int(spec["chunk_actions"])
    dimension = int(spec["action_dim"])
    out = Path(context["output"])
    if horizon < 1 or dimension < 1:
        raise ValueError("Invalid native interface")
    for row in decisions:
        if not row["accepted"]:
            continue
        with np.load(row["trajectory"], allow_pickle=False) as src:
            t = row["accepted_step"]
            actions = src["actions"]
            frames = src["frames"]
            states = src["states"]
            if (
                actions.ndim != 2
                or actions.shape[1] != dimension
                or len(actions) < t
                or len(frames) < t + 1
            ):
                raise ValueError("Missing actions or boundary-aligned RGB")
            if len(states) < (t - 1) // horizon + 1:
                raise ValueError("Missing window starting states")
            chunks = []
            for k, lo in enumerate(range(0, t, horizon)):
                hi = min(lo + horizon, t)
                length = hi - lo
                a = actions[lo:hi]
                state = states[k]
                if not np.isfinite(a).all() or not np.isfinite(state).all():
                    raise ValueError("Nonfinite data")
                padded = np.zeros((horizon, dimension), dtype=a.dtype)
                padded[:length] = a
                mask = np.arange(horizon) < length
                # A short prefix has no fabricated terminal-state target.
                terminal_valid = length == horizon and k + 1 < len(states)
                terminal = states[k + 1] if terminal_valid else np.zeros_like(state)
                path = out / f"episode_{len(episodes):06d}_chunk_{k:05d}.npz"
                np.savez_compressed(
                    path,
                    actions=padded,
                    action_mask=mask,
                    frames=frames[lo : hi + 1],
                    initial_state=state,
                    terminal_state=terminal,
                    terminal_state_mask=np.asarray(terminal_valid),
                    start=lo,
                    end=hi,
                )
                chunks.append(
                    dict(
                        path=str(path),
                        sha256=sha(path),
                        boundary=lo,
                        valid_actions=length,
                    )
                )
                artifacts.append(path)
            episodes.append(
                dict(
                    id=row["id"],
                    instruction=row["instruction"],
                    accepted_step=t,
                    chunks=chunks,
                    interface=spec["interface"],
                )
            )
    if spec.get("previous_pool"):
        prior = json_rows(spec["previous_pool"])
        ids = {r["id"] for r in prior}
        if ids.intersection(r["id"] for r in episodes):
            raise ValueError("Duplicate episode in cumulative pool")
        episodes = prior + episodes
    manifest = out / "manifest.json"
    write(
        manifest,
        dict(
            schema="evo-wam.boundary_npz.v1",
            interface=spec["interface"],
            episodes=episodes,
        ),
    )
    complete(context, artifacts + [manifest], manifest=str(manifest))


def prepare_data(context):
    """Validate a preselected split and preserve its exact file identities."""
    spec = context["spec"]
    episodes = json_rows(spec["manifest"])
    seen = set()
    for row in episodes:
        if row["id"] in seen:
            raise ValueError("Duplicate episode in selected split")
        seen.add(row["id"])
        for name, path in row["files"].items():
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(path)
            expected = row.get("sha256", {}).get(name)
            if expected and sha(p) != expected:
                raise ValueError("Source checksum mismatch: " + path)
    out = Path(context["output"]) / "manifest.json"
    write(
        out,
        dict(
            schema="evo-wam.selected_data.v1",
            episodes=episodes,
            interface=spec["interface"],
            split=spec["split"],
        ),
    )
    complete(
        context,
        [out],
        manifest=str(out),
        dependency_paths=sorted(
            {str(Path(p).resolve()) for row in episodes for p in row["files"].values()}
        ),
    )


def calibrate(context):
    """Explicit quantile definition; preserves calibration IDs and scores."""
    import numpy as np

    rows = json_rows(context["spec"]["scores"])
    values = np.asarray([r["normalized_mse"] for r in rows], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid calibration scores")
    q = float(context["spec"]["quantile"])
    method = context["spec"].get("quantile_method", "linear")
    if not 0 <= q <= 1:
        raise ValueError("Invalid quantile")
    threshold = float(np.quantile(values, q, method=method))
    out = Path(context["output"]) / "calibration.json"
    write(out, dict(threshold=threshold, quantile=q, method=method, samples=rows))
    complete(context, [out], threshold=threshold)


def execute(context):
    adapter = context["spec"].get("adapter")
    if adapter:
        module, name = adapter.split(":", 1)
        getattr(importlib.import_module(module), name)(context)
        return
    functions = {
        "prepare-data": prepare_data,
        "verify": verify,
        "export-prefix": export_prefix,
        "idm-calibrate": calibrate,
    }
    if context["stage"] not in functions:
        raise ValueError(
            "A native adapter or explicit command executor is required for "
            + context["stage"]
        )
    functions[context["stage"]](context)
