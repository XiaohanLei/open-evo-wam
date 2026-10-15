"""Native RoboTwin prefix scoring and online selection, without campaign scheduling."""

import hashlib
from evo_wam.native_idm import native_module
from pathlib import Path
import numpy as np
from evo_wam.io import read, write, sha, identity
from evo_wam.stages import json_rows, complete
from .stage import NativeFrames
from .protocol import Protocol


def window_seed(uid, start, length, interface, namespace):
    if interface == "dense":
        return int(
            hashlib.sha256(f"{namespace}|{uid}|{start}|{length}".encode()).hexdigest()[
                :8
            ],
            16,
        )
    raise ValueError("Unknown native IDM interface")


def score_prefixes(
    uid, ends, video, actions, states, instruction, predict, stats, interface, namespace
):
    """Exact endpoint windows, frozen state context and the historical seed formula."""
    if interface != "dense":
        raise ValueError("Unknown native IDM interface")
    size, stride = 64, 1
    if not namespace or ends != sorted(set(ends)):
        raise ValueError("Missing seed namespace or unordered endpoints")
    std = np.maximum(np.asarray(stats["action"]["std"], np.float64), 1e-6)
    if std.shape != (14,) or not np.isfinite(std).all():
        raise ValueError("Invalid normalization")
    cache = {}
    prefixes = {}
    records = []
    for end in ends:
        if type(end) is not int or not 1 <= end <= len(actions) or end % stride:
            raise ValueError("Invalid native endpoint")
        pieces = []
        for start in range(0, end, size):
            stop = min(start + size, end)
            key = (start, stop)
            if key not in cache:
                state = np.asarray(states[start // size], np.float32)
                frames = video[start // stride : stop // stride + 1]
                if (
                    state.shape != (14,)
                    or not np.isfinite(state).all()
                    or len(frames) != (stop - start) // stride + 1
                ):
                    raise ValueError("Missing window context")
                seed = window_seed(uid, start, stop - start, interface, namespace)
                kw = dict(
                    video=frames,
                    initial_state=state,
                    instruction=instruction,
                    frame_indices=np.arange(start, stop + 1, stride),
                    seed=seed,
                )
                kw["fps"] = 15
                pred = np.asarray(predict(**kw))
                candidate = np.asarray(actions[start:stop], np.float64)
                if (
                    pred.shape != (stop - start, 14)
                    or candidate.shape != pred.shape
                    or not np.isfinite(pred).all()
                    or not np.isfinite(candidate).all()
                ):
                    raise ValueError("Invalid prediction or candidate")
                error = ((candidate - pred) / std) ** 2
                if not np.isfinite(error).all():
                    raise ValueError("Nonfinite score")
                cache[key] = error
                records.append(
                    dict(
                        start=start,
                        end=stop,
                        length=stop - start,
                        seed=seed,
                        mse=float(error.mean()),
                        state_source="frozen_generation_boundary",
                    )
                )
            pieces.append(cache[key])
        errors = np.concatenate(pieces)
        prefixes[str(end)] = dict(
            mse=float(errors.mean()),
            arm_mse=float(errors[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].mean()),
            gripper_mse=float(errors[:, [6, 13]].mean()),
            steps=end,
            windows=len(pieces),
            scored_actions=end,
            unscored_tail_actions=0,
        )
    return dict(prefixes=prefixes, windows=records)


def load_predictor(spec):
    import torch
    from hydra.utils import instantiate

    cfg = native_module("configuration", spec["source_root"])
    cfg.TOKENIZER = spec["tokenizer"]
    configure_environment = native_module("train", spec["source_root"]).configure_environment
    strict_load = native_module("checkpoints", spec["source_root"]).strict_load
    WindowPredictor = native_module("inference", spec["source_root"]).WindowPredictor

    configure_environment()
    torch.cuda.set_device(0)
    checkpoint = Path(spec["idm_checkpoint"])
    config = read(checkpoint / "model_config.json")
    config["config"]["action_head_cfg"]["config"]["skip_component_loading"] = True
    model = instantiate(config).to(device="cuda:0", dtype=torch.bfloat16)
    strict_load(model, checkpoint)
    model.eval()
    stats = read(checkpoint / "statistics.json")
    return WindowPredictor(model, stats), stats


def score(context):
    spec = context["spec"]
    out = Path(context["output"])
    predict, stats = load_predictor(spec)
    data_module = native_module("data", spec["source_root"])
    canvas_views, render_views = data_module.canvas_views, data_module.render_views

    manifest = (
        spec.get("generated_manifest")
        or read(Path(context["previous_stage"]) / "result.json")["manifest"]
    )
    protocol = Protocol(spec["visual_protocol"])
    records = []
    dependencies = [manifest]
    model_identity = identity(spec["idm_checkpoint"])
    for row in json_rows(manifest):
        reader = NativeFrames(row, spec)
        uid = row["id"]
        ends = []
        horizon = min(reader.length, int(row.get("horizon_gate_steps", reader.length)))
        for t in range(spec["scan_start_seconds"][reader.task] * 15, horizon + 1, 15):
            if not protocol.contract.prefix_gate(reader.actions, reader.state, t)[
                "pass"
            ]:
                break
            ends.append(t)
        top = max(ends, default=0)
        states = []
        case = reader.case
        paths = list(reader.paths)
        if spec["interface"] == "dense":
            generation = case / "generation_manifest.json"
            g = read(generation)
            paths.append(generation)
            if (
                g["recurrence_space"] != "latent"
                or g["decoded_rgb_used_for_recurrence"] is not False
            ):
                raise ValueError("Invalid recurrence")
            for k, ch in enumerate(g["chunks"]):
                state = np.asarray(ch["state"], np.float32)
                expected = reader.state if k == 0 else reader.actions[k * 64 - 1]
                if ch["boundary"] != k * 64 or not np.array_equal(state, expected):
                    raise ValueError("Generation boundary state mismatch")
                states.append(state)
            future = reader.frames[:top]
            if future.shape[1:] == (352, 640, 3):
                future = np.pad(future, ((0, 0), (0, 8), (0, 0), (0, 0)), mode="edge")
            video = render_views(
                canvas_views(np.concatenate([reader.initial[None], future]), {}),
                False,
                0,
            )
        else:
            raise ValueError("Unknown native IDM interface")
        namespace = row.get("idm_seed_namespace", spec["seed_namespace"])
        result = score_prefixes(
            uid,
            ends,
            video,
            reader.actions,
            states,
            reader.instruction,
            predict,
            stats,
            spec["interface"],
            namespace,
        )
        dependencies.extend(paths)
        records.append(
            dict(
                result,
                id=uid,
                seed_namespace=namespace,
                checkpoint=spec["idm_checkpoint"],
                checkpoint_identity=model_identity,
                source_manifest_sha256=sha(manifest),
                source_identities={str(p): identity(p) for p in paths},
                scoring_version="native_exact_prefix_v1",
            )
        )
    target = out / "prefix-scores.json"
    write(target, dict(episodes=records))
    complete(
        context,
        [target],
        scores=str(target),
        dependency_paths=list(map(str, dependencies)),
    )


def verify(context):
    from .stage import run

    spec = context["spec"]
    out = Path(context["output"])
    vlm_only = spec.get("selection_mode") == "vlm_only"
    if vlm_only:
        if spec.get("idm_checkpoint") or spec.get("threshold") is not None:
            raise ValueError("VLM-only must not configure an IDM threshold or checkpoint")
        scoring = dict(artifacts=[], dependency_paths=[])
        run(context)
    else:
        score(context)
        scoring = read(out / "result.json")
        run(dict(context, spec=dict(spec, prefix_scores=scoring["scores"])))
    visual = read(out / "result.json")
    rows = json_rows(visual["decisions"])
    manifest = (
        spec.get("generated_manifest")
        or read(Path(context["previous_stage"]) / "result.json")["manifest"]
    )
    sources = {r["id"]: r for r in json_rows(manifest)}
    for row in rows:
        source = sources[row["id"]]
        row["source"] = source
        row["selection_mode"] = "vlm_only" if vlm_only else "idm_vlm"
        row["case"] = source["case"]
        row["round"] = context["round"]
        row["target_source"] = "imagined_rgb_and_generated_actions"
        row["actions_sha256"] = sha(Path(source["case"]) / "actions_qpos.npy")
        receipt = "generation_manifest.json"
        row["generation_sha256"] = sha(Path(source["case"]) / receipt)
    target = out / "decisions.json"
    write(target, dict(episodes=rows))
    complete(
        context,
        visual["artifacts"] + scoring["artifacts"] + [target],
        decisions=str(target),
        dependency_paths=sorted(
            set(visual["dependency_paths"] + scoring["dependency_paths"])
        ),
        mode="native_vlm_only" if vlm_only else "native_idm_then_chronological_vlm",
    )
