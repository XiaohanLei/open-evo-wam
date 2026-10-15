"""Run native VLM description/judgment and fixed-endpoint vote3 against an existing API."""

import hashlib, io
from pathlib import Path
import numpy as np
from evo_wam.io import read, write, sha, identity
from evo_wam.stages import json_rows, complete
from .client import Client
from .protocol import Protocol
from .scan import scan


def png(x):
    from PIL import Image

    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB")
    stream = io.BytesIO()
    Image.fromarray(x).save(stream, format="PNG")
    return stream.getvalue()


def run(context):
    spec = context["spec"]
    vlm_only = spec.get("selection_mode") == "vlm_only"
    if vlm_only and (spec.get("prefix_scores") or spec.get("visual_protocol") != "robotwin_dense"):
        raise ValueError("VLM-only requires dense RoboTwin without IDM scores")
    out = Path(context["output"])
    protocol = Protocol(spec["visual_protocol"])
    manifest = (
        spec.get("generated_manifest")
        or read(Path(context["previous_stage"]) / "result.json")["manifest"]
    )
    rows = json_rows(manifest)
    decisions = []
    dependencies = [manifest]
    if spec.get("format_instructions"):
        dependencies.append(spec["format_instructions"])
    prefix_scores = {}
    if spec.get("prefix_scores"):
        prefix_scores = {r["id"]: r for r in json_rows(spec["prefix_scores"])}
        dependencies.append(spec["prefix_scores"])
        if set(prefix_scores) != {r["id"] for r in rows}:
            raise ValueError(
                "Prefix scores must cover exactly the generated candidates"
            )
    client = Client(
        out / "vlm-requests",
        spec["endpoint"],
        spec["model_revision"],
        protocol.parse,
        max_attempts=spec.get("max_attempts", 6),
        timeout=spec.get("timeout", 240),
        api_key_env=spec.get("api_key_env"),
    )
    ids = set()
    for row in rows:
        uid = row["id"]
        if uid in ids:
            raise ValueError("Duplicate candidate ID")
        ids.add(uid)
        # Raw IDs never become directory traversal components.
        key = hashlib.sha256(uid.encode()).hexdigest()
        reader = NativeFrames(row, spec)
        dependencies.extend(reader.paths)
        task = reader.task
        instruction = reader.instruction
        if prefix_scores:
            record = prefix_scores[uid]
            if record.get("source_manifest_sha256") != sha(manifest):
                raise ValueError("Score manifest identity differs")
            if record.get("checkpoint_identity") != identity(spec["idm_checkpoint"]):
                raise ValueError("Score checkpoint changed")
            if not set(map(str, reader.paths)).issubset(
                record.get("source_identities", {})
            ):
                raise ValueError("Score lacks generation dependencies")
            for path, expected in record["source_identities"].items():
                if identity(path) != expected:
                    raise ValueError("Score input changed: " + path)
        if protocol.name == "droid":
            windows = read(protocol.root / "windows.json")[task]
            goals = [
                list(
                    range(
                        lo * 15,
                        min(reader.length, hi * 15 if hi is not None else reader.length)
                        + 1,
                        15,
                    )
                )
                for lo, hi in windows
            ]
        else:
            start = spec["scan_start_seconds"][task] * 15
            horizon = reader.length if vlm_only else min(
                reader.length, int(row.get("horizon_gate_steps", reader.length))
            )
            goals = [list(range(start, horizon + 1, 15))]

        def prepare(goal, step):
            first, current, state, actions = reader.at(step)
            if protocol.name != "droid" and not vlm_only:
                gate = protocol.contract.prefix_gate(actions, state, step)
                if not gate["pass"]:
                    return dict(action_filtered=True, gate=gate)
            if prefix_scores:
                record = prefix_scores[uid]
                if record.get("checkpoint") != spec["idm_checkpoint"]:
                    raise ValueError("IDM model identity differs")
                score = record["prefixes"][str(step)]["mse"]
                if not np.isfinite(score):
                    raise ValueError("Nonfinite prefix score")
                if score > float(spec["threshold"]):
                    return dict(gate_skipped=True)
            if protocol.name == "droid":
                if reader.backbone == "dreamzero-droid":
                    crop = lambda x: [x[:176], x[176:, :320], x[176:, 320:]]
                else:
                    crop = lambda x: [x[:360], x[360:528, :320], x[360:528, 320:]]
                if (
                    not np.isfinite(actions[:step]).all()
                    or not np.isfinite(state).all()
                ):
                    raise ValueError("Nonfinite candidate state/actions")
            else:
                crop = protocol.contract.crops
            raw = [png(x) for x in crop(first) + crop(current)]
            return dict(raw=raw)

        def chain(goal, step, vote, evidence):
            desc = None
            records = {}
            for kind in ("description", "judgment"):
                payload = protocol.payload(
                    task, goal, kind, instruction, evidence["raw"], desc
                )
                payload["model"] = spec.get("model", payload["model"])
                if "max_tokens" in spec:
                    payload["max_tokens"] = spec["max_tokens"]
                if "thinking_token_budget" in spec:
                    payload["thinking_token_budget"] = spec["thinking_token_budget"]
                if spec.get("format_instructions"):
                    guidance = read(spec["format_instructions"])[kind]
                    payload["messages"][-1]["content"].append(dict(type="text", text=guidance))
                record = client.request(
                    f"{key}/g{goal}_t{step}_v{vote}/{kind}",
                    payload,
                    dict(kind=kind, description=desc, task=task, goal=goal),
                )
                records[kind] = record
                if kind == "description":
                    desc = record["prediction"]
            return dict(
                decision=records["judgment"]["prediction"]["decision"], **records
            )

        decision = scan(
            goals,
            prepare,
            chain,
            vote_timing=spec.get(
                "vote_timing",
                (
                    "per_moment"
                    if protocol.name != "droid"
                    else "per_goal"
                    if reader.backbone == "dreamzero-droid"
                    else "after_scan"
                ),
            ),
        )
        if prefix_scores and decision.get("accepted"):
            point = decision["accepted_step"]
            decision["idm"] = dict(
                mse=prefix_scores[uid]["prefixes"][str(point)]["mse"],
                endpoint=point,
                checkpoint=spec["idm_checkpoint"],
                threshold=spec["threshold"],
            )
        decisions.append(
            dict(
                decision,
                id=uid,
                task=task,
                instruction=instruction,
                source_manifest_sha256=sha(manifest),
            )
        )
    target = out / "visual-decisions.json"
    write(target, dict(episodes=decisions))
    complete(
        context,
        [target, out / "vlm-requests"] if (out / "vlm-requests").exists() else [target],
        decisions=str(target),
        dependency_paths=sorted(set(map(str, dependencies))),
        mode="online_native_vlm_visual_only",
    )


class NativeFrames:
    def __init__(self, row, spec):
        self.row = row
        self.backbone = spec["backbone"]
        self.case = Path(row["case"])
        self.paths = []
        job = row.get("job", row)
        scene = job.get("scene", {})
        self.task = row.get("task", row.get("task_name", scene.get("task")))
        self.instruction = row.get("instruction", scene.get("instruction"))
        if not self.task or not self.instruction:
            raise ValueError("Task and instruction are required")
        if self.backbone == "cosmos3-droid":
            start = Path(row["initial_observation"])
            self.paths = [
                start,
                self.case / "actions_model_space.npy",
                self.case / "self_rollout.mp4",
            ]
            with np.load(start, allow_pickle=False) as z:
                self.initial = z["global_image"].copy()
                self.state = z["state_model"].copy()
            self.actions = np.load(self.paths[1], allow_pickle=False)
            self.length = len(self.actions)
            self.frames = None
        elif self.backbone == "dreamzero-robotwin":
            self.paths = [
                self.case / "initial.npz",
                self.case / "actions_qpos.npy",
                self.case / "imagined_frames.npy",
            ]
            with np.load(self.paths[0], allow_pickle=False) as z:
                self.initial = z["canvas"].copy()
                self.state = z["state"].copy()
            self.actions = np.load(self.paths[1], allow_pickle=False)
            self.frames = np.load(self.paths[2], mmap_mode="r", allow_pickle=False)
            self.length = len(self.actions)
        elif self.backbone == "dreamzero-droid":
            self.paths = [self.case / "initial.npz"] + [
                self.case / f"chunk_{k:05d}.npz" for k in range(scene["chunks"])
            ]
            with np.load(self.paths[0], allow_pickle=False) as z:
                self.initial = z["canvas"].copy()
                self.state = z["state"].copy()
            parts = [np.load(p, allow_pickle=False) for p in self.paths[1:]]
            try:
                self.actions = np.concatenate([z["actions"] for z in parts])
                self.frames = np.concatenate([z["rgb"] for z in parts])
            finally:
                for z in parts:
                    z.close()
            self.length = min(len(self.actions), scene["max_actions"])
        elif self.backbone == "cosmos3-robotwin":
            from evo_wam.data.robotwin_canvas import observation_image

            initial = Path(row["initial_observation"])
            self.paths = [
                initial,
                self.case / "actions_qpos.npy",
                self.case / "imagined_frames.npz",
            ]
            with np.load(initial, allow_pickle=False) as z:
                if "q0" in z:
                    self.state = z["q0"].copy()
                    self.initial = observation_image(
                        {
                            "observation": {
                                k[:-4]: {"rgb": z[k]}
                                for k in z.files
                                if k.endswith("_rgb")
                            }
                        }
                    )
                else:
                    self.state = z["state"].copy()
                    self.initial = z["global_image"].copy()
            self.actions = np.load(self.paths[1], allow_pickle=False)
            with np.load(self.paths[2], allow_pickle=False) as z:
                self.frames = z["frames"].copy()
            self.length = min(len(self.actions), row["max_steps"])
        else:
            raise ValueError("Unknown backbone " + self.backbone)

    def at(self, step):
        if not 1 <= step <= self.length:
            raise ValueError("Endpoint outside generated trajectory")
        if self.frames is not None:
            stride = 1 if self.backbone == "cosmos3-robotwin" else 3
            if step % stride:
                raise ValueError("Sparse endpoint is not on the native grid")
            current = self.frames[step // stride - 1]
        else:
            import cv2

            cap = cv2.VideoCapture(str(self.paths[-1]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, step - 1)
            ok, bgr = cap.read()
            cap.release()
            if not ok:
                raise ValueError("Missing generated endpoint frame")
            current = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.initial, current, self.state, self.actions
