"""Lossless export from portable generation/verification receipts into the native dataset."""

import hashlib, subprocess, sys
from pathlib import Path
import numpy as np
from evo_wam.io import read, write, sha
from evo_wam.stages import json_rows
from evo_wam.verification.stage import NativeFrames
from evo_wam.verification.protocol import Protocol
from .robotwin_dense import publish


def training_scene_identity(row, registry=None):
    """Bind training grouping to explicit collection metadata, never infer it from acceptance."""
    fields = ('episode_key','scene_set','repeat_index')
    result = {key:row[key] for key in fields if key in row}
    if registry is not None:
        entries = registry['episodes']
        if registry.get('schema') != 'evo-wam.export-scene-registry.v1' or len({e['id'] for e in entries}) != len(entries):
            raise ValueError('Invalid export scene registry')
        entries = {e['id']:e for e in entries}
        bound = entries[row['id']]
        if (bound['generation_sha256'] != sha(Path(row['case'])/'generation_manifest.json')
            or bound['initial_observation_sha256'] != sha(row['initial_observation'])):
            raise ValueError('Scene registry refers to different generated evidence')
        for key in fields:
            if key in result and result[key] != bound[key]:
                raise ValueError('Scene registry conflicts with generation metadata')
            result[key] = bound[key]
    if result.get('scene_set') != 'A' or type(result.get('repeat_index')) is not int:
        raise ValueError('Explicit scene-set A and repeat index required by the native training view')
    if not (0 <= result['repeat_index'] < 20 or 40 <= result['repeat_index'] < 120):
        raise ValueError('Repeat index outside the native training split')
    if not result.get('episode_key'):
        raise ValueError('Stable scene identity required for balanced native sampling')
    return result


def export(context):
    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = context["spec"]
    vlm_only = spec.get("selection_mode") == "vlm_only"
    oracle = spec.get("selection_mode") == "oracle_vlm"
    no_idm = vlm_only or oracle
    if no_idm and (spec.get("threshold") is not None or spec.get("history_refilter")):
        raise ValueError("VLM-only export must not apply IDM threshold filtering")
    sys.path.insert(0, spec["source_root"])
    out = Path(context["output"])
    dataset = out / "dataset"
    source = (
        spec.get("decisions")
        or read(Path(context["previous_stage"]) / "result.json")["decisions"]
    )
    rows = json_rows(source)
    index = []
    artifacts = []
    dependencies = [source]
    registry = read(spec['scene_registry']) if spec.get('scene_registry') else None
    if registry is not None:
        dependencies.append(spec['scene_registry'])
    seen = set()
    for decision in rows:
        if type(decision.get("accepted")) is not bool:
            raise ValueError("Missing explicit acceptance")
        if not decision["accepted"]:
            continue
        uid = decision["id"]
        if decision.get("selection_mode", "idm_vlm") != spec.get("selection_mode", "idm_vlm"):
            raise ValueError("Decision selection mode differs from export recipe")
        if oracle:
            from evo_wam.verification.oracle import endpoint
            if sha(decision['oracle_result']) != decision['oracle_result_sha256']:
                raise ValueError('Oracle receipt changed')
            if endpoint(read(decision['oracle_result']), decision['accepted_step'])[0] != decision['accepted_step']:
                raise ValueError('Export differs from Oracle first success')
            dependencies.append(decision['oracle_result'])
        if uid in seen:
            raise ValueError("Duplicate selected candidate")
        seen.add(uid)
        row = decision["source"]
        scene = training_scene_identity(row, registry)
        t = decision["accepted_step"]
        reader = NativeFrames(row, dict(backbone="cosmos3-robotwin"))
        if not 1 <= t <= reader.length:
            raise ValueError("Invalid selected endpoint")
        if not no_idm and (
            decision["idm"]["endpoint"] != t
            or not np.isfinite(decision["idm"]["mse"])
            or decision["idm"]["mse"] > spec["threshold"]
        ):
            raise ValueError("IDM rejection cannot be exported")
        if not vlm_only and not Protocol("robotwin_dense").contract.prefix_gate(
            reader.actions, reader.state, t
        )["pass"]:
            raise ValueError("Action gate rejection")
        groups = decision["vote_groups"]
        if (
            len(groups) != 1
            or groups[0]["endpoint"] != t
            or len(groups[0]["votes"]) != (1 if oracle else 3)
            or sum(v["accepted"] for v in groups[0]["votes"]) < (1 if oracle else 2)
        ):
            raise ValueError("Expected fixed-endpoint majority")
        for v in groups[0]["votes"]:
            record = v["record"]
            description = record["description"]
            judgment = record["judgment"]
            protocol = Protocol("robotwin_dense")
            dp, error = protocol.parse(description["response"], "description")
            if error or dp != description["prediction"]:
                raise ValueError("Description receipt changed")
            jp, error = protocol.parse(judgment["response"], "judgment", dp)
            if (
                error
                or jp != judgment["prediction"]
                or v["accepted"] != (jp["decision"] == "accept")
            ):
                raise ValueError("Judgment receipt changed")
        case = Path(row["case"])
        generation = case / "generation_manifest.json"
        g = read(generation)
        if (
            sha(case / "actions_qpos.npy") != decision["actions_sha256"]
            or sha(generation) != decision["generation_sha256"]
        ):
            raise ValueError("Generation changed after verification")
        if (
            g["recurrence_space"] != "latent"
            or g["decoded_rgb_used_for_recurrence"] is not False
        ):
            raise ValueError("Invalid recurrence")
        for k, ch in enumerate(g["chunks"]):
            if (
                k
                and ch["input_history_latent_sha256"]
                != g["chunks"][k - 1]["output_history_suffix_sha256"]
            ):
                raise ValueError("Broken recurrence lineage")
        slot = hashlib.sha256(uid.encode()).hexdigest()[:24]
        ep = dataset / "episodes" / slot
        ep.mkdir(parents=True, exist_ok=False)
        video = ep / "video.mp4"
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "640x360",
            "-r",
            "15",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
            str(video),
        ]
        h = hashlib.sha256()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for k in range(t + 1):
                frame = reader.initial if k == 0 else reader.frames[k - 1]
                if frame.shape == (352, 640, 3):
                    frame = np.pad(frame, ((0, 8), (0, 0), (0, 0)), mode="edge")
                if frame.shape != (360, 640, 3) or frame.dtype != np.uint8:
                    raise ValueError("Invalid native RGB geometry")
                payload = frame.tobytes()
                h.update(payload)
                proc.stdin.write(payload)
            proc.stdin.close()
            if proc.wait(timeout=180):
                raise RuntimeError(proc.stderr.read().decode())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        commands = reader.actions[:t].astype(np.float32)
        states = np.concatenate([reader.state[None], commands])
        actions = np.concatenate([commands, commands[-1:]])
        path = ep / "data.parquet"
        pq.write_table(
            pa.table(
                {"observation.state": states.tolist(), "action": actions.tolist()}
            ),
            path,
        )
        check = pq.read_table(path)
        np.testing.assert_array_equal(
            np.asarray(check["observation.state"].to_pylist(), np.float32), states
        )
        np.testing.assert_array_equal(
            np.asarray(check["action"].to_pylist(), np.float32), actions
        )
        marker = ep / "EXPORT_COMPLETE.json"
        write(
            marker,
            dict(
                passed=True,
                source="portable_online_verification",
                candidate_id=uid,
                prefix_steps=t,
                length=t + 1,
                training_rgb_sha256=h.hexdigest(),
                file_sha256={p.name: sha(p) for p in (path, video)},
                decision_sha256=sha(source),
                generation_sha256=sha(generation),
                terminal_action_row="alignment placeholder; not additional execution",
                virtual_tail_steps=30,
                official_success=True if oracle else None,
                replay_status="completed" if oracle else "not_run",
            ),
        )
        task_config = row["task_config"]
        if task_config not in ("demo_clean", "demo_randomized"):
            raise ValueError("Unrecognized RoboTwin task configuration")
        index.append(
            dict(
                split="Clean" if task_config == "demo_clean" else "Randomized",
                task=reader.task,
                episode_index=len(index),
                caption=reader.instruction,
                length=t + 1,
                fps=15,
                video_path=str(video),
                source_data_path=str(path),
                source_logical_slot_id=uid,
                **scene,
                accepted_step=t,
                generated_training_actions=t,
                round=context["round"],
                idm_mse=None if no_idm else decision["idm"]["mse"],
                selection_mode=spec.get("selection_mode", "idm_vlm"),
                training_target="self_imagine_video_and_actions",
                state_source="initial_real_qpos_then_previous_generated_action",
                replay_status="completed" if oracle else "not_run",
                official_whole_replay_success=True if oracle else None,
                unreplayed_prefix=not oracle,
                actual_replay_steps=t if oracle else 0,
                export_marker_sha256=sha(marker),
            )
        )
        artifacts.append(ep)
        dependencies.extend(map(str, reader.paths))
        dependencies.append(str(generation))
    if not index:
        raise ValueError("No accepted native prefixes; training cannot proceed")
    return publish(context, index, artifacts, dependencies)
