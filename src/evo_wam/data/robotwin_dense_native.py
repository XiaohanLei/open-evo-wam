"""Frozen dense RoboTwin materializer, with dataset/verifier/threshold supplied by the stage.

Source: the native RoboTwin export worker.
Pixel packaging, state/action rows and scientific assertions are preserved.
"""

from pathlib import Path
import json, hashlib, fcntl, sys, zipfile, subprocess, time, os, shutil
import numpy as np, h5py
import pyarrow as pa, pyarrow.parquet as pq
from evo_wam.io import sha

arrsha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(x))
    q.replace(p)


def shape_header(stream):
    v = np.lib.format.read_magic(stream)
    if v == (1, 0):
        shape, f, d = np.lib.format.read_array_header_1_0(stream)
    elif v == (2, 0):
        shape, f, d = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported NPY header")
    assert not f and d == np.uint8 and len(shape) == 4 and shape[-1] == 3
    return shape


def export(row):
    slot = row["logical_slot_id"]
    out = D / "episodes" / slot
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "export.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    t = row["training_prefix_steps"]
    a = row["verifier"]
    source = Path(row["directory"])
    c = source / "client_rollout"
    moment = V / "cases" / a["id"] / f"t{t:04d}_moment.json"
    assert (
        row["scene_set"] == "A"
        and (0 <= row["repeat_index"] < 20 or 40 <= row["repeat_index"] < 120)
        and row["mode"] == "self_imagine"
    )
    assert row["replay_status"] == "not_run" and row["official_success"] is None
    assert row["idm"]["mse"] <= CFG["threshold"] and row["idm"]["endpoint"] == t
    assert (
        a["accepted_step"] == t
        and a["state"] == "accepted"
        and a["actions_range"] == [0, t]
        and a["observations_range"] == [0, t + 1]
    )
    assert (
        sha(source / "result.json") == a["source_result_sha256"]
        and sha(moment) == a["last_moment_sha256"]
    )
    m = json.loads(moment.read_text())
    assert (
        m["gate"]["pass"]
        and m["future_frame_index"] == t - 1
        and m["decision"] == "accept"
    )
    assert (
        len(m["votes"]) == 3 and sum(v["decision"] == "accept" for v in m["votes"]) >= 2
    )
    for vote in m["votes"]:
        if vote["decision"] == "accept":
            assert all(
                x["value"] == "yes"
                for x in vote["judgment"]["prediction"]["checks"].values()
            )
    audit = json.loads((V / "cases" / a["id"] / "source_audit.json").read_text())
    gen = json.loads((c / "generation_manifest.json").read_text())
    assert sha(c / "generation_manifest.json") == audit["generation_sha256"]
    assert (
        gen["recurrence_space"] == "latent"
        and gen["decoded_rgb_used_for_recurrence"] is False
    )
    q = np.load(c / "actions_qpos.npy")
    assert q.ndim == 2 and q.shape[1] == 14 and np.isfinite(q).all() and 0 < t <= len(q)
    assert arrsha(q) == a["generated_actions_sha256"] == gen["actions_sha256"]
    for prev, cur in zip(gen["chunks"], gen["chunks"][1:]):
        assert (
            prev["output_history_suffix_sha256"] == cur["input_history_latent_sha256"]
        )
    cond = json.loads((source / "initial_condition.json").read_text())
    q0 = np.asarray(cond["state"], dtype=np.float32)
    assert q0.shape == (14,) and np.isfinite(q0).all()
    assert sha(row["initial_observation_path"]) == row["initial_observation_sha256"]
    from evo_wam.data.robotwin_canvas import observation_image

    with np.load(row["initial_observation_path"]) as z:
        assert np.array_equal(q0, z["q0"])
        initial = observation_image(
            {
                "observation": {
                    k[:-4]: {"rgb": z[k].copy()} for k in z.files if k.endswith("_rgb")
                },
                "joint_action": {"vector": q0},
            }
        )
    assert (
        initial.shape == (360, 640, 3) and arrsha(initial) == cond["global_rgb_sha256"]
    )
    identity = {
        "source_generation_sha256": sha(c / "generation_manifest.json"),
        "selected_moment_sha256": sha(moment),
        "prefix_steps": t,
        "generated_actions_sha256": arrsha(q),
    }
    marker = out / "EXPORT_COMPLETE.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        assert old["identity"] == identity
        for name, digest in old["file_sha256"].items():
            assert sha(out / name) == digest
        return old
    # Rebind unchanged data bytes to the new vote3 selection evidence.
    reuse_root = CFG.get("reuse_dataset_root")
    previous = Path(reuse_root) / "episodes" / slot if reuse_root else None
    if previous and (previous / "EXPORT_COMPLETE.json").exists():
        old = json.loads((previous / "EXPORT_COMPLETE.json").read_text())
        same = all(
            old["identity"].get(k) == identity[k]
            for k in [
                "source_generation_sha256",
                "prefix_steps",
                "generated_actions_sha256",
            ]
        )
        if same:
            assert old["native_last_rgb_sha256"] == m["native_frame_sha256"] and old[
                "initial_rgb_sha256"
            ] == arrsha(initial)
            assert old["prefix_steps"] == t and old["length"] == t + 1
            for name, digest in old["file_sha256"].items():
                assert sha(previous / name) == digest
                target = out / name
                if target.exists():
                    assert sha(target) == digest
                else:
                    try:
                        os.link((previous / name).resolve(), target)
                    except OSError:
                        shutil.copy2(previous / name, target)
                assert sha(target) == digest
            result = dict(
                old,
                identity=identity,
                time=time.time(),
                reused_content_from=str(previous),
                reused_marker_sha256=sha(previous / "EXPORT_COMPLETE.json"),
                selection_rebound_to_vote3=True,
            )
            write(marker, result)
            return result
    stamp = c / "imagined_frames.npz"
    before = (stamp.stat().st_size, stamp.stat().st_mtime_ns)
    with zipfile.ZipFile(stamp) as archive:
        with archive.open("frames.npy") as stream:
            shape = shape_header(stream)
            assert (
                shape[0] == 64 * len(gen["chunks"])
                and shape[1:] in [(352, 640, 3), (360, 640, 3)]
                and t <= shape[0]
            )
            n = t * int(np.prod(shape[1:]))
            raw = stream.read(n)
            assert len(raw) == n
            future = np.frombuffer(raw, dtype=np.uint8).reshape(t, *shape[1:])
    assert (stamp.stat().st_size, stamp.stat().st_mtime_ns) == before
    assert arrsha(future[-1]) == m["native_frame_sha256"], (
        "selected frame mismatch between full NPZ and verifier chunk"
    )
    # Compare each consumed chunk's two endpoint frames with its native chunk artifact.
    chunk_checks = []
    for k in range((t + 63) // 64):
        f = c / "chunks" / f"chunk_{k:03d}" / "future_rgb.npz"
        used = min(64, t - 64 * k)
        if f.exists():
            expected = audit["stamps"][str(f.relative_to(c))]
            st = f.stat()
            assert (
                st.st_size == expected["size"]
                and st.st_mtime_ns == expected["mtime_ns"]
            )
            with np.load(f) as z:
                chunk = z["frames"]
                assert np.array_equal(future[k * 64], chunk[0])
                assert np.array_equal(future[k * 64 + used - 1], chunk[used - 1])
            chunk_checks.append(k)
    temp = out / "video.partial.mp4"
    temp.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [
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
            str(temp),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rgb_sha = hashlib.sha256()
    pad = np.empty((360, 640, 3), np.uint8)
    try:
        for i in range(t + 1):
            if i == 0:
                frame = initial
            elif shape[1] == 360:
                frame = future[i - 1]
            else:
                pad[:352] = future[i - 1]
                pad[352:] = future[i - 1, -1:]
                frame = pad
            block = frame.tobytes()
            rgb_sha.update(block)
            proc.stdin.write(block)
        proc.stdin.close()
        rc = proc.wait(timeout=180)
        if rc:
            raise RuntimeError("ffmpeg error: " + proc.stderr.read().decode()[-2000:])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate",
                "-of",
                "json",
                str(temp),
            ]
        )
    )["streams"][0]
    assert (
        int(info["nb_frames"]),
        info["height"],
        info["width"],
        info["avg_frame_rate"],
    ) == (t + 1, 360, 640, "15/1")
    commands = q[:t].astype(np.float32, copy=False)
    states = np.concatenate([q0[None], commands])
    actions = np.concatenate([commands, commands[-1:]])
    pq.write_table(
        pa.table({"observation.state": states.tolist(), "action": actions.tolist()}),
        out / "data.partial.parquet",
    )
    check = pq.read_table(out / "data.partial.parquet")
    assert np.array_equal(
        np.asarray(check["observation.state"].to_pylist(), np.float32), states
    )
    assert np.array_equal(np.asarray(check["action"].to_pylist(), np.float32), actions)
    temp.replace(out / "video.mp4")
    (out / "data.partial.parquet").replace(out / "data.parquet")
    result = {
        "logical_slot_id": slot,
        "id": a["id"],
        "task": row["task_name"],
        "config": row["task_config"],
        "length": t + 1,
        "prefix_steps": t,
        "identity": identity,
        "training_rgb_sha256": rgb_sha.hexdigest(),
        "initial_rgb_sha256": arrsha(initial),
        "native_last_rgb_sha256": arrsha(future[-1]),
        "chunk_endpoint_checks": chunk_checks,
        "imagined_native_shape": list(shape),
        "file_sha256": {
            name: sha(out / name) for name in ["video.mp4", "data.parquet"]
        },
        "bytes": sum(
            (out / name).stat().st_size for name in ["video.mp4", "data.parquet"]
        ),
        "official_whole_replay_success": None,
        "replay_status": "not_run",
        "unreplayed_prefix": a["unreplayed"],
        "training_state_source": row["training_state_source"],
        "fps": 15,
        "real_future_training_frames": 0,
        "terminal_action_row": "last command repeated as observation-alignment placeholder, not extra execution",
        "virtual_tail_steps": 30,
        "time": time.time(),
    }
    write(marker, result)
    return result
