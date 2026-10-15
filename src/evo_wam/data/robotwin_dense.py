"""Native Cosmos RoboTwin lossless export and dataset metadata, without dispatch."""

import collections, json, shutil, sys, os
from pathlib import Path
from evo_wam.io import read, write, sha
from evo_wam.stages import json_rows, complete


def training_universe(template, index):
    authorized = template.get('authorized_tasks')
    if not isinstance(authorized,list) or len(set(authorized)) != 7:
        raise ValueError('Native scene-balanced training requires an explicit seven-task universe')
    if not {r['task'] for r in index}.issubset(authorized):
        raise ValueError('Export contains a task outside the declared native training universe')
    return sorted(authorized)


def export(context):
    from . import robotwin_dense_native as native

    spec = context["spec"]
    out = Path(context["output"])
    dataset = out / "dataset"
    sys.path.insert(0, spec["source_root"])
    native.D = dataset
    native.V = Path(spec["verifier_root"])
    native.CFG = dict(threshold=spec["threshold"], reuse_dataset_root=None)
    rows = json_rows(spec["selected_prefixes"])
    index = []
    artifacts = []
    dependencies = []
    if not rows:
        raise ValueError("No approved native prefixes")
    if len({r["logical_slot_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate native prefix identity")
    for row in rows:
        native.export(row)
        ep = dataset / "episodes" / row["logical_slot_id"]
        t = row["training_prefix_steps"]
        index.append(
            dict(
                split="Clean" if row["task_config"] == "demo_clean" else "Randomized",
                task=row["task_name"],
                episode_index=len(index),
                caption=row["instruction"],
                length=t + 1,
                fps=15,
                video_path=str(ep / "video.mp4"),
                source_data_path=str(ep / "data.parquet"),
                episode_key=row["episode_key"],
                scene_set="A",
                repeat_index=row["repeat_index"],
                fixed_seed=row["fixed_seed"],
                sample_index=row["sample_index"],
                inference_noise_seed=row["inference_noise_seed"],
                source_logical_slot_id=row["logical_slot_id"],
                source_directory=row["directory"],
                accepted_step=t,
                generated_training_actions=t,
                actual_replay_steps=0,
                unreplayed_prefix=True,
                replay_status="not_run",
                official_whole_replay_success=None,
                earlier_missing=row["verifier"]["earlier_missing"],
                training_target="self_imagine_video_and_actions",
                state_source=row["training_state_source"],
                source_actions_sha256=row["verifier"]["generated_actions_sha256"],
                verifier_moment_sha256=row["verifier"]["last_moment_sha256"],
                export_marker_sha256=sha(ep / "EXPORT_COMPLETE.json"),
                round=row["round"],
                branch=row["branch"],
                idm_mse=row["idm"]["mse"],
            )
        )
        artifacts.append(ep)
        capture = Path(row["directory"]) / "client_rollout"
        dependencies += [
            str(capture / name)
            for name in (
                "actions_qpos.npy",
                "imagined_frames.npz",
                "generation_manifest.json",
            )
        ]
        dependencies += [
            row["initial_observation_path"],
            str(Path(row["directory"]) / "result.json"),
            str(native.V / "cases" / row["verifier"]["id"] / f"t{t:04d}_moment.json"),
            str(native.V / "cases" / row["verifier"]["id"] / "source_audit.json"),
        ]
    return publish(context, index, artifacts, dependencies)


def publish(context, index, artifacts, dependencies):
    spec = context["spec"]
    out = Path(context["output"])
    dataset = out / "dataset"
    history_refilter = spec.get("history_refilter", False)
    prior_index, source_indices = [], []
    if spec.get("previous_pool"):
        prior = read(spec["previous_pool"])
        prior_index = json_rows(Path(spec["previous_pool"]).parent / "meta/index.jsonl")
        dependencies.extend(
            [
                spec["previous_pool"],
                str(Path(spec["previous_pool"]).parent / "meta/index.jsonl"),
            ]
        )
        if prior["statistics"]["sha256"] != sha(spec["statistics"]):
            raise ValueError("Cumulative pool normalization changed")
        if prior.get("selection_mode", "idm_vlm") != spec.get("selection_mode", "idm_vlm"):
            raise ValueError("Cannot combine pools with different selection modes")
        if {r["source_logical_slot_id"] for r in prior_index}.intersection(
            r["source_logical_slot_id"] for r in index
        ):
            raise ValueError("Duplicate cumulative native prefix")
        if history_refilter:
            if (
                not prior.get("history_refilter")
                or prior["idm_threshold"] < spec["threshold"]
            ):
                raise ValueError(
                    "Progressive history requires a monotone previous training view"
                )
            source_indices = prior["original_export_indices"]
        else:
            index = prior_index + index
        dependencies += [
            r[k] for r in prior_index for k in ("video_path", "source_data_path")
        ]
    if history_refilter:
        from .progressive import select

        if context["round"] > 1 and not spec.get("previous_pool"):
            raise ValueError("Progressive refilter requires the previous round pool")
        original_index = dataset / "meta/original_export.jsonl"
        original_index.parent.mkdir(parents=True, exist_ok=True)
        original_index.write_text("".join(json.dumps(r) + "\n" for r in index))
        if [r["round"] for r in source_indices] != list(range(1, context["round"])):
            raise ValueError("Missing progressive original export indices")
        pools = []
        for source in source_indices:
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("Original export index changed")
            pools.append((source["round"], json_rows(source["path"])))
            dependencies.append(source["path"])
        pools.append((context["round"], index))
        index, ledger, summary = select(
            context["round"], pools, prior_index, spec["threshold"]
        )
        source_indices = [
            *source_indices,
            dict(
                round=context["round"],
                path=str(original_index),
                sha256=sha(original_index),
            ),
        ]
        decision_path = dataset / "REFILTER_DECISIONS.jsonl"
        decision_path.write_text("".join(json.dumps(r) + "\n" for r in ledger))
        summary_path = dataset / "REFILTER_SUMMARY.json"
        write(
            summary_path,
            dict(
                summary,
                round=context["round"],
                threshold=spec["threshold"],
                policy="delete_original_accepted_prefix_only",
                decisions_sha256=sha(decision_path),
            ),
        )
        artifacts += [original_index, decision_path, summary_path]
    for i, row in enumerate(index):
        row["episode_index"] = i
    (dataset / "meta").mkdir(parents=True, exist_ok=True)
    index_path = dataset / "meta/index.jsonl"
    index_path.write_text("".join(json.dumps(r) + "\n" for r in index))
    stats = dataset / "state_action_stats.json"
    shutil.copy2(spec["statistics"], stats)
    template = read(spec["loader_template"])
    if (
        template["bootstrap_sampling_slots"] != 32
        or template["terminal_hold_steps"] != 30
    ):
        raise ValueError("Unexpected historical Cosmos RoboTwin loader protocol")
    authorized = training_universe(template, index)
    manifest = dict(
        schema_version="robotwin.dense.vlm_prefix.imagined_sft.v1",
        training_view=True,
        source_manifest=str(dataset / "ADAPTATION_MANIFEST.json"),
        episodes=len(index),
        authorized_tasks=sorted(authorized),
        training_target="self_imagine_video_and_actions",
        statistics=dict(path=str(stats), sha256=sha(stats)),
        loader_kwargs=template["loader_kwargs"],
        raw_loader_kwargs=template["raw_loader_kwargs"],
        bootstrap_sampling_slots=32,
        terminal_hold_steps=30,
        selection=spec["selection_description"],
        selection_mode=spec.get("selection_mode", "idm_vlm"),
        idm_threshold=spec["threshold"],
        replay_status="completed" if spec.get("selection_mode") == "oracle_vlm" else "not_run",
        official_success=True if spec.get("selection_mode") == "oracle_vlm" else None,
        metadata_sha256={"meta/index.jsonl": sha(index_path)},
        task_config_counts=dict(
            collections.Counter(r["task"] + "/" + r["split"] for r in index)
        ),
    )
    target = dataset / "ADAPTATION_MANIFEST.json"
    if history_refilter:
        manifest.update(history_refilter=True, original_export_indices=source_indices)
    write(target, manifest)
    # READY is published only after the native reader has validated the metadata.
    from cosmos_framework.data.vfm.action.datasets.robotwin_preprocessed_dataset import (
        RobotWinPreprocessedVideoDataset,
    )

    previous_stats = os.environ.get("ROBOTWIN_STATE_ACTION_STATS_PATH")
    os.environ["ROBOTWIN_STATE_ACTION_STATS_PATH"] = str(stats)
    try:
        loader = RobotWinPreprocessedVideoDataset(
            root=dataset, **manifest["raw_loader_kwargs"]
        )
    finally:
        if previous_stats is None:
            os.environ.pop("ROBOTWIN_STATE_ACTION_STATS_PATH", None)
        else:
            os.environ["ROBOTWIN_STATE_ACTION_STATS_PATH"] = previous_stats
    if len(loader.episodes) != len(index) or len(loader) == 0:
        raise ValueError("Native loader rejected exported pool")
    ready = dataset / "READY.json"
    write(
        ready,
        dict(
            passed=True,
            episodes=len(index),
            windows=len(loader),
            manifest_sha256=sha(target),
            statistics_sha256=sha(stats),
            validation_scope="metadata and native dataset construction; RGB samples tested separately",
        ),
    )
    complete(
        context,
        artifacts + [target, index_path, stats, ready],
        manifest=str(target),
        dependency_paths=sorted(set(dependencies)),
    )
