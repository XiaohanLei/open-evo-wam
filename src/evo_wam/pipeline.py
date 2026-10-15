"""Sequential stage execution. No resource allocation, queues or remote dispatch."""

import datetime, fcntl, json, os, subprocess, sys, traceback
from pathlib import Path
from .io import read, write, digest, identity, expand

STAGES = ("generate", "verify", "export-prefix", "train", "evaluate")
OTHER = ("prepare-data", "train-r0", "idm-train", "idm-calibrate", "oracle-replay")


def load_config(path):
    path = Path(path).resolve()
    if path.suffix == ".json":
        config = read(path)
    else:
        import yaml

        config = yaml.safe_load(path.read_text())
    if config.get("schema") != "evo-wam.pipeline.v1":
        raise ValueError("Unsupported config schema")
    variables = dict(
        os.environ,
        CONFIG_DIR=str(path.parent),
        PACKAGE_ROOT=str(Path(__file__).resolve().parents[2]),
    )
    variables.update(config.get("variables", {}))
    # Defer stage/round/output variables until a stage is constructed.
    for key in (
        "ROUND",
        "RUN_DIR",
        "STAGE_DIR",
        "PARENT_CHECKPOINT",
        "CURRENT_CHECKPOINT",
        "PREVIOUS_STAGE_DIR",
    ):
        variables[key] = "${" + key + "}"

    def early(v):
        if isinstance(v, str):
            # Config variables may reference built-ins regardless of insertion order.
            for _ in range(len(variables) + 1):
                previous = v
                for k, x in variables.items():
                    v = v.replace("${" + k + "}", str(x))
                if v == previous:
                    break
            return v
        if isinstance(v, list):
            return [early(x) for x in v]
        if isinstance(v, dict):
            return {k: early(x) for k, x in v.items()}
        return v

    config = early(config)
    config["_config_path"] = str(path)
    return config


def completed_result(path):
    """A result is consumable only after its producer committed a valid receipt."""
    path = Path(path)
    receipt = path.parent / "stage.json"
    if not path.is_file() or not receipt.is_file():
        raise FileNotFoundError("Completed stage receipt required: " + str(path.parent))
    state = read(receipt)
    if state.get("status") != "complete":
        raise ValueError("Upstream stage is not complete: " + str(path.parent))
    outputs = state.get("outputs", {})
    if str(path) not in outputs:
        raise ValueError("Upstream receipt does not bind result: " + str(path))
    for filename, expected in {**outputs, **state.get("dependencies", {})}.items():
        if identity(filename) != expected:
            raise ValueError("Upstream artifact changed: " + filename)
    result = read(path)
    if result.get("status") != "complete":
        raise ValueError("Upstream result is not complete: " + str(path))
    return result


def parent_checkpoint(config, round_number, run_dir):
    explicit = config.get("round_parent_checkpoints", {}).get(str(round_number))
    if explicit is not None:
        if not Path(explicit).exists():
            raise ValueError("Explicit round parent checkpoint does not exist")
        return explicit
    if round_number <= 1:
        return config.get("initial_checkpoint", "")
    receipt = run_dir / f"R{round_number - 1:02d}" / "train" / "result.json"
    if not receipt.exists():
        raise FileNotFoundError("Previous round train result required: " + str(receipt))
    parent = completed_result(receipt).get("checkpoint")
    if not parent or not Path(parent).exists():
        raise ValueError("Training result has no existing checkpoint")
    return parent


def stage_input_paths(spec):
    """Include nested sequential generation dependencies in stage cache identities."""
    inputs = list(spec.get("inputs", []))
    for name in (
        "jobs",
        "manifest",
        "decisions",
        "requests",
        "statistics",
        "train_config",
        "sampling_config",
        "bindings_file",
        "registry",
        "scenes",
        "episode_results",
        "scores",
        "previous_pool",
        "checkpoint",
        "selected_prefixes",
        "base_manifest",
        "model_config",
        "tokenizer",
        "vlm_decisions",
        "prefix_scores",
        "loader_template",
        "idm_checkpoint",
        "generated_manifest",
        "native_source_root",
        "utility_source_root",
        "agibot_checkpoint",
        "a72_config",
        "initial_checkpoint",
        "idm_prepared_dir",
        "admission_policy",
        "threshold_schedule",
        "calibration_scores",
        "evidence_manifest",
        "oracle_results",
        "generation_core_source",
        "assets_manifest",
    ):
        if spec.get(name):
            inputs.append(spec[name])
    if spec.get("source_root"):
        inputs.append(spec["source_root"])
    # Immutable public datasets may supply an explicit metadata/version identity.
    # Native readers still check the files they consume; no implicit override.
    asset_identities = spec.get("asset_identity_paths", {})
    if set(asset_identities) - set(spec.get("asset_paths", {})):
        raise ValueError("Asset identity override has no matching asset binding")
    inputs.extend(asset_identities.get(key, value) for key, value in spec.get("asset_paths", {}).items())
    inputs.extend(spec.get("extra_python_paths", []))
    if spec.get("oracle_replay"):
        inputs.extend(stage_input_paths(spec["oracle_replay"]))
    if spec.get("replenish_generation"):
        child = spec["replenish_generation"]
        inputs.extend(stage_input_paths(child))
        inputs.extend(child.get("code_inputs", []))
    return inputs


def run_stage(config, stage, round_number=1, dry_run=False):
    if stage not in STAGES + OTHER:
        raise ValueError("Unknown stage " + stage)
    if type(round_number) is not int or round_number < 1:
        raise ValueError("Round must be positive")
    run_dir = Path(expand(config["run_dir"], {})).expanduser().resolve()
    directory = run_dir / f"R{round_number:02d}" / stage
    missing = []
    try:
        parent = parent_checkpoint(config, round_number, run_dir)
    except (FileNotFoundError, ValueError) as exc:
        raw_spec = config["stages"][stage]
        independent_without_parent = (
            isinstance(exc, FileNotFoundError)
            and (
                stage not in ("generate", "train", "evaluate")
                or (stage == "evaluate" and raw_spec.get("checkpoint"))
            )
            and not raw_spec.get("requires_previous", True)
            and "${PARENT_CHECKPOINT}" not in json.dumps(raw_spec)
            and not raw_spec.get("requires_parent", False)
        )
        if independent_without_parent:
            parent = ""
        elif not dry_run:
            raise
        else:
            parent = "<pending previous round checkpoint>"
            missing.append(str(exc))
    previous = (
        STAGES[STAGES.index(stage) - 1]
        if stage in STAGES and STAGES.index(stage) > 0
        else None
    )
    current_checkpoint = parent
    if stage == "evaluate":
        trained = directory.parent / "train" / "result.json"
        explicit_evaluation = (
            config["stages"][stage].get("checkpoint")
            if not config["stages"][stage].get("requires_previous", True)
            else None
        )
        if explicit_evaluation:
            current_checkpoint = expand(
                explicit_evaluation,
                {"PARENT_CHECKPOINT": parent, "ROUND": round_number},
            )
            if not Path(current_checkpoint).exists():
                raise ValueError("Explicit evaluation checkpoint does not exist")
        elif trained.exists():
            current_checkpoint = completed_result(trained).get("checkpoint")
            if not current_checkpoint or not Path(current_checkpoint).exists():
                raise ValueError(
                    "Current round train result has no existing checkpoint"
                )
        elif dry_run:
            current_checkpoint = "<pending current round checkpoint>"
            missing.append(str(trained))
        else:
            raise FileNotFoundError(
                "Current round train result required: " + str(trained)
            )
    variables = dict(
        ROUND=round_number,
        RUN_DIR=str(run_dir),
        STAGE_DIR=str(directory),
        PARENT_CHECKPOINT=parent,
        CURRENT_CHECKPOINT=current_checkpoint,
        PREVIOUS_STAGE_DIR=str(directory.parent / previous) if previous else "",
    )
    spec = expand(config["stages"][stage], variables)
    if spec.get("threshold_schedule"):
        from evo_wam.data.progressive import threshold_record

        record = threshold_record(
            spec["threshold_schedule"], spec["calibration_scores"], round_number
        )
        if "threshold" in spec and spec["threshold"] != record["threshold"]:
            raise ValueError("Explicit threshold differs from progressive schedule")
        spec.update(threshold=record["threshold"], percentile=record["percentile"])
    if stage == "export-prefix" and spec.get("cumulative", False) and round_number > 1:
        previous_pool = (
            run_dir / f"R{round_number - 1:02d}" / "export-prefix" / "result.json"
        )
        if previous_pool.exists():
            pool = completed_result(previous_pool).get("manifest")
            if not pool or not Path(pool).is_file():
                raise ValueError("Previous round export has no existing pool manifest")
            if spec.get("previous_pool") and spec["previous_pool"] != pool:
                raise ValueError("Explicit pool differs from previous round export")
            spec["previous_pool"] = pool
        elif dry_run:
            missing.append(str(previous_pool))
        else:
            raise FileNotFoundError(
                "Previous round pool required: " + str(previous_pool)
            )
    inputs = stage_input_paths(spec)
    if stage == "evaluate" and current_checkpoint:
        inputs.append(current_checkpoint)
    if (stage in ("generate", "train") or spec.get("requires_parent")) and parent:
        inputs.append(parent)
    if previous and spec.get("requires_previous", True):
        previous_result = directory.parent / previous / "result.json"
        inputs.append(str(previous_result))
        if previous_result.exists():
            upstream = completed_result(previous_result)
            inputs.extend(upstream.get("artifacts", []))
            inputs.extend(upstream.get("dependency_paths", []))
    # Small interchange manifests name their data explicitly. Include these files
    # so an unchanged manifest cannot hide a changed initial observation/trajectory.
    reference_keys = {
        "path",
        "input",
        "trajectory",
        "case",
        "initial",
        "initial_observation",
        "initial_observation_path",
        "tensor",
        "video",
        "start",
    }

    from evo_wam.path_bindings import PathBindings

    def references(obj, bindings, key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from references(v, bindings, k)
        elif isinstance(obj, list):
            for v in obj:
                yield from references(v, bindings, key)
        elif isinstance(obj, str) and key in reference_keys and Path(obj).is_absolute():
            yield bindings.path(obj)

    for manifest in list(inputs):
        p = Path(manifest)
        if (
            p.is_file()
            and (p.suffix in (".json", ".jsonl") or p.name.endswith(".json.gz"))
            and p.stat().st_size <= 32 * 1024 * 1024
        ):
            try:
                objects = (
                    [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
                    if p.suffix == ".jsonl"
                    else [read(p)]
                )
                for obj in objects:
                    bindings = PathBindings(
                        obj.get("path_bindings", []) if isinstance(obj, dict) else [],
                        spec.get("path_bindings", []),
                    )
                    inputs.extend(references(obj, bindings))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError("Invalid input manifest: " + str(p))
    source_files = [Path(__file__).parent]
    source_files.extend(Path(p) for p in spec.get("code_inputs", []))
    identities = {}
    for p in sorted(set(inputs + [str(p) for p in source_files])):
        try:
            identities[p] = identity(p)
        except FileNotFoundError:
            missing.append(p)
    key = digest(
        dict(
            schema=1,
            stage=stage,
            round=round_number,
            spec=spec,
            config=config,
            inputs=identities,
            parent=parent,
        )
    )
    plan = dict(
        stage=stage,
        round=round_number,
        output=str(directory),
        parent_checkpoint=parent,
        current_checkpoint=current_checkpoint,
        missing_dependencies=missing,
        cache_identity=key,
        executor=spec.get("executor", "builtin"),
        command=spec.get("argv"),
    )
    if dry_run:
        return plan
    if missing:
        raise FileNotFoundError("Missing dependencies: " + ", ".join(missing))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = directory / "stage.json"
        if receipt.exists():
            old = read(receipt)
            if old.get("cache_identity") != key:
                raise ValueError(
                    "Inputs/config/code changed. Use a new run_dir; stale output is not reused."
                )
            if old.get("status") == "complete":
                for p, wanted in {
                    **old["outputs"],
                    **old.get("dependencies", {}),
                }.items():
                    if identity(p) != wanted:
                        raise ValueError("Completed stage output changed: " + p)
                return dict(plan, status="reused")
        state = dict(
            plan,
            status="running",
            started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            input_identities=identities,
        )
        # Retain failed-attempt evidence, but require this invocation to emit a new result.
        prior_result = directory / "result.json"
        if prior_result.exists():
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%S%f"
            )
            prior_result.rename(directory / ("result.previous-" + stamp + ".json"))
        write(receipt, state)
        context = dict(
            config=config,
            spec=spec,
            stage=stage,
            round=round_number,
            output=str(directory),
            parent_checkpoint=parent,
            current_checkpoint=current_checkpoint,
            previous_stage=str(directory.parent / previous) if previous else None,
        )
        write(directory / "context.json", context)
        try:
            if spec.get("executor", "builtin") == "command":
                argv = spec["argv"]
                if (
                    not isinstance(argv, list)
                    or not argv
                    or not all(isinstance(x, str) for x in argv)
                ):
                    raise ValueError("argv must be a nonempty string list")
                env = dict(
                    os.environ,
                    **{k: str(v) for k, v in spec.get("env", {}).items()},
                    EVO_WAM_CONTEXT=str(directory / "context.json"),
                )
                with (directory / "execution.log").open("a") as log:
                    subprocess.run(
                        argv,
                        cwd=spec.get("cwd"),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            elif spec.get("executor", "builtin") == "builtin":
                if spec.get("python"):
                    env = dict(
                        os.environ,
                        **{k: str(v) for k, v in spec.get("env", {}).items()},
                    )
                    package_src = str(Path(__file__).resolve().parents[1])
                    env["PYTHONPATH"] = (
                        package_src + os.pathsep + env.get("PYTHONPATH", "")
                    )
                    with (directory / "execution.log").open("a") as log:
                        subprocess.run(
                            [
                                spec["python"],
                                "-m",
                                "evo_wam.worker",
                                str(directory / "context.json"),
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                else:
                    from .stages import execute

                    execute(context)
            else:
                raise ValueError("Unsupported executor")
            result = read(directory / "result.json")
            if result.get("status") != "complete":
                raise ValueError("Stage must produce a complete result.json")
            outputs = [str(directory / "result.json")] + result.get("artifacts", [])
            if stage in ("train", "train-r0", "idm-train"):
                if not result.get("checkpoint"):
                    raise ValueError("Training must return checkpoint")
                outputs.append(result["checkpoint"])
            if any(Path(p).resolve() == directory.resolve() for p in outputs):
                raise ValueError(
                    "Artifacts must name stage files or subdirectories, not the stage root"
                )
            state.update(
                status="complete",
                outputs={p: identity(p) for p in outputs},
                dependencies={
                    p: identity(p) for p in result.get("dependency_paths", [])
                },
                finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            write(receipt, state)
        except BaseException as exc:
            state.update(
                status="failed", error=repr(exc), traceback=traceback.format_exc()
            )
            write(receipt, state)
            raise
    return dict(plan, status="complete")


def improve(
    config,
    round_number=None,
    rounds=None,
    start="generate",
    end="evaluate",
    dry_run=False,
):
    first, last = STAGES.index(start), STAGES.index(end)
    if first > last:
        raise ValueError("--from must precede --to")
    if round_number is not None and rounds is not None:
        raise ValueError("Use either --round or --rounds")
    if rounds is not None and rounds < 1:
        raise ValueError("--rounds must be positive")
    numbers = (
        [round_number]
        if round_number is not None
        else range(1, (rounds or config.get("rounds", 1)) + 1)
    )
    for r in numbers:
        for stage in STAGES[first : last + 1]:
            yield run_stage(config, stage, r, dry_run)
