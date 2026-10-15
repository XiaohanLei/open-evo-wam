"""Single-host native RoboTwin evaluation against a supplied Cosmos policy service."""

import json, os, shutil, subprocess, sys, tempfile, time, urllib.request, signal
from pathlib import Path
from evo_wam.io import read, write, sha
from evo_wam.stages import json_rows, complete
from evo_wam.evaluation import summarize


def stage_runtime(source, target, assets, policy_source, episode_runner=None):
    """Copy executable/configuration sources; keep large simulator assets external."""
    source = Path(source)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    allowed = {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".sh",
        ".urdf",
        ".srdf",
        ".toml",
        ".ini",
        ".cpp",
        ".h",
        ".hpp",
        ".cuh",
        ".cu",
    }
    records = []
    for folder in ("envs", "script", "task_config", "description"):
        for current, dirs, files in os.walk(source / folder, followlinks=False):
            dirs[:] = [
                d for d in dirs if d not in ("__pycache__", ".git", "cache", "build")
            ]
            for name in files:
                src = Path(current) / name
                if src.suffix not in allowed:
                    continue
                dst = target / src.relative_to(source)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                records.append(
                    dict(
                        path=str(dst.relative_to(target)),
                        source=str(src),
                        sha256=sha(dst),
                    )
                )
    policy = target / "policy/WAMDenseGlobalSinkHTTP"
    policy.mkdir(parents=True, exist_ok=True)
    for src in (Path(policy_source),):
        dst = policy / ("deploy_policy.py" if src.suffix == ".py" else src.name)
        shutil.copy2(src, dst)
        records.append(
            dict(path=str(dst.relative_to(target)), source=str(src), sha256=sha(dst))
        )
    for name, text in [("deploy_policy.yml", "policy_name: WAMDenseGlobalSinkHTTP\nseed: 0\n"),
                       ("__init__.py", "from .deploy_policy import get_model, eval, reset_model\n")]:
        dst = policy / name
        dst.write_text(text)
        records.append(dict(path=str(dst.relative_to(target)), generated=True, sha256=sha(dst)))
    (target / "policy/__init__.py").write_text("")
    if episode_runner:
        original = target / "script/eval_policy.py"
        upstream = original.with_name("_upstream_eval_policy.py")
        shutil.copy2(original, upstream)
        shutil.copy2(episode_runner, original)
        records = [r for r in records if r["path"] != "script/eval_policy.py"]
        records.extend([dict(path="script/_upstream_eval_policy.py", source=str(source / "script/eval_policy.py"), sha256=sha(upstream)),
                        dict(path="script/eval_policy.py", source=str(episode_runner), sha256=sha(original))])
    link = target / "assets"
    if link.exists() or link.is_symlink():
        if link.resolve() != Path(assets).resolve():
            raise ValueError("Simulator asset binding changed")
    else:
        link.symlink_to(Path(assets).resolve(strict=True), target_is_directory=True)
    return records


def runtime_dependencies(records, assets):
    """Generated runtime files have hashes, but no external source dependency."""
    return sorted({str(assets), *[r['source'] for r in records if 'source' in r]})


def execution_mode(spec):
    mode = spec.get('dense_mode','closed_loop')
    if mode not in ('closed_loop','self_imagine'):
        raise ValueError('Use an explicit closed-loop policy or self-imagined action replay')
    if type(spec.get('save_lossless_rgb',False)) is not bool:
        raise ValueError('Lossless recording must be an explicit boolean')
    return mode


def run(context):
    spec = context["spec"]
    out = Path(context["output"])
    checkpoint = context["current_checkpoint"]
    mode = execution_mode(spec)
    sys.path.insert(0, spec["source_root"])
    from cosmos_framework.evaluation.robotwin_dense_evidence import (
        verify_identity,
        audit_artifacts,
    )

    source = Path(spec["source_root"])
    robotwin = out / "RoboTwin"
    code = stage_runtime(
        spec["robotwin_source"],
        robotwin,
        spec["assets_root"],
        source / "cosmos_framework/inference/robotwin_dense_http_client.py",
        spec.get("episode_runner"),
    )
    write(out / "runtime-source-manifest.json", code)
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    for key, name in [
        ("statistics", "state_action_stats.json"),
        ("train_config", "train_config.yaml"),
    ]:
        shutil.copy2(spec[key], inputs / name)
    frozen = dict(
        checkpoint=checkpoint,
        checkpoint_metadata_sha256=sha(Path(checkpoint) / "model/.metadata"),
        input_sha256={p.name: sha(p) for p in inputs.iterdir() if p.is_file()},
    )
    write(out / "evaluation_identity.json", frozen)
    url = spec["server_url"].rstrip("/")
    health = json.load(urllib.request.urlopen(url + "/health", timeout=10))
    verify_identity(health["identity"], out)
    if Path(health["identity"]["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Server loaded a different checkpoint path")
    from urllib.parse import urlparse

    address = urlparse(url)
    if address.scheme != "http" or not address.hostname or not address.port:
        raise ValueError("Explicit native HTTP server host/port required")
    registry = json_rows(spec["registry"])
    results = []
    if len({r["episode_key"] for r in registry}) != len(registry):
        raise ValueError("Duplicate frozen episode")
    env = dict(os.environ, **{k: str(v) for k, v in spec.get("native_env", {}).items()})
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(source), str(robotwin), *spec.get("python_paths", [])]
        ),
        DENSE_EVAL_IDENTITY=str(out / "evaluation_identity.json"),
        DENSE_SAVE_LOSSLESS_RGB='1' if spec.get('save_lossless_rgb',False) else '0',
        DREAMZERO_ROBOTWIN_NORMALIZATION_STATISTICS_PATH_OVERRIDE=str(
            inputs / "state_action_stats.json"
        ),
        DREAMZERO_ROBOTWIN_NORMALIZATION_STATISTICS_SHA256=frozen["input_sha256"][
            "state_action_stats.json"
        ],
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        ROBOTWIN_SAPIEN_CPU_THREADS="2",
        ROBOTWIN_POLICY_ACTION_RENDER_MODE="all",
        ROBOTWIN_QUIET_STEPS="1",
    )
    if len(env.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1 or not env.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise ValueError("Specify one rendering GPU in native_env")
    artifacts = [
        robotwin,
        out / "runtime-source-manifest.json",
        out / "evaluation_identity.json",
        inputs,
    ]
    with tempfile.TemporaryDirectory(prefix="evo-wam-eval-render-", dir=out) as runtime:
        env["XDG_RUNTIME_DIR"] = runtime
        os.chmod(runtime, 0o700)
        for i, episode in enumerate(registry):
            directory = out / "episodes" / f"{i:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            write(directory / "episode_spec.json", episode)
            limit = int(episode["max_steps"])
            cap = spec.get("smoke_max_steps")
            if cap is not None:
                limit = min(limit, int(cap))
            if limit < 1:
                raise ValueError("Invalid episode step limit")
            env["DREAMZERO_ROBOTWIN_INFERENCE_NOISE_SEED"] = str(
                episode["inference_noise_seed"]
            )
            tag = f"evo_wam_{i:06d}"
            options = dict(
                task_name=episode["task_name"],
                task_config=episode["task_config"],
                ckpt_setting=tag,
                policy_name="WAMDenseGlobalSinkHTTP",
                instruction_type=episode["instruction_type"],
                instruction_override=episode["instruction"],
                fixed_seed=episode["fixed_seed"],
                seed=episode["seed"],
                test_num=1,
                host=address.hostname,
                port=address.port,
                timeout_s=spec.get("request_timeout", 600),
                imagined_dump_dir=str(directory / "client_rollout"),
                episode_key=episode["episode_key"],
                dense_mode=mode,
                physics_failure_retry="True",
                max_physics_attempts=1,
                eval_step_limit_override=limit,
            )
            command = [
                spec["robotwin_python"],
                "script/eval_policy.py",
                "--config",
                "policy/WAMDenseGlobalSinkHTTP/deploy_policy.yml",
                "--policy_ckpt_path",
                checkpoint,
                "--overrides",
            ]
            for key, value in options.items():
                command += ["--" + key, str(value)]
            write(directory / "command.json", command)
            result = dict(
                episode,
                id=episode["episode_key"],
                status="infrastructure_error",
                checkpoint_identity=frozen,
                requested_step_limit=limit,
                smoke=cap is not None,
            )
            try:
                with (directory / "client.log").open("w") as log:
                    client = subprocess.Popen(
                        command,
                        cwd=robotwin,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        rc = client.wait(timeout=spec.get("episode_timeout", 900))
                        if rc:
                            raise subprocess.CalledProcessError(rc, command)
                    finally:
                        if client.poll() is None:
                            os.killpg(client.pid, signal.SIGTERM)
                            try:
                                client.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(client.pid, signal.SIGKILL)
                                client.wait()
                path = (
                    robotwin
                    / "eval_result"
                    / episode["task_name"]
                    / "WAMDenseGlobalSinkHTTP"
                    / episode["task_config"]
                    / tag
                )
                ledgers = list(path.glob("*/physics_retry_ledger.json"))
                if len(ledgers) != 1:
                    raise ValueError("Expected exactly one simulator result ledger")
                ledger = read(ledgers[0])
                if (
                    ledger["seed"] != episode["fixed_seed"]
                    or ledger["instruction"] != episode["instruction"]
                    or len(ledger["attempts"]) != 1
                ):
                    raise ValueError("Frozen episode or retry invariant violated")
                write(directory / "physics_ledger.json", ledger)
                if ledger.get("final_counted") is False:
                    result['status'] = ledger['attempts'][0]['status']
                    if result['status'] not in ('physics_invalid','expert_check_fail'):
                        raise ValueError('Unknown preflight failure class')
                    result['failure_reason'] = ledger['attempts'][0].get('reason')
                else:
                    audited = audit_artifacts(
                        directory / "client_rollout", mode
                    )
                    verify_identity(audited["generation"]["identity"], out)
                    actual = audited["execution"]["termination"]["official_success"]
                    if type(actual) is not bool or actual != ledger["final_success"]:
                        raise ValueError("Simulator success accounting differs")
                    for k, chunk in enumerate(audited["generation"]["chunks"]):
                        if (
                            chunk["inference_noise_seed"]
                            != episode["inference_noise_seed"] + k
                        ):
                            raise ValueError("Inference noise seed changed")
                    result.update(
                        status="success" if actual else "model_task_failure",
                        steps_executed=audited["execution"]["steps_executed"],
                    )
            except subprocess.TimeoutExpired as exc:
                result.update(status="timeout", error=str(exc))
            except Exception as exc:
                result["error"] = repr(exc)
            write(directory / "result.json", result)
            results.append(result)
            artifacts.append(directory)
    expected = [dict(r, id=r["episode_key"]) for r in registry]
    summary = summarize(expected, results)
    if spec.get("smoke_max_steps") is not None:
        summary.update(
            success_rate=None,
            conservative_success=None,
            valid_success=None,
            task_success_rate_evaluated=False,
        )
        for group in summary["by_task_config"].values():
            group.update(conservative_success=None, valid_success=None)
    target = out / "episode-results.json"
    write(target, dict(episodes=results))
    artifacts.append(target)
    report = out / "evaluation.json"
    write(
        report,
        dict(summary, smoke=spec.get("smoke_max_steps") is not None, formal=False),
    )
    artifacts.append(report)
    complete(
        context,
        artifacts,
        summary=summary,
        episode_results=str(target),
        smoke=spec.get("smoke_max_steps") is not None,
        dependency_paths=runtime_dependencies(code, spec["assets_root"]),
    )
    if any(
        r["status"] in ("infrastructure_error", "timeout", "physics_invalid", "expert_check_fail")
        for r in results
    ):
        raise RuntimeError(
            "Evaluation retained infrastructure/physics failures; inspect episode results"
        )
