"""Cosmos native autoregressive rollout adapters, without server/queue startup."""

import sys
from pathlib import Path
from evo_wam.io import write
from evo_wam.stages import json_rows, complete


def generate_robotwin(context):
    import numpy as np

    spec = context["spec"]
    sys.path.insert(0, spec["source_root"])
    from cosmos_framework.inference.robotwin_dense_globalsink import (
        DenseGlobalSinkPolicy,
    )
    from cosmos_framework.inference.robotwin_dense_contract import encode_image
    from evo_wam.data.robotwin_canvas import observation_image

    root = Path(context["output"])
    policy = DenseGlobalSinkPolicy(
        checkpoint=Path(context["parent_checkpoint"]),
        train_config=Path(spec["train_config"]),
        stats_path=Path(spec["statistics"]),
        artifact_root=root,
        num_steps=spec["num_steps"],
        guidance=spec["guidance"],
        shift=spec["shift"],
    )
    rows = []
    artifacts = []
    for i, job in enumerate(json_rows(spec["jobs"])):
        with np.load(job["initial_observation"], allow_pickle=False) as initial:
            if spec.get("initial_format", "canvas") == "robotwin_registry":
                state = initial["q0"]
                image = observation_image(
                    {
                        "observation": {
                            key[:-4]: {"rgb": initial[key]}
                            for key in initial.files
                            if key.endswith("_rgb")
                        }
                    }
                )
            elif spec.get("initial_format", "canvas") == "canvas":
                image = initial["global_image"]
                state = initial["state"]
            else:
                raise ValueError("Choose canvas or robotwin_registry initial_format")
        if image.shape != (360, 640, 3) or image.dtype != np.uint8:
            raise ValueError("Cosmos RoboTwin requires its native 360x640 uint8 canvas")
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Cosmos RoboTwin requires 14-D starting state")
        case = root / f"case_{i:06d}"
        receipt = policy.generate_openloop(
            dict(
                artifact_dir=str(case),
                global_image=encode_image(image),
                state=state.tolist(),
                max_steps=job["max_steps"],
                instruction=job["instruction"],
                inference_noise_seed=job["seed"],
                episode_id=job["id"],
                export_video=True,
            )
        )
        rows.append(
            dict(
                job,
                case=str(case),
                receipt=receipt,
                interface="cosmos_robotwin_dense64",
            )
        )
        artifacts.append(case)
    manifest = root / "generated.json"
    write(manifest, dict(episodes=rows))
    complete(context, artifacts + [manifest], manifest=str(manifest))
