"""Validate the pipeline profile before importing native GPU implementations."""
import json
import os
from pathlib import Path
import re
import yaml
from .pipeline import load_config

THRESHOLD = 0.004184869576192262
LATE = {"ROUND", "RUN_DIR", "STAGE_DIR", "PARENT_CHECKPOINT", "CURRENT_CHECKPOINT", "PREVIOUS_STAGE_DIR"}
ADAPTERS = {
    "generate": "evo_wam.adapters.cosmos:generate_robotwin",
    "verify": "evo_wam.verification.robotwin:verify",
    "export-prefix": "evo_wam.data.robotwin_dense_online:export",
    "train": "evo_wam.adapters.cosmos_robotwin_train:train",
    "evaluate": "evo_wam.adapters.cosmos_robotwin_evaluate:run",
    "idm-train": "evo_wam.adapters.idm_training:train",
    "idm-calibrate": "evo_wam.adapters.idm_calibration:calibrate",
}

def validate_profile(config):
    if config.get("schema") != "evo-wam.pipeline.v1":
        raise ValueError("Unsupported pipeline schema")
    stages = config.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("At least one explicit stage is required")
    for name, spec in stages.items():
        if name not in ADAPTERS or spec.get("adapter") != ADAPTERS[name]:
            raise ValueError("Stage/adapter outside the supported RoboTwin profile: " + name)
        if spec.get("interface", "dense") != "dense":
            raise ValueError("Only dense is supported")
        if spec.get("recipe", "robotwin") != "robotwin":
            raise ValueError("Only the RoboTwin recipe is supported")
        if name in ("verify", "export-prefix") and spec.get("threshold") != THRESHOLD:
            raise ValueError("The action-consistency threshold must remain explicit")
        if name == "train" and set(map(str, spec.get("recipe_configs", {}))) != {"1", "2", "3", "4"}:
            raise ValueError("Explicit standalone R1-R4 recipes are required")
    return config

def doctor(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    validate_profile(config)
    names = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", path.read_text())) - LATE - {"PACKAGE_ROOT", "CONFIG_DIR"}
    supplied = set(os.environ) | set(config.get("variables", {}))
    missing = sorted(names - supplied)
    return {"profile": "robotwin+dense", "configuration_valid": True,
            "missing_environment_variables": missing, "all_variables_supplied": not missing,
            "assets_checked": False, "ready_for_native_run": False,
            "gpu_execution_verified": False}

def load_profile(path):
    config = validate_profile(load_config(path))
    for spec in config["stages"].values():
        if spec.get("recipe") == "robotwin" and spec.get("bindings_file"):
            bindings = json.loads(Path(spec["bindings_file"]).read_text())
            if not isinstance(bindings, dict) or not all(isinstance(k, str) and k.startswith("/assets/") and isinstance(v, str) and Path(v).is_absolute() and not v.startswith("/replace/") for k, v in bindings.items()):
                raise ValueError("Asset bindings must map /assets aliases to real absolute paths")
            required = set()
            for recipe in spec.get("recipe_configs", {}).values():
                data = yaml.safe_load(Path(recipe).read_text())
                def visit(value):
                    if isinstance(value, str) and value.startswith("/assets/"): required.add(value)
                    elif isinstance(value, list):
                        for child in value: visit(child)
                    elif isinstance(value, dict):
                        for child in value.values(): visit(child)
                visit(data)
            if required - bindings.keys():
                raise ValueError("Missing recipe asset aliases: " + ", ".join(sorted(required - bindings.keys())))
            spec["asset_bindings"] = bindings
    return config
