"""Resolve deployed IDM assets in memory while retaining frozen manifest bytes."""

import functools
import os
from pathlib import Path
from evo_wam.io import read, sha
from evo_wam.path_bindings import PathBindings


def configure(path):
    if path is None:
        os.environ.pop("EVO_IDM_PATH_BINDINGS", None)
        os.environ.pop("EVO_IDM_PATH_BINDINGS_SHA256", None)
        return None
    path = Path(path).resolve()
    digest = sha(path)
    PathBindings(read(path))
    os.environ["EVO_IDM_PATH_BINDINGS"] = str(path)
    os.environ["EVO_IDM_PATH_BINDINGS_SHA256"] = digest
    return digest


@functools.lru_cache(maxsize=8)
def _load(path, digest):
    if sha(path) != digest:
        raise ValueError("IDM path bindings changed before worker initialization")
    return PathBindings(read(path))


def bindings():
    path = os.environ.get("EVO_IDM_PATH_BINDINGS")
    return (
        _load(path, os.environ["EVO_IDM_PATH_BINDINGS_SHA256"])
        if path
        else PathBindings()
    )


def asset_path(path):
    return bindings().path(path)


def model_assets(value):
    """Model configuration and training recipe contain no user instruction text."""
    if isinstance(value, dict):
        return {k: model_assets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [model_assets(v) for v in value]
    return asset_path(value) if isinstance(value, str) else value


def data_rows(rows):
    fields = {
        "cache_path",
        "source_path",
        "video_path",
        "source_data_path",
        "source_task_root",
    }

    def visit(value, key=""):
        if isinstance(value, dict):
            return {k: visit(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, key) for v in value]
        return asset_path(value) if key in fields else value

    return visit(rows)
