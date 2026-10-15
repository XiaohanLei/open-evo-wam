"""Bind the separately installed IDM framework without assuming its package name."""

import importlib
import os
from pathlib import Path
import re
import sys


def package_name():
    name = os.environ.get("IDM_PACKAGE", "idm")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("IDM_PACKAGE must be a top-level Python package name")
    return name


def native_module(component, source_root=None):
    name = package_name()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", component):
        raise ValueError("Invalid native IDM component")
    expected = None
    if source_root is not None:
        root = Path(source_root).resolve()
        expected = root / name
        if not expected.is_dir():
            raise FileNotFoundError("Native IDM package not found under source_root")
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if origin is None or Path(origin).resolve().parent != expected:
                raise RuntimeError("Different IDM source already imported; use a fresh stage process")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    module = importlib.import_module(name + "." + component)
    if expected is not None:
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(expected):
            raise RuntimeError("Native IDM component resolves outside source_root")
    return module


def remap_environment(environment, aliases):
    """Translate launcher variable names to a native framework's configured API."""
    if not isinstance(aliases, dict) or set(aliases) - environment.keys():
        raise ValueError("Native environment mapping contains unknown source keys")
    result = {}
    for name, value in environment.items():
        target = aliases.get(name, name)
        if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
            raise ValueError("Invalid native environment variable name")
        if target in result:
            raise ValueError("Native environment mapping has duplicate targets")
        result[target] = value
    return result
