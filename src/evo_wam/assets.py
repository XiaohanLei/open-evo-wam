"""Inventory checks are read-only. Large files are hashed only with --verify."""

from pathlib import Path
from .io import read, sha


def inspect_manifest(path, verify=False):
    obj = read(path)
    items = obj.get("assets", []) if isinstance(obj, dict) else obj
    result = dict(
        entries=len(items),
        existing=0,
        missing=0,
        changed=0,
        verified_hashes=0,
        issues=[],
    )
    for item in items:
        p = Path(item["path"])
        if not p.exists():
            result["missing"] += 1
            result["issues"].append(dict(path=str(p), status="missing"))
            continue
        result["existing"] += 1
        if (
            p.is_file()
            and item.get("bytes") is not None
            and p.stat().st_size != item["bytes"]
        ):
            result["changed"] += 1
            result["issues"].append(dict(path=str(p), status="size_changed"))
        if verify and p.is_file():
            actual = sha(p)
            result["verified_hashes"] += 1
            expected = item.get("verified_sha256") or item.get("recorded_sha256")
            if isinstance(expected, list):
                expected = expected[0] if len(set(expected)) == 1 else None
            if expected and actual != expected:
                result["changed"] += 1
                result["issues"].append(dict(path=str(p), status="hash_mismatch"))
    return result
