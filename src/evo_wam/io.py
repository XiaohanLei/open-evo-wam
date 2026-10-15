"""Portable manifests and deterministic stage identities."""

import gzip, hashlib, json, os, tempfile
from pathlib import Path


def read(path):
    with (
        gzip.open(path, "rt", encoding="utf-8")
        if str(path).endswith(".gz")
        else Path(path).open(encoding="utf-8")
    ) as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    """Full hashes for small files; explicit stat identity for large model/data files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    files = (
        sorted(
            p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
        if path.is_dir()
        else [path]
    )
    rows = []
    for p in files:
        s = p.stat()
        row = {
            "path": str(p.relative_to(path)) if path.is_dir() else p.name,
            "bytes": s.st_size,
        }
        if s.st_size <= 16 * 1024 * 1024:
            row["sha256"] = sha(p)
        else:
            row.update(
                mtime_ns=s.st_mtime_ns,
                inode=s.st_ino,
                verification="size_mtime_inode_only",
            )
        rows.append(row)
    return {
        "digest": digest(rows),
        "files": len(rows),
        "large_files_use_stat_identity": any("mtime_ns" in r for r in rows),
    }


def expand(value, variables):
    if isinstance(value, str):
        for key, val in variables.items():
            value = value.replace("${" + key + "}", str(val))
        if "${" in value:
            raise ValueError("Unresolved config variable: " + value)
        return value
    if isinstance(value, list):
        return [expand(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, variables) for k, v in value.items()}
    return value
