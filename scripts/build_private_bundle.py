"""Bundle committed code only; never include ignored assets or Git history."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--privacy-rules", type=Path, help="Private JSON regex policy; never included in the bundle")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=root)
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("Commit tracked changes before deployment")
    commit = git("rev-parse", "HEAD").decode().strip()
    files = git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
    payload = {name: git("show", f"{commit}:{name}") for name in files if name}
    forbidden = {".private", ".native", ".venv", "weights", "datasets"}
    for name in payload:
        path = Path(name)
        if forbidden.intersection(path.parts) or path.suffix in {".safetensors", ".distcp", ".pt", ".pth"} or path.name.startswith("client_secret") or path.name in {"token.json", "credentials.json"}:
            raise SystemExit("Private asset must not enter deployment: " + name)
    if args.privacy_rules:
        rules = json.loads(args.privacy_rules.read_text())
        for rule in rules:
            pattern = re.compile(rule["pattern"], re.I)
            for name, data in payload.items():
                if pattern.search(name) or pattern.search(data.decode("utf-8")):
                    raise SystemExit("Privacy policy failed: " + rule["category"] + " in " + name)
    created = datetime.now(timezone.utc)
    manifest = {"commit": commit, "created_utc": created.isoformat(),
                "private": True, "native_assets_included": False,
                "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in payload.items()}}
    payload["DEPLOYMENT.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        args.output.chmod(0o600)
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, data in payload.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o600
                info.mtime = int(created.timestamp())
                archive.addfile(info, io.BytesIO(data))
    print(json.dumps({"bundle": str(args.output.resolve()), "commit": commit,
                      "files": len(manifest["files"]), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
