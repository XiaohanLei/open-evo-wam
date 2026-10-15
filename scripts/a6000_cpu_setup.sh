#!/usr/bin/env bash
# Run only inside the explicitly authorized private A6000 project directory.
set -euo pipefail
umask 077
project=$(realpath "${EVO_PRIVATE_ROOT:?Set the private workspace root in your local environment}")
code=$(cd "$(dirname "$0")/.." && pwd -P)
resolved=$(realpath "$project")
[[ "$resolved" == "$project" && "$code" == "$project"/code/* ]]
[[ -O "$project" && -w "$project" ]]
export UV_CACHE_DIR="$project/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$project/python"
export UV_HTTP_TIMEOUT=60
export UV_HTTP_RETRIES=3
uv=${UV_EXECUTABLE:-$(command -v uv || true)}
[[ -n "$uv" && -x "$uv" ]]
environment="$project/environments/cpu"
"$uv" python install 3.11
if [[ ! -x "$environment/bin/python" ]]; then
    "$uv" venv --python 3.11 "$environment"
fi
"$uv" pip install --python "$environment/bin/python" -r "$code/requirements-cpu.txt"
"$uv" pip install --python "$environment/bin/python" --no-deps -e "$code"
cd "$code"
"$environment/bin/python" scripts/validate_deployment.py
