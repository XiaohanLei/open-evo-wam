# Native frameworks and assets

The orchestration package expects two separately installed native framework
trees under `NATIVE_ROOT`:

```text
$NATIVE_ROOT/
├── cosmos_robotwin_inference/   # WAM generation, export, training and evaluation
└── idm_dense/                  # dense IDM loading and scoring
```

Compatible source snapshots, their manifests and selected checkpoint download
URLs are not distributed here. Installing an arbitrary current upstream checkout
does not establish compatibility. The adapters depend on specific native
modules, action/state formats and training readers. Before GPU execution, obtain
the matching sources with their revision information, licenses and notices.

## Install an inventoried source archive

If you have a compatible archive and an independently verified SHA256:

```bash
python scripts/install_native_asset.py \
  --bundle "$WAM_NATIVE_BUNDLE" --sha256 "$WAM_NATIVE_SHA256" \
  --manifest WAM_NATIVE_ASSET_MANIFEST.json \
  --destination "$NATIVE_ROOT/cosmos_robotwin_inference"

python scripts/install_native_asset.py \
  --bundle "$IDM_NATIVE_BUNDLE" --sha256 "$IDM_NATIVE_SHA256" \
  --manifest NATIVE_ASSET_MANIFEST.json \
  --destination "$NATIVE_ROOT/idm_dense"
```

`--url` accepts an HTTPS archive URL in place of `--bundle`. The installer checks
the archive hash and per-file inventory, rejects unsafe archive paths and links,
and requires a new destination. Archive integrity does not establish source
licensing or runtime compatibility.

## Component environments

| Component | Reference setup | Requirements |
| --- | --- | --- |
| CPU orchestration | Python 3.10+ | `requirements-cpu.txt` |
| WAM inference | Python 3.11, Torch 2.8.0 + CUDA 12.8 | `requirements-wam-inference.lock.txt` |
| IDM inference | Python 3.11 | `requirements-idm-inference.txt`, `requirements-idm-inference.lock.txt` |
| WAM training | Python 3.11, Torch 2.8.0 + CUDA 12.8, Transformer Engine 2.12.0 | `requirements-wam-training.lock.txt` |
| RoboTwin | Python 3.10, Torch 2.4.1 + CUDA 12.1 | `requirements-robotwin.txt`, `requirements-robotwin.lock.txt` |

These are separate environments. CUDA wheels, driver/toolkit compatibility and
native extensions must match the target machine. The optional `data` extra in
`pyproject.toml` is not a complete native export environment.

For the WAM training environment, inspect the installer and select your device's
compute capability explicitly:

```bash
python scripts/install_wam_training_env.py \
  --venv "$TRAINING_ENV" --python "$PYTHON311" --uv "$UV_EXECUTABLE" \
  --cuda-home "$CUDA_HOME" --cuda-arch "$CUDA_ARCH"
```

Use a new environment directory. This helper installs the locked CUDA packages
and builds Transformer Engine with the available cuDNN/NCCL headers. A prior
environment's successful import does not certify a fresh build on another host.

## Compatibility patches

The `patches/` directory contains changes for native framework checkouts:

| Patch | Purpose |
| --- | --- |
| `idm-full-export-loading.patch` | Load a complete IDM export without unnecessary bootstrap components |
| `wam-python-typing-compat.patch` | Support the selected WAM imports under Python 3.11 |
| `wam-sdpa-fallback.patch` | Add a packed PyTorch SDPA attention fallback |
| `wam-fused-adam-master-resume.patch` | Preserve FP32 optimizer master and moment buffers across resume |
| `robotwin-public-action-api.patch` | Match the public RoboTwin action API |
| `robotwin-lossless-initial-observation.patch` | Preserve the initial observation in lossless form |

Inspect patch applicability against the exact source revision. A matching file
path alone does not prove a patch is compatible, and an already patched source
archive should not be patched twice. Preserve upstream notices and record the
source revision and applied changes for your experiment.

## Bind checkpoints and data

Copy `recipes/asset-bindings.example.json` to an ignored local file and replace
all `/replace/` values with real absolute paths. `recipes/asset-contract.json`
describes the aliases for the VAE, base data, auxiliary data, adaptation pools
and parent checkpoints. Set `MODEL_ASSET_BINDINGS` to this local file.

Full R1–R4 training requires prepared base/auxiliary datasets and per-round
generation registries; these are not supplied by a source clone. Prepare the
loader manifest from `recipes/native-loader-template.example.json` and consult
`configs/prefix-jobs.example.jsonl` for a generation job's fields. Example
observation paths do not identify included fixture assets.

Use the same WAM statistics for generation, export and WAM training. IDM scoring
uses the statistics supplied with its own checkpoint; preserve that checkpoint's
statistics and calibration identity rather than replacing them with WAM
statistics. Accepted prefix exports preserve lossless observations, action/state
rows, scene identity and supervision masks. Calibration trajectories must remain
separate from training data. Checkpoint steps and training provenance should
describe the actual assets used.

## Native interface bindings

The pipeline uses `interface: dense` for IDM scoring and training. Set
`IDM_PACKAGE` to the top-level Python package actually supplied by the native
IDM source installation (default: `idm`). The source directory must contain
that package. Use the same value for training, inference and diagnostics.
Changing this variable selects an existing package; it does not rewrite imports
or class targets inside external source files or model configurations.

Diagnostic scripts use `weights/idm-default` and `weights/wam-default` as local
asset aliases; the WAM save/resume check also uses `weights/wam-base`. Bind or
symlink these paths to the intended assets. An alias is not a claim about the
checkpoint's training step; retain the actual training metadata and integrity
receipts with the model assets.

If the installed WAM framework expects different launcher environment variable
names, set `stages.train.native_environment_keys` in your local pipeline config.
The mapping translates the pipeline's variable names to the native framework's
actual names, preserving values. For example:

```yaml
native_environment_keys:
  ROBOTWIN_DENSE_BASE_ROOT: NATIVE_BASE_ROOT
  ROBOTWIN_DENSE_BOOTSTRAP_SLOTS: NATIVE_BOOTSTRAP_SLOTS
  ROBOTWIN_DENSE_AUX_MANIFEST: NATIVE_AUX_MANIFEST
```

The right-hand names above are illustrative. Match them to the installed
framework before GPU training. These bindings are required for compatibility
when using a framework with a different environment API; CPU tests alone do not
validate that external installation.
