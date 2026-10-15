# EVO-WAM: A RoboTwin Reproduction

A work-in-progress reproduction of **EVO-WAM** on RoboTwin, exploring how
video and action verification can help improve a world action model.

[Paper](https://arxiv.org/abs/2609.38057) ·
[Original project](https://evo-wam.github.io/) ·
[Official implementation](https://github.com/Clausy9/EVO-WAM)

This is an unofficial implementation. The goal is to make the RoboTwin
pipeline easier to study, run and extend, with explicit configurations for
generation, verification, training and evaluation.

**Model training is in progress. We expect to release the model weights in November 2026.**

## Contents

- [Overview](#overview)
- [Reproduction plan](#reproduction-plan)
- [Installation](#installation)
- [Preparing models and data](#preparing-models-and-data)
- [Running the pipeline](#running-the-pipeline)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

EVO-WAM uses imagined trajectories to improve a robot policy. A world action
model generates candidate videos and actions; an inverse dynamics model checks
whether the actions agree with the imagined motion, while a vision-language
model checks task completion. Accepted trajectory prefixes become additional
training examples for the next round.

This repository focuses on the RoboTwin setting and implements the following loop:

```text
Generate trajectories → Verify actions and task completion
         ↑                            ↓
Evaluate the policy ← Fine-tune ← Export accepted prefixes
```

The default recipe uses a fixed action-consistency threshold across rounds.
Training recipes and pipeline settings live in configuration files so that
experiments can be inspected and repeated without changing the orchestration code.

The repository is organized as follows:

| Directory | Purpose |
| --- | --- |
| `src/evo_wam/` | Pipeline stages, model adapters and verification |
| `configs/` | Generation, training, evaluation and VLM settings |
| `recipes/` | Per-round training recipes and asset-binding examples |
| `scripts/` | Setup utilities and diagnostic checks |
| `patches/` | Compatibility patches for the native frameworks |
| `tests/` | CPU tests for pipeline behavior and data contracts |
| `docs/` | Setup notes, experiment details and known limitations |

## Reproduction plan

The implementation is organized around three parts:

1. **Generate and verify.** Connect the world action model, inverse dynamics
   model and VLM service, then export the accepted trajectory prefixes.
2. **Improve the policy.** Add accepted prefixes to the cumulative training
   pool and fine-tune the model using the round-specific recipe.
3. **Evaluate in RoboTwin.** Run the resulting checkpoint on registered tasks
   and seeds, retaining successes and failures for analysis.

The first target is a usable RoboTwin pipeline with documented setup and small
integration checks. Reproducing the full training schedule and the paper's
benchmark results requires a separate, larger experiment.

## Installation

Use Python 3.10 or newer for the orchestration package. Python 3.11 is the
reference version for the GPU environment recipes.

```bash
git clone https://github.com/XiaohanLei/evo-wam.git
cd evo-wam

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps -e .
```

Check the installation and configuration:

```bash
evo-wam --help
evo-wam doctor --config configs/cosmos3-robotwin-improve.yaml
python -m unittest discover -s tests
```

These checks run without model weights or a simulator. The doctor validates the
configuration and reports missing environment variables; it does not load models
or validate the GPU runtime.

## Preparing models and data

GPU execution needs the native WAM and IDM frameworks, compatible checkpoints,
RoboTwin data and simulator assets, and a VLM API endpoint. These assets are
installed separately. Exact native source snapshots and selected checkpoint
download links are not yet provided in this repository.

Follow the [native asset setup](docs/native-assets.md) for the expected layout
and component environments. The requirement lock files describe separate
environments for WAM, IDM and RoboTwin; they should not be installed together
into the orchestration environment.

Create a local asset-binding file:

```bash
mkdir -p .private
cp recipes/asset-bindings.example.json .private/asset-bindings.json
```

Fill in the paths to your assets, then use `environment.example` as a starting
point for the runtime variables. The doctor lists the variables required by each
configuration. Keep local bindings and API keys outside version control, and
choose a new `RUN_OUTPUT` directory for each experiment. Set `IDM_PACKAGE` to
the package name in your native IDM installation; see the native interface
bindings in the setup guide.

For verification, both a locally served model and a hosted API can be used
through the VLM endpoint interface. See [VLM setup](docs/vlm-api.md) for the
request format and configuration examples.

## Running the pipeline

Start by inspecting the stage plan:

```bash
evo-wam improve \
  --config configs/cosmos3-robotwin-improve.yaml \
  --rounds 4 --dry-run
```

The dry run requires the local asset-binding file and plans the stages without
executing models. Cluster allocation and multi-node worker launch are managed
outside this CLI.

For a smaller integration run, generate trajectories, verify them and stop
after prefix export:

```bash
evo-wam improve \
  --config configs/cosmos3-robotwin-prefix-check.yaml \
  --rounds 1 --to export-prefix
```

Check the generation jobs, GPU indices and runtime bindings in that configuration
before running it. Once the required assets and launchers are configured, the
four-round workflow uses the same `improve` command without `--dry-run`.

Each stage writes a receipt that records its inputs and outputs. Resume with
the same configuration and inputs so completed stages can be reused safely.

IDM training and calibration have separate entry points:

```bash
evo-wam idm-train --config configs/idm-train.yaml
evo-wam idm-calibrate --config configs/idm-calibrate.yaml
```

## Evaluation

To evaluate an existing checkpoint in RoboTwin:

```bash
evo-wam evaluate \
  --config configs/cosmos3-robotwin-evaluate.yaml \
  --round 1
```

Set the checkpoint path, task/seed registry, simulator paths and GPU indices
before launching. Keep the complete outcome set, including failed episodes,
when comparing runs. See the [simulator notes](docs/simulator-validation.md)
for the evaluation protocol and observed setup issues.

## Acknowledgements

This project builds on the EVO-WAM paper and the work of the Cosmos,
DreamZero, RoboTwin, Qwen and Wan communities. Credit for the original method
belongs to the EVO-WAM authors. Please cite the original paper when using or
discussing the method; the authors' [BibTeX entry](https://github.com/Clausy9/EVO-WAM/blob/main/citation.bib)
is available in the official repository.

## License

Maintainer contributions are provided under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Third-party code, model weights
and datasets retain their respective terms. See [NOTICE](NOTICE.md) and the
[license inventory](docs/license-inventory.md) for attribution and outstanding
redistribution requirements.
