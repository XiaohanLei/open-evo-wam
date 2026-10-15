# Validation status and reproducibility limits

This repository is a third-party reproduction. Its available checks do not
establish the original paper's reported results. Prior GPU diagnostics used
separately supplied native sources, checkpoints and datasets; those assets and
complete run artifacts are not part of the source tree. The summary below
describes recorded limited checks, rather than a new independent GPU rerun.

| Area | Recorded scope | Limit |
| --- | --- | --- |
| CPU contracts | Configuration, recipe invariants, stage receipts, asset packaging and protocol behavior | No native model execution |
| Model loading and generation | Selected RoboTwin R4 WAM and selected IDM checkpoint; short generation and scoring runs | No full benchmark or four-round training |
| RoboTwin execution | Small registered task/seed sets using upstream task conditions | Too small to estimate paper-level success; expert preflight instability was observed |
| IDM diagnostic | Three lossless trajectories scored through their full action tails; two successful tasks and one failed task | The fixed threshold rejected one success; selected-checkpoint calibration equivalence remains unverified |
| Prefix export | One accepted imagined prefix, lossless reload and native training-reader/mask checks | No certification of the complete training population |
| VLM API | A local small Qwen service, strict response parsing and endpoint votes | External-provider execution and verifier accuracy remain unverified; deterministic votes are correlated |
| Training/save/resume | Two real updates of the selected full model, with a complete save and fresh model/FP32 optimizer reload | Effective batch 4; does not reproduce the formal batch-256, 128-GPU, four-round recipe |

The original action-consistency threshold is `0.004184869576192262`. The three-trajectory
diagnostic measured a success-only P90 of `0.004260649041127885` from just two
successes. That sample does not justify changing the threshold or claiming
equivalence to the earlier checkpoint used for calibration.

Run the CPU commands in the root README to check your own checkout. Record the
Python/dependency versions and distinguish these results from any native GPU
experiment. For new GPU checks, retain the exact source revisions, checkpoint
hashes, data registry, task/seed list, complete outcomes and failed runs. Inspect
generated artifacts for local bindings or sensitive content before sharing them.

## Remaining release requirements

The maintainer's contributions use the
[PolyForm Noncommercial License 1.0.0](../LICENSE). Confirmation that historical
implementation and patch sources may be redistributed remains outstanding.
Native source snapshots, selected weights and complete training data also lack
a public acquisition path in this repository. These are separate requirements
from passing functional tests. See [license inventory](license-inventory.md) and
[native assets](native-assets.md).

Publishing a limited reproduction should describe these missing assets and
unverified results explicitly. It should not use the original authors' benchmark
numbers as results of this implementation.
