# Bounded training/save/resume check

`scripts/wam_checkpoint_training_smoke.py` exercises the complete selected WAM
with the native approved-prefix reader, loss, backward pass and FusedAdam update.
It is intended to check one update/save cycle and a fresh-process resume followed
by a second update/save cycle. It does not execute the formal R1–R4 training.

The reference setup uses two CUDA GPUs, Python 3.11, Torch 2.8.0 + CUDA 12.8 and
Transformer Engine 2.12.0, with activation checkpointing, CPU offloading and
enough host memory/disk for full model and optimizer states. The effective global
batch is 4; the formal recipe remains batch 256 on 128 GPUs.

Prepare compatible native sources, the complete parent checkpoint, its model
configuration, and an accepted exported-prefix dataset. Use a new output path:

```bash
"$TRAINING_PYTHON" -m pip install --no-deps -e .

"$TRAINING_PYTHON" -m torch.distributed.run \
  --nproc-per-node=2 --master-port=18910 \
  scripts/wam_checkpoint_training_smoke.py \
  --source-root "$NATIVE_ROOT/cosmos_robotwin_inference" \
  --checkpoint "$PARENT_CHECKPOINT_PATH" --config "$MODEL_TRAIN_CONFIG" \
  --recipe recipes/robotwin/r1.yaml --dataset "$APPROVED_DATASET" \
  --output "$TRAINING_SMOKE_OUTPUT" --phase initial
```

Wait for the process to exit successfully and inspect both `initial-rank0.json`
and `initial-rank1.json`. Then run the same command with `--phase resume`. Inspect
both resume receipts and require a successful process exit.

The resume check compares per-rank model hashes and every FP32 master/first/second
moment buffer, checks the saved step, and restores RNG/sampler state before
another update. Use the FusedAdam compatibility patch when the native source
does not already preserve
these states. A newly created complete checkpoint cannot establish exact
recovery of an older checkpoint that omitted FP32 master weights.

Diagnose NCCL communication separately if a multi-GPU run stalls. One prior
environment required `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`; those settings are
machine-specific and should not be applied to formal multi-node training without
checking the target interconnect. Inspect live processes and receipts before
starting a retry into an existing output directory.

Prior bounded checks completed two updates, full saves and an exact fresh reload.
This supports that limited path only. Training data must consist of accepted
imagined prefixes; keep held-out calibration trajectories separate and preserve
the verification/export checks rather than bypassing a rejected prefix.
