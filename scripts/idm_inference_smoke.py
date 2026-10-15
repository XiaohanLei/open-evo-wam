"""Strict IDM reload and one synthetic window; no simulator or training claim."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    output = root / '.private/idm-inference-smoke.json'
    report = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint='idm-default', synthetic_input=True,
                  strict_load=False, inference=False, training=False,
                  simulator_episode=False, full_reproduction=False)
    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')
        output.chmod(0o600)
    start = time.monotonic()
    save()
    try:
        import numpy as np
        import torch
        from evo_wam.verification.robotwin import load_predictor
        checkpoint = root / 'weights/idm-default'
        receipt = json.loads((checkpoint / 'DOWNLOAD_COMPLETE.json').read_text())
        if receipt.get('status') != 'download_integrity_verified' or len(receipt['files']) != 8:
            raise ValueError('Incomplete IDM download receipt')
        exported = json.loads((checkpoint / 'EXPORT_VERIFIED.json').read_text())
        for name, sha256 in exported['files'].items():
            if receipt['files'][name]['sha256'] != sha256:
                raise ValueError('Download digest differs from original verified export')
        report['download_receipt_present'] = True
        report['source_export_digests_match'] = True
        report['torch_version'] = torch.__version__
        report['gpu'] = torch.cuda.get_device_name(0)
        predict, stats = load_predictor(dict(source_root=str(root / 'native/idm_dense'),
            tokenizer=str(root / 'assets/tokenizer/umt5-xxl'), idm_checkpoint=str(checkpoint)))
        report['strict_load'] = True
        report['strict_load_seconds'] = time.monotonic() - start
        save()
        state = np.asarray(stats['state']['mean'], dtype=np.float32)
        result = np.asarray(predict(video=np.full((2,288,192,3),127,dtype=np.uint8),
            initial_state=state, instruction='move the object',
            frame_indices=np.arange(2), fps=15, seed=42))
        if result.shape != (1,14) or not np.isfinite(result).all():
            raise ValueError('Synthetic output shape or finiteness check failed')
        report.update(status='passed', inference=True, output_shape=list(result.shape),
                      output_finite=True, seed=42, denoising_steps=4,
                      total_seconds=time.monotonic()-start,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        save()
        print(json.dumps(report))
    except Exception as error:
        # Runtime paths and stack traces remain outside delivery receipts.
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['total_seconds'] = time.monotonic()-start
        save()
        raise


if __name__ == '__main__':
    main()
