"""Load the selected WAM and generate two chunks from a synthetic observation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    os.environ.update(COSMOS_TRAINING='0', COSMOS_INTERNAL='0', COSMOS_DEVICE='cuda')
    sys.path.insert(0, str(root / 'native/cosmos_robotwin_inference'))
    report = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint='wam-default', synthetic_input=True,
                  architecture_load=False, generation=False, simulator_episode=False,
                  full_reproduction=False, num_steps=30, guidance=1.0, shift=5.0)
    output = root / '.private/wam-generation-smoke.json'
    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')
        output.chmod(0o600)
    save()
    start = time.monotonic()
    try:
        import numpy as np
        import yaml
        import torch
        from evo_wam.metadata_privacy import rewrite_pickle_strings
        from cosmos_framework.inference.robotwin_dense_globalsink import DenseGlobalSinkPolicy
        from cosmos_framework.inference.robotwin_dense_contract import encode_image
        online = root / 'weights/wam-default'
        # Normalize only the Python-version-dependent pathlib module name.
        metadata = online / '.metadata'
        original = metadata.read_bytes()
        normalized = rewrite_pickle_strings(original, {'pathlib._local': 'pathlib'})
        if normalized != original:
            backup = root / '.private/wam-metadata-before-python-compat.pkl'
            backup.write_bytes(original)
            backup.chmod(0o600)
            metadata.write_bytes(normalized)
        report['metadata_python_compatibility'] = dict(
            changed=normalized != original,
            before_sha256=hashlib.sha256(original).hexdigest(),
            after_sha256=hashlib.sha256(normalized).hexdigest(), tensor_mapping_changed=False)
        parent = root / 'weights/wam-base'
        parent.mkdir(mode=0o700, exist_ok=True)
        model = parent / 'model'
        if not model.exists():
            model.symlink_to('../wam-default', target_is_directory=True)
        if model.resolve() != online:
            raise ValueError('WAM checkpoint alias points to another model')
        config = yaml.safe_load((online / 'config.yaml').read_text())
        mc = config['model']['config']
        mc['tokenizer']['vae_path'] = str(root / 'assets/vae/wan2.2/Wan2.2_VAE.pth')
        mc['vlm_config']['tokenizer']['pretrained_model_name'] = str(root / 'assets/tokenizer/qwen3-vl')
        mc['vlm_config']['pretrained_weights']['backbone_path'] = ''
        runtime = root / '.private/wam-runtime-config.yaml'
        runtime.write_text(yaml.safe_dump(config))
        runtime.chmod(0o600)
        artifacts = root / 'run-output/wam-generation-smoke'
        policy = DenseGlobalSinkPolicy(checkpoint=parent, train_config=runtime,
            stats_path=online / 'statistics.json', artifact_root=artifacts,
            num_steps=30, guidance=1.0, shift=5.0)
        report['architecture_load'] = True
        report['load_seconds'] = time.monotonic()-start
        save()
        state = policy.builder.mean.numpy()
        policy.generate_openloop(dict(artifact_dir=str(artifacts / 'synthetic-case'),
            global_image=encode_image(np.full((360,640,3),127,dtype=np.uint8)),
            state=state.tolist(), max_steps=128, instruction='place the empty cup',
            inference_noise_seed=42, episode_id='synthetic-smoke', export_video=True))
        actions = np.load(artifacts / 'synthetic-case/actions_qpos.npy', allow_pickle=False)
        if actions.shape != (128,14) or not np.isfinite(actions).all():
            raise ValueError('Generated action shape or finiteness check failed')
        report.update(status='passed', generation=True, output_action_shape=list(actions.shape),
                      elapsed_seconds=time.monotonic()-start,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        save()
        print(json.dumps(report))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__,
                      elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__ == '__main__':
    main()
