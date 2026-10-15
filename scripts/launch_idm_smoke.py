"""Detach the limited GPU check from an unreliable SSH control connection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    script = Path(__file__).with_name('idm_inference_smoke.py').resolve()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(arg.endswith(b'/scripts/idm_inference_smoke.py') for arg in argv):
            raise RuntimeError('An IDM smoke process is already running')
    private = root / '.private'
    private.mkdir(mode=0o700, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key, directory in [('HF_HOME','huggingface'), ('TRITON_CACHE_DIR','triton'),
                           ('TORCHINDUCTOR_CACHE_DIR','torchinductor')]:
        env[key] = str(root / '.cache' / directory)
    log = private / 'idm-inference-smoke.log'
    with log.open('wb') as output:
        log.chmod(0o600)
        child = subprocess.Popen([sys.executable, str(script), '--root', str(root)],
            cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
    pidfile = private / 'idm-inference-smoke.pid'
    pidfile.write_text(str(child.pid) + '\n')
    pidfile.chmod(0o600)
    print('Started limited IDM validation:', child.pid)


if __name__ == '__main__':
    main()
