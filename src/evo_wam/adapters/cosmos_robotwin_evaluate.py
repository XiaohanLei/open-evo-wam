"""Own a native policy process for one evaluation stage, then close it."""
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from .process_lifecycle import stop
from .robotwin_evaluation import run as evaluate, execution_mode
from evo_wam.io import write


def run(context):
    spec = context['spec']
    if spec.get('server_url'):
        return evaluate(context)
    out = Path(context['output']); url = f"http://127.0.0.1:{spec['port']}"
    env = dict(os.environ, **{k: str(v) for k,v in spec.get('model_env', {}).items()})
    for key in list(env):
        if key.startswith(('RAY_', 'PET_', 'TORCHELASTIC_')) or key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=str(spec.get('model_gpu',0)),
               PYTHONPATH=spec['source_root']+os.pathsep+env.get('PYTHONPATH',''))
    artifact_root = out/'episodes' if execution_mode(spec)=='self_imagine' else out/'policy-artifacts'
    argv = [spec['model_python'], '-m', 'cosmos_framework.scripts.serve_robotwin_dense_globalsink',
        '--checkpoint', context['current_checkpoint'], '--train-config', spec['train_config'],
        '--stats', spec['statistics'], '--artifact-root', str(artifact_root),
        '--host', '127.0.0.1', '--port', str(spec['port']),
        '--num-steps', str(spec.get('num_steps',30)), '--guidance', str(spec.get('guidance',1.0)),
        '--shift', str(spec.get('shift',5.0))]
    write(out/'policy-command.json', argv)
    with (out/'policy-server.log').open('a') as log:
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+spec.get('server_timeout',1800)
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Native policy exited; see policy-server.log')
                try:
                    with urllib.request.urlopen(url+'/health', timeout=5):
                        break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic()>deadline:
                        raise TimeoutError('Native policy startup timed out')
                    time.sleep(1)
            return evaluate(dict(context, spec=dict(spec, server_url=url)))
        finally:
            stop(process)
