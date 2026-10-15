"""Sequential official simulator replay of action tapes; no policy service or Ray."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evo_wam.io import read, write, sha
from evo_wam.stages import complete, json_rows
from .robotwin_evaluation import stage_runtime
from .dreamzero_robotwin_evaluation import stop


def verify_identity(identity, root):
    """Replay checks the explicitly bound generation source, without loading its model."""
    import cosmos_framework.evaluation.robotwin_dense_evidence as evidence
    original = evidence.execution_code_identity
    def bound(path):
        result = original(path)
        result['cosmos_framework/inference/robotwin_dense_globalsink.py'] = sha(os.environ['EVO_ORACLE_GENERATION_CORE'])
        return result
    evidence.execution_code_identity = bound
    try:
        return evidence.verify_identity(identity, root)
    finally:
        evidence.execution_code_identity = original


def run(context):
    import numpy as np
    spec = context['spec']; out = Path(context['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(spec['source_root']).resolve(); sys.path.insert(0, str(source))
    from cosmos_framework.evaluation.robotwin_dense_evidence import audit_artifacts
    from cosmos_framework.inference.robotwin_dense_contract import array_sha256
    from evo_wam.data.robotwin_canvas import observation_image
    manifest = spec.get('generated_manifest') or read(Path(context['previous_stage'])/'result.json')['manifest']
    rows = json_rows(manifest)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Replay needs nonempty, unique generated candidates')
    policy = Path(__file__).resolve().parents[1]/'recipes/oracle/policy.py'
    core = Path(spec.get('generation_core_source',source/'cosmos_framework/inference/robotwin_dense_globalsink.py')).resolve()
    results = []; dependencies = [str(core), manifest, str(source), str(policy), spec['statistics'], spec['train_config']]
    assets = Path(spec['assets_root']).resolve(strict=True)
    write(out/'asset-binding.json',dict(path=str(assets),version=spec.get('assets_version'),
        verification='Caller-supplied version; simulator assets are not recursively hashed'))
    if spec.get('assets_manifest'): dependencies.append(spec['assets_manifest'])
    for row in rows:
        case = Path(row.get('case') or Path(row['directory'])/'client_rollout').resolve()
        if not (case/'COMPLETE').is_file():
            raise ValueError('Cannot replay uncommitted generation: '+str(case))
        generation = read(case/'generation_manifest.json'); identity = generation['identity']
        if sha(core) != identity['core_source_sha256']: raise ValueError('Bind generation_core_source to the source used for this tape')
        if sha(spec['statistics']) != identity['stats_sha256'] or sha(spec['train_config']) != identity['train_config_sha256']:
            raise ValueError('Replay statistics/config differ from the generation')
        slot = row.get('logical_slot_id', row['id'])
        episode = out/'episodes'/hashlib.sha256(slot.encode()).hexdigest()[:24]
        # Failed attempts remain available; a rerun creates a fresh owned directory.
        attempt = episode/f'attempt_{time.time_ns()}'; attempt.mkdir(parents=True)
        runtime = attempt/'runtime'; runtime.mkdir()
        code = stage_runtime(spec['robotwin_source'], runtime/'RoboTwin', spec['assets_root'], policy)
        dependencies.extend(record['source'] for record in code)
        deployed = runtime/'RoboTwin/policy/WAMDenseGlobalSinkHTTP/deploy_policy.py'
        deployed.write_text(deployed.read_text().replace('from cosmos_framework.evaluation.robotwin_dense_evidence import verify_identity','from evo_wam.adapters.oracle_replay import verify_identity'))
        write(runtime/'source-manifest.json',dict(source_files=code,replay_policy_sha256=sha(deployed),generation_core_source=str(core),generation_core_sha256=sha(core)))
        (runtime/'inputs').mkdir()
        for key, name in [('statistics','state_action_stats.json'),('train_config','train_config.yaml')]:
            shutil.copy2(spec[key], runtime/'inputs'/name)
        write(runtime/'evaluation_identity.json', dict(checkpoint=identity['checkpoint'],
            checkpoint_metadata_sha256=identity['checkpoint_metadata_sha256'],
            input_sha256={p.name:sha(p) for p in (runtime/'inputs').iterdir()}))
        native = dict(row, logical_slot_id=slot, directory=str(case.parent), status='generated')
        native.setdefault('inference_noise_seed', row['seed'])
        native.setdefault('initial_observation_path', row.get('initial_observation'))
        initial = Path(native['initial_observation_path'])
        native.setdefault('initial_observation_sha256', sha(initial))
        write(attempt/'episode_spec.json', native)
        tape = case
        if not (case.parent/'initial_condition.json').is_file():
            wrapper = attempt/'source'; wrapper.mkdir(); tape = wrapper/'client_rollout'; tape.symlink_to(case, target_is_directory=True)
            with np.load(initial, allow_pickle=False) as data:
                if 'q0' not in data: raise ValueError('Oracle needs the original simulator registry observation')
                image = observation_image(dict(observation={k[:-4]:dict(rgb=data[k]) for k in data.files if k.endswith('_rgb')}))
                condition = dict(episode_key=native['episode_key'],global_rgb_sha256=array_sha256(image),state=data['q0'].astype(np.float32).tolist())
            write(wrapper/'initial_condition.json', condition)
        env = {k:v for k,v in os.environ.items() if not k.startswith(('RAY_','TORCHELASTIC_','PET_')) and k not in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT')}
        env.update({k:str(v) for k,v in spec.get('native_env',{}).items()})
        xdg = attempt/'xdg-runtime'; xdg.mkdir(mode=0o700)
        env.update(CUDA_VISIBLE_DEVICES=str(spec.get('gpu',0)),LD_LIBRARY_PATH='',
            PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[2]),str(source),str(runtime/'RoboTwin')]),EVO_ORACLE_GENERATION_CORE=str(core),XDG_RUNTIME_DIR=str(xdg),
            DENSE_EVAL_IDENTITY=str(runtime/'evaluation_identity.json'),DENSE_REPLAY_SOURCE=str(tape),DENSE_SAVE_LOSSLESS_RGB='0',
            DREAMZERO_ROBOTWIN_INFERENCE_NOISE_SEED=str(native['inference_noise_seed']),
            DREAMZERO_ROBOTWIN_NORMALIZATION_STATISTICS_PATH_OVERRIDE=str(runtime/'inputs/state_action_stats.json'),
            DREAMZERO_ROBOTWIN_NORMALIZATION_STATISTICS_SHA256=sha(spec['statistics']),
            ROBOTWIN_POLICY_ACTION_RENDER_MODE='all',ROBOTWIN_QUIET_STEPS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',ROBOTWIN_SAPIEN_CPU_THREADS='2')
        tag='oracle_'+attempt.name
        argv=[spec['robotwin_python'],'script/eval_policy.py','--config','policy/WAMDenseGlobalSinkHTTP/deploy_policy.yml','--policy_ckpt_path',identity['checkpoint'],'--overrides']
        values=dict(task_name=native['task_name'],task_config=native['task_config'],ckpt_setting=tag,policy_name='WAMDenseGlobalSinkHTTP',instruction_type=native['instruction_type'],instruction_override=native['instruction'],fixed_seed=native['fixed_seed'],seed=native['seed'],test_num=1,host='127.0.0.1',port=1,timeout_s=7200,imagined_dump_dir=str(attempt/'client_rollout'),episode_key=native['episode_key'],dense_mode='self_imagine',physics_failure_retry=True,max_physics_attempts=1,eval_step_limit_override=native['max_steps'])
        for k,v in values.items(): argv.extend(['--'+k,str(v)])
        write(attempt/'command.json',argv)
        result=dict(native,replay_directory=str(attempt),replay_status='infrastructure_error',official_success=None,first_success_step=None,
            source_result_sha256=sha(case.parent/'result.json') if (case.parent/'result.json').is_file() else sha(manifest),source_actions_file_sha256=sha(case/'actions_qpos.npy'))
        process=None
        try:
            with (attempt/'client.log').open('w') as log:
                process=subprocess.Popen(argv,cwd=runtime/'RoboTwin',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=process.wait(timeout=spec.get('episode_timeout',7200))
            ledgers=sorted((runtime/'RoboTwin/eval_result'/native['task_name']/'WAMDenseGlobalSinkHTTP'/native['task_config']/tag).glob('*/physics_retry_ledger.json'))
            if not ledgers:raise RuntimeError('No simulator outcome; see client.log')
            ledger=read(ledgers[-1]);write(attempt/'physics_ledger.json',ledger)
            if ledger['seed']!=native['fixed_seed'] or ledger['instruction']!=native['instruction'] or len(ledger['attempts'])!=1:
                raise ValueError('Replay scene/attempt identity changed')
            invalid=ledger.get('final_counted') is False and any(a.get('physics_retry_triggered') for a in ledger['attempts'])
            if invalid: result.update(replay_status='physics_invalid',physical_valid=False)
            else:
                if code:raise RuntimeError('Simulator exited with code '+str(code))
                execution=audit_artifacts(attempt/'client_rollout','self_imagine')['execution']
                trace=execution['oracle_steps'];count=execution['steps_executed'];success=ledger['final_success']
                first=next((r['step'] for r in trace if r['success']),None)
                if len(trace)!=count or any(r['step']!=i+1 or r['environment_steps']!=i+1 for i,r in enumerate(trace)):
                    raise ValueError('Noncontiguous Oracle replay trace')
                if type(success) is not bool or success!=execution['termination']['official_success'] or first!=execution['first_success_step'] or (first!=count if success else first is not None):
                    raise ValueError('Oracle replay did not stop at first success')
                result.update(replay_status='completed',official_success=success,physical_valid=True,first_success_step=first,steps_executed=count,oracle_trace=trace,
                    execution_manifest_sha256=sha(attempt/'client_rollout/execution_manifest.json'),initial_condition_sha256=sha(attempt/'initial_condition.json'),model_queries=0,artifact_audit_passed=True)
        except Exception as exc:
            result['error']=repr(exc);write(attempt/'result.json',result);raise
        finally:stop(process)
        write(attempt/'result.json',result);results.append(result)
        dependencies.extend([str(case),str(initial)])
    target=out/'oracle-results.json';write(target,dict(episodes=results))
    complete(context,[target,out/'episodes',out/'asset-binding.json'],oracle_results=str(target),manifest=manifest,dependency_paths=dependencies)
