"""Round-aware native RoboTwin recipes, for the standalone RoboTwin profile."""
from pathlib import Path
from evo_wam.native_idm import remap_environment
from evo_wam.io import read, write, sha
from .cosmos_training import materialize


def select(context):
    spec = dict(context['spec'])
    recipe = spec['recipe']
    number = context['round']
    if recipe != 'robotwin' or not 1 <= number <= 4:
        raise ValueError('This profile supports RoboTwin rounds 1..4')
    recipes = spec.get('recipe_configs', {})
    config = recipes.get(number, recipes.get(str(number)))
    if not config:
        raise ValueError('An explicit standalone recipe is required for each round')
    spec['train_config'] = config
    nodes = 16
    adapter = 'native'
    spec['launcher'] = [spec['python'], '-m', 'torch.distributed.run', f'--nnodes={nodes}',
                        '--nproc-per-node=8', '--node-rank=0',
                        '--master-addr='+spec['master_addr'], '--master-port='+str(spec['master_port'])]
    return dict(context, spec=spec), adapter


def prepare_native(context):
    spec = context['spec']; out = Path(context['output']); out.mkdir(parents=True, exist_ok=True)
    manifest = Path(spec.get('manifest') or read(Path(context['previous_stage'])/'result.json')['manifest'])
    data = read(manifest)
    expected = 0.004184869576192262
    if data['idm_threshold'] != expected:
        raise ValueError('Training pool threshold differs from the selected branch and round')
    if sha(spec['statistics']) != data['statistics']['sha256']:
        raise ValueError('Training and exported pool statistics differ')
    overrides = {
        'model.config.tokenizer.vae_path': spec['vae'],
        'checkpoint.load_path': context['parent_checkpoint'],
        'trainer.callbacks.sft_audit.root': str(out),
        'dataloader_train.dataloaders.base.dataloader.dataset.root': spec['base_data'],
        'dataloader_train.dataloaders.d2.dataloader.dataset.manifest_path': spec['aux_manifest'],
        'dataloader_train.dataloaders.d4.dataloader.dataset.manifest_path': spec['aux_manifest'],
        'dataloader_train.dataloaders.adaptation.dataloader.dataset.manifest_path': str(manifest),
    }
    path = out/'resolved_config.yaml'
    config = materialize(spec['train_config'], spec.get('asset_bindings', {}), overrides, path)
    parallel = config['model']['config']['parallelism']
    if (parallel['data_parallel_shard_degree'], parallel['data_parallel_replicate_degree']) != (8,16):
        raise ValueError('Frozen regular rounds require shard8 x replicate16')
    if config['checkpoint']['load_training_state'] or config['trainer']['max_iter'] != 1000:
        raise ValueError('Expected 1000 updates with a fresh optimizer per round')
    env = dict(spec.get('native_env', {}), BASE_CHECKPOINT_PATH=context['parent_checkpoint'],
        WAN_VAE_PATH=spec['vae'], ROBOTWIN_DENSE_BASE_ROOT=spec['base_data'],
        ROBOTWIN_DENSE_BOOTSTRAP_SLOTS='32', ROBOTWIN_DENSE_AUX_MANIFEST=spec['aux_manifest'],
        ROBOTWIN_STATE_ACTION_STATS_PATH=spec['statistics'], ROBOTWIN_ADAPTATION_MANIFEST=str(manifest),
        IMAGINAIRE_OUTPUT_ROOT=str(out/'training'), WANDB_MODE='disabled')
    env = remap_environment(env, spec.get('native_environment_keys', {}))
    job = config['job']
    checkpoint = out/'training'/job['project']/job['group']/job['name']/'checkpoints/iter_000001000'
    write(out/'RECIPE.json', dict(recipe=spec['recipe'], round=context['round'], world_size=128,
        source_config_sha256=sha(spec['train_config']), materialized_config_sha256=sha(path),
        manifest=str(manifest), parent_checkpoint=context['parent_checkpoint'], checkpoint=str(checkpoint)))
    return path, env, checkpoint


def train(context):
    context, adapter = select(context)
    spec = context['spec']; out = Path(context['output'])
    config, env, checkpoint = prepare_native(context)
    argv = [*spec['launcher'], '-m', 'evo_wam.adapters.cosmos_training',
            '--source-root', spec['source_root'], '--config', str(config)]
    launch = dict(argv=argv, env=env, source_root=spec['source_root'], config_path=str(config),
                  config_sha256=sha(config), package_src=str(Path(__file__).resolve().parents[2]))
    write(out/'worker-launch.json', launch)
    from .training import run
    run(dict(context, spec=dict(spec, train_argv=argv, native_env=env,
        extra_python_paths=[launch['package_src'], *spec.get('extra_python_paths', [])],
        checkpoint_output=str(checkpoint), dependency_paths=[str(Path(spec.get('manifest') or read(Path(context['previous_stage'])/'result.json')['manifest'])),
            spec['statistics'],spec['aux_manifest']],
        recipe_artifacts=[str(config),str(out/'RECIPE.json'),str(out/'worker-launch.json')])) )


def worker():
    import argparse, os, subprocess
    p = argparse.ArgumentParser(); p.add_argument('--launch-receipt', required=True)
    p.add_argument('--node-rank', type=int, required=True); args = p.parse_args()
    launch = read(args.launch_receipt)
    if sha(launch['config_path']) != launch['config_sha256']:
        raise ValueError('Prepared configuration changed')
    argv = launch['argv']; nodes = int(next(v.split('=')[1] for v in argv if v.startswith('--nnodes=')))
    if not 1 <= args.node_rank < nodes:
        raise ValueError('Worker rank outside prepared node count')
    argv = [f'--node-rank={args.node_rank}' if v == '--node-rank=0' else v for v in argv]
    env = dict(os.environ, **launch['env'])
    env['PYTHONPATH'] = os.pathsep.join([launch['source_root'], launch['package_src'], env.get('PYTHONPATH','')])
    subprocess.run(argv, env=env, cwd=launch['source_root'], check=True)


if __name__ == '__main__':
    worker()
