"""Launch an archived resolved Cosmos configuration with explicit path rebinding."""
import argparse,json,os,sys
from pathlib import Path
from evo_wam.io import read,write,sha


def relocate(value,bindings):
    if isinstance(value,str):
        for old,new in sorted(bindings.items(),key=lambda kv:-len(kv[0])):
            if not old.startswith('/') or not new.startswith('/'):raise ValueError('Path bindings must be absolute')
            if value==old or value.startswith(old.rstrip('/')+'/'):return new.rstrip('/')+value[len(old.rstrip('/')):]
        return value
    if isinstance(value,list):return [relocate(v,bindings) for v in value]
    if isinstance(value,dict):return {k:relocate(v,bindings) for k,v in value.items()}
    return value


def override(config,values):
    for key,value in values.items():
        parts=key.split('.');node=config
        for part in parts[:-1]:node=node[part]
        if parts[-1] not in node:raise KeyError('Override must name an existing field: '+key)
        node[parts[-1]]=value
    return config


def materialize(source,bindings,overrides,destination):
    import yaml
    data=yaml.safe_load(Path(source).read_text())
    data=override(relocate(data,bindings),overrides)
    Path(destination).write_text(yaml.safe_dump(data,sort_keys=False))
    return data


def load_callable(name):
    """Resolve module.Class.method as well as module.function in archived YAML."""
    import importlib
    if not name:return None
    parts=name.split('.')
    for end in range(len(parts),0,-1):
        module='.'.join(parts[:end])
        try:obj=importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if not (module==exc.name or module.startswith(exc.name+'.')):raise
            continue
        for attr in parts[end:]:obj=getattr(obj,attr)
        return obj
    raise ImportError(name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source-root',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    sys.path.insert(0,a.source_root)
    from cosmos_framework.utils import serialization
    from cosmos_framework.utils.config import Config
    serialization.load_callable=load_callable
    config=serialization.from_yaml(a.config,clazz=Config)
    if a.check_only:
        print(json.dumps(dict(config_reconstructed=True,config_type=type(config).__name__,max_iter=config.trainer.max_iter,checkpoint=config.checkpoint.load_path)));return
    from cosmos_framework.scripts.train import launch
    launch(config,argparse.Namespace(deterministic=False,attach_vscode_debugger=False,config=a.config,sft_toml=a.config,opts=[]))

if __name__=='__main__':main()
