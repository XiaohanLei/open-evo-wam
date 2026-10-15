"""Install/verify the pinned CUDA12 training environment without system changes."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--venv',type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--uv',default='uv')
    p.add_argument('--cuda-home',type=Path,required=True);p.add_argument('--cuda-arch',required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    v=a.venv.resolve();python=v/'bin/python';env=os.environ.copy()
    if not a.verify_only:
        if v.exists():raise ValueError('Use a new environment; existing environments are preserved')
        if not (a.cuda_home/'bin/nvcc').is_file():raise ValueError('CUDA toolkit with nvcc is required')
        subprocess.run([a.uv,'venv','--python',a.python,str(v)],check=True)
        version=subprocess.check_output([str(python),'-c','import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")'],text=True).strip()
        assert version=='python3.11','The lock was validated with Python3.11'
        packages=v/'lib'/version/'site-packages';cudnn=packages/'nvidia/cudnn';nccl=packages/'nvidia/nccl'
        env.update(UV_CACHE_DIR=str(v.parent/'.cache/uv'),CUDA_HOME=str(a.cuda_home.resolve()),
            NVTE_CUDA_ARCHS=a.cuda_arch,NVTE_FRAMEWORK='pytorch',MAX_JOBS='4',CMAKE_BUILD_PARALLEL_LEVEL='4',
            CUDNN_PATH=str(cudnn),NVTE_CUDNN_INCLUDE_PATH=str(cudnn/'include'),NVTE_CUDNN_LIB_PATH=str(cudnn/'lib'),
            NVTE_NCCL_INCLUDE_PATH=str(nccl/'include'),CPLUS_INCLUDE_PATH=os.pathsep.join([str(cudnn/'include'),str(nccl/'include')]))
        lock=Path(__file__).parents[1]/'requirements-wam-training.lock.txt'
        lines=[x for x in lock.read_text().splitlines() if not x.startswith(('transformer-engine==','transformer-engine-torch==','transformer-engine-cu12=='))]
        with tempfile.TemporaryDirectory(dir=v.parent) as temp:
            base=Path(temp)/'base.lock.txt';base.write_text('\n'.join(lines)+'\n')
            subprocess.run([a.uv,'pip','install','--python',str(python),'--extra-index-url','https://download.pytorch.org/whl/cu128','--index-strategy','unsafe-best-match','-r',str(base)],env=env,check=True)
        subprocess.run([a.uv,'pip','install','--python',str(python),'--no-build-isolation','transformer-engine[pytorch,core-cu12]==2.12.0'],env=env,check=True)
    code='import json,torch,transformer_engine,transformer_engine_torch; assert torch.__version__=="2.8.0+cu128"; assert transformer_engine.__version__=="2.12.0"; assert hasattr(transformer_engine_torch,"multi_tensor_adam"); print(json.dumps(dict(status="passed",torch=torch.__version__,transformer_engine=transformer_engine.__version__,cuda_available=torch.cuda.is_available())))'
    subprocess.run([str(python),'-c',code],env=env,check=True)


if __name__=='__main__':main()
