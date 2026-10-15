"""GPU regression for exact FusedAdam FP32 master/moment checkpoint round trips."""
import argparse,copy,hashlib,importlib.util
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--original-module',type=Path,required=True,
                   help='Unmodified reference retained before applying the resume patch')
    p.add_argument('--patched-module',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source_root))
    import torch,transformer_engine
    from evo_wam.io import write
    original_spec=importlib.util.spec_from_file_location('fused_adam_resume_original',a.original_module)
    original_module=importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original_module);Original=original_module.FusedAdam
    spec=importlib.util.spec_from_file_location('fused_adam_resume_candidate',a.patched_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);Fixed=module.FusedAdam
    torch.manual_seed(193)
    initial=torch.randn((64,64),device='cuda',dtype=torch.bfloat16)*.02
    gradients=[torch.randn_like(initial)*.3 for _ in range(2)]
    settings=dict(lr=1e-5,betas=(.9,.99),eps=1e-8,weight_decay=.05,capturable=True,master_weights=True)
    def make(kind,value):
        parameter=torch.nn.Parameter(value.clone());return parameter,kind([parameter],**settings)
    def update(parameter,optimizer,gradient):
        parameter.grad=gradient.clone();optimizer.step();parameter.grad=None
    baseline,optimizer=make(Fixed,initial);legacy,old=make(Original,initial)
    update(baseline,optimizer,gradients[0]);update(legacy,old,gradients[0])
    assert torch.equal(baseline,legacy)
    saved_parameter=baseline.detach().clone();saved=copy.deepcopy(optimizer.state_dict());old_saved=copy.deepcopy(old.state_dict())
    assert 'master_param' in optimizer.state[baseline]
    assert 'master_param' not in old.state[legacy]
    update(baseline,optimizer,gradients[1]);update(legacy,old,gradients[1])
    resumed,restored=make(Fixed,saved_parameter);restored.load_state_dict(saved)
    assert restored.master_state_restored
    assert all(restored.state[resumed][key].dtype==torch.float32 for key in ['master_param','exp_avg','exp_avg_sq'])
    update(resumed,restored,gradients[1])
    assert torch.equal(baseline,resumed)
    errors={}
    for key in ['master_param','exp_avg','exp_avg_sq']:
        expected=optimizer.state[baseline][key];actual=restored.state[resumed][key]
        assert torch.equal(expected,actual);errors[key]=float((expected-actual).abs().max())
    degraded,degraded_optimizer=make(Original,saved_parameter);degraded_optimizer.load_state_dict(old_saved)
    update(degraded,degraded_optimizer,gradients[1])
    legacy_error=float((old.param_groups_master[0]['params'][0]-degraded_optimizer.param_groups_master[0]['params'][0]).abs().max())
    assert legacy_error>0
    write(a.output,dict(status='passed',scope='optimizer component GPU regression; not full checkpoint training',
        torch=torch.__version__,transformer_engine=transformer_engine.__version__,parameter_dtype='bfloat16',
        restored_fp32_buffers_equal=True,restored_model_parameter_equal=True,fp32_buffer_max_abs_errors=errors,
        legacy_fp32_master_resume_error=legacy_error,
        original_source_sha256=hashlib.sha256(a.original_module.read_bytes()).hexdigest(),
        fixed_source_sha256=hashlib.sha256(a.patched_module.read_bytes()).hexdigest()))
    a.output.chmod(0o600)


if __name__=='__main__':main()
