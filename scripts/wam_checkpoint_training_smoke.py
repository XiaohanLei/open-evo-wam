"""Bounded actual-checkpoint training on two GPUs with CPU offload.

Uses the native model/loss, native parameter selection and original FusedAdam
CUDA update kernel. Optimizer buffers are staged one parameter at a time to fit
the validation hardware. This does not replace the unchanged 128-GPU recipe.
"""
import argparse,contextlib,hashlib,json,os,random,sys,time
from collections import deque
from pathlib import Path
from datetime import datetime,timezone,timedelta


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['initial','resume'],required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source_root))
    import numpy as np,torch,torch.distributed as dist,torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from torch.distributed.fsdp import CPUOffloadPolicy
    from omegaconf import DictConfig
    import hydra,yaml
    from evo_wam.io import read,write,sha
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',local_rank)
    torch.cuda.set_device(device);dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    a.output.mkdir(parents=True,exist_ok=True);target=a.output/f'{a.phase}-rank{rank}.json'
    report=dict(status='running',phase=a.phase,rank=rank,created_utc=datetime.now(timezone.utc).isoformat(),
        original_checkpoint_modified=False,formal_recipe_modified=False,world_size=dist.get_world_size(),
        native_model=True,native_loss=True,native_fused_adam_kernel=True,cpu_parameter_and_optimizer_offload=True,
        original_model_metadata_sha256=sha(a.checkpoint/'model/.metadata'),recipe_sha256=sha(a.recipe),updates=[])
    def save():write(target,report);target.chmod(0o600)
    save();start=time.monotonic()
    def local(value):return value.to_local() if hasattr(value,'to_local') else value
    def tensor_hash(value):
        value=local(value).detach().contiguous().cpu()
        return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    def model_hash(model):
        h=hashlib.sha256()
        for name,value in sorted(get_model_state_dict(model).items()):
            h.update(name.encode());h.update(tensor_hash(value).encode())
        return h.hexdigest()
    try:
        assert dist.get_world_size()==2
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        from cosmos_framework.model.vfm.mot import parallelize_unified_mot as pu
        from cosmos_framework.model.vfm.mot import parallelize_vfm_network as pv
        for module in [pu,pv]:
            original=module.fully_shard
            def offloaded(*args,_original=original,**kwargs):
                kwargs['offload_policy']=CPUOffloadPolicy(pin_memory=True)
                return _original(*args,**kwargs)
            module.fully_shard=offloaded
        from cosmos_framework.inference.model import structure_config
        config=yaml.safe_load(a.config.read_text());recipe=yaml.safe_load(a.recipe.read_text())
        model_dict=structure_config(config['model'],DictConfig)
        model_dict.config.parallelism.data_parallel_shard_degree=2
        model_dict.config.parallelism.data_parallel_replicate_degree=1
        model_dict.config.parallelism.enable_inference_mode=False
        model_dict.config.compile.enabled=False
        model_dict.config.activation_checkpointing.mode='full'
        model_dict.config.ema.enabled=False
        with contextlib.chdir(a.source_root):model=hydra.utils.instantiate(model_dict)
        # CPU offload parameters; retain execution buffers on the local GPU.
        model.net.to('cpu')
        for module in model.net.modules():
            for name,buffer in module.named_buffers(recurse=False):setattr(module,name,buffer.to(device))
        state=get_model_state_dict(model)
        source=a.checkpoint/'model' if a.phase=='initial' else a.output/'checkpoint-step1/model'
        dcp.load(state_dict=state,checkpoint_id=source)
        from cosmos_framework.utils.vfm.optimizer import _build_params_with_metadata
        from cosmos_framework.utils.vfm.fused_adam import FusedAdam
        optcfg=recipe['optimizer']
        selected=_build_params_with_metadata(model,keys_to_select=optcfg['keys_to_select'],
            lr_multipliers=optcfg['lr_multipliers'],base_lr=optcfg['lr'],
            disable_weight_decay_for_1d_params=optcfg['disable_weight_decay_for_1d_params'])
        names={id(parameter):name for name,parameter in model.net.named_parameters()}
        selected=[(names[id(parameter)],parameter,metadata) for parameter,metadata in selected]
        report.update(model_loaded=True,selected_parameter_tensors=len(selected),
            selected_global_numel=sum(parameter.numel() for _,parameter,_ in selected),
            total_global_numel=sum(parameter.numel() for parameter in model.net.parameters()))
        save();optimizer_state={};step=0
        dataset_manifest=read(a.dataset/'ADAPTATION_MANIFEST.json')
        os.environ['ROBOTWIN_STATE_ACTION_STATS_PATH']=dataset_manifest['statistics']['path']
        from cosmos_framework.data.vfm.action.datasets.robotwin_balanced_adaptation import get_balanced_adaptation_dataset
        from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn,JointDataLoader
        def native_batch(sample):
            # Reuse the actual joint loader's split/accumulate rules. Direct
            # custom_collate_fn output is an inner-loader batch, not the
            # packed list-of-lists consumed by native training_step.
            loader=object.__new__(JointDataLoader)
            loader.buffers={'smoke':deque()}
            loader.dataloaders={'smoke':iter([custom_collate_fn([sample])])}
            packed={}
            loader._update_output_batch(packed,loader._get_next_sample('smoke'))
            assert len(packed['text_token_ids'])==1
            assert len(packed['text_token_ids'][0])==1
            assert packed['text_token_ids'][0][0].ndim==1
            return packed
        balanced=get_balanced_adaptation_dataset(manifest_path=a.dataset/'ADAPTATION_MANIFEST.json')
        rng=np.random.default_rng(np.random.SeedSequence([20260907,rank,0,1]))
        if a.phase=='resume':
            saved=torch.load(a.output/f'checkpoint-step1/optimizer-rank{rank}.pt',map_location='cpu',weights_only=True)
            optimizer_state=saved['optimizer'];step=saved['step'];assert step==1
            prior=read(a.output/f'initial-rank{rank}.json')
            assert model_hash(model)==prior['saved_model_local_sha256']
            assert all(tensor_hash(value[key])==prior['optimizer_hashes'][name][key]
                       for name,value in optimizer_state.items() for key in ['master_param','exp_avg','exp_avg_sq'])
            rng.bit_generator.state=saved['sampler_state']
            torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],device)
            np.random.set_state(('MT19937',np.asarray(saved['numpy_rng'][1],np.uint32),saved['numpy_rng'][2],saved['numpy_rng'][3],saved['numpy_rng'][4]))
            def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value
            random.setstate(tuples(saved['python_rng']))
            report.update(fresh_model_reload_equal=True,optimizer_fp32_reload_equal=True,resumed_step=step)
        model.train();model.zero_grad(set_to_none=True)
        losses=[];indices=[];accum=int(recipe['trainer']['grad_accum_iter'])
        for micro in range(accum):
            index,_=balanced.sample_index(rng);indices.append(int(index))
            sample=balanced.dataset[index];batch=native_batch(sample)
            output,loss=model.training_step(batch,step)
            if not torch.isfinite(loss):raise ValueError('Nonfinite native loss')
            losses.append(float(loss.detach()));(loss/accum).backward();del output,loss,batch,sample
            report.update(microbatches_completed=micro+1,losses=losses);save()
        total=torch.zeros((),dtype=torch.float64,device=device);gradient_tensors=0
        for _,parameter,_ in selected:
            if parameter.grad is None:continue
            gradient=local(parameter.grad).detach()
            if not torch.isfinite(gradient).all():raise ValueError('Nonfinite model gradient')
            total+=gradient.double().square().sum().to(device);gradient_tensors+=1
        dist.all_reduce(total);norm=float(total.sqrt());assert gradient_tensors>0 and norm>0
        coefficient=min(1.,1./(norm+1e-6));updated=0
        for name,parameter,metadata in selected:
            if parameter.grad is None:continue
            storage=local(parameter).detach();before=tensor_hash(storage)
            gpu=torch.nn.Parameter(storage.to(device));gpu.grad=(local(parameter.grad).detach()*coefficient).to(device)
            optimizer=FusedAdam([gpu],lr=metadata.lr,betas=tuple(optcfg['betas']),eps=optcfg['eps'],
                weight_decay=optcfg['weight_decay'] if metadata.enable_weight_decay else 0.,capturable=True,master_weights=True)
            if name in optimizer_state:
                buffers=optimizer_state[name]
                optimizer.state[gpu]={key:buffers[key].to(device) for key in ['master_param','exp_avg','exp_avg_sq']}
                optimizer.param_groups[0]['step']=torch.tensor([step],dtype=torch.int32,device=device)
            optimizer.step();storage.copy_(gpu.detach().to(storage.device))
            buffers=optimizer.state[gpu];optimizer_state[name]={key:buffers[key].detach().cpu() for key in ['master_param','exp_avg','exp_avg_sq']}
            updated+=tensor_hash(storage)!=before
            del optimizer,gpu,buffers
        step+=1;model.zero_grad(set_to_none=True);assert updated>0
        report['updates'].append(dict(step=step,native_losses=losses,gradient_norm=norm,
            gradient_tensors=gradient_tensors,changed_local_parameter_tensors=updated,
            sampled_window_indexes=indices,grad_accum_iter=accum,effective_global_batch=accum*2))
        save();checkpoint=a.output/f'checkpoint-step{step}';checkpoint.mkdir(exist_ok=True)
        dcp.save(state_dict=get_model_state_dict(model),checkpoint_id=checkpoint/'model')
        numpy_rng=np.random.get_state();numpy_rng=[numpy_rng[0],numpy_rng[1].tolist(),numpy_rng[2],numpy_rng[3],numpy_rng[4]]
        torch.save(dict(optimizer=optimizer_state,step=step,sampler_state=rng.bit_generator.state,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),numpy_rng=numpy_rng,
            python_rng=list(random.getstate())),checkpoint/f'optimizer-rank{rank}.pt')
        report.update(saved_model_local_sha256=model_hash(model),optimizer_hashes={name:{key:tensor_hash(value[key])
            for key in ['master_param','exp_avg','exp_avg_sq']} for name,value in optimizer_state.items()},
            status='passed',elapsed_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
        save();dist.barrier()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-start)
        save();raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
