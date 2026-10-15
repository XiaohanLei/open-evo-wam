"""Compare packed SDPA against independent dense float32 attention."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    sys.path.insert(0,str(a.source_root.resolve()))
    import torch
    from cosmos_framework.model.attention import attention
    from cosmos_framework.model.attention.masks import CausalType
    torch.manual_seed(42)
    cases = []
    for causal in [None,CausalType.TopLeft,CausalType.BottomRight,CausalType.DontCare]:
        qs,ks = [3,5],[3,5] if causal==CausalType.DontCare else [5,4]
        q = torch.randn((1,sum(qs),4,128),device=a.device,dtype=torch.bfloat16)
        k,v = [torch.randn((1,sum(ks),2,128),device=a.device,dtype=torch.bfloat16) for _ in range(2)]
        q,k,v = [t.requires_grad_() for t in (q,k,v)]
        qr,kr,vr = [t.detach().float().requires_grad_() for t in (q,k,v)]
        qc = torch.tensor([0,qs[0],sum(qs)],device=a.device,dtype=torch.int32)
        kc = torch.tensor([0,ks[0],sum(ks)],device=a.device,dtype=torch.int32)
        out = attention(q,k,v,is_causal=causal is not None,causal_type=causal,
            cumulative_seqlen_Q=qc,cumulative_seqlen_KV=kc,
            max_seqlen_Q=max(qs),max_seqlen_KV=max(ks),backend='sdpa')
        refs,qa,ka = [],0,0
        for nq,nk in zip(qs,ks):
            x = qr[:,qa:qa+nq].transpose(1,2)
            y,z = [t[:,ka:ka+nk].transpose(1,2).repeat_interleave(2,dim=1) for t in (kr,vr)]
            logits = x@y.transpose(-2,-1)/128**.5
            if causal is not None:
                shift = nk-nq if causal==CausalType.BottomRight else 0
                mask = torch.arange(nk,device=a.device)[None,:]<=torch.arange(nq,device=a.device)[:,None]+shift
                logits = logits.masked_fill(~mask,float('-inf'))
                valid = mask.any(dim=-1)[None,None,:,None]
                logits = torch.where(valid,logits,torch.zeros_like(logits))
                probabilities = torch.softmax(logits,dim=-1)*valid
            else:
                probabilities = torch.softmax(logits,dim=-1)
            refs.append((probabilities@z).transpose(1,2))
            qa,ka = qa+nq,ka+nk
        ref = torch.cat(refs,dim=1)
        torch.testing.assert_close(out.float(),ref,atol=.02,rtol=.02)
        upstream = torch.randn_like(ref)
        actual_grad = torch.autograd.grad((out.float()*upstream).sum(),(q,k,v))
        expected_grad = torch.autograd.grad((ref*upstream).sum(),(qr,kr,vr))
        errors = []
        for actual,expected in zip(actual_grad,expected_grad):
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            torch.testing.assert_close(actual.float(),expected,atol=.04,rtol=.04)
            errors.append(float((actual.float()-expected).abs().max()))
        cases.append(dict(causal=str(causal),max_abs_error=float((out.float()-ref).detach().abs().max()),
                          gradient_max_abs_errors_qkv=errors))
    report = dict(status='passed',backend='torch-sdpa',torch=torch.__version__,
                  packed_gqa_checks=cases,attention_backward_verified=True,
                  full_model_training_verified=False)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    a.report.chmod(0o600)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
