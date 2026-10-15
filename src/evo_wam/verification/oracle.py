"""One VLM judgment at the official replay's first success, without Oracle labels in prompts."""
import hashlib
from pathlib import Path
from evo_wam.io import read,write,sha
from evo_wam.stages import json_rows,complete
from .stage import NativeFrames,png
from .protocol import Protocol
from .client import Client


def endpoint(proof, length):
    status=proof['replay_status']
    if status=='physics_invalid':return None,'physics_invalid'
    if status!='completed' or type(proof['official_success']) is not bool:
        raise ValueError('Oracle replay is unresolved')
    if proof.get('model_queries')!=0 or proof.get('physical_valid') is not True or proof.get('artifact_audit_passed') is not True:
        raise ValueError('Oracle replay lacks valid simulator evidence')
    if not proof['official_success']:return None,'oracle_failure'
    t=proof['first_success_step']
    if type(t) is not int or not 0<t<=length or proof['steps_executed']!=t:
        raise ValueError('Invalid first-success endpoint')
    trace=proof['oracle_trace']
    if len(trace)!=t or any(r['step']!=i+1 or type(r['success']) is not bool for i,r in enumerate(trace)):
        raise ValueError('Invalid Oracle step trace')
    if any(r['success'] for r in trace[:-1]) or not trace[-1]['success']:
        raise ValueError('Replay did not stop at first success')
    return (None,'prefix_too_short') if t<34 else (t,'oracle_success')


def verify(context):
    spec=context['spec'];out=Path(context['output']);protocol=Protocol('robotwin_dense')
    manifest=spec.get('generated_manifest') or read(Path(context['previous_stage'])/'result.json')['manifest']
    oracle_results=spec.get('oracle_results')
    if not oracle_results:
        from evo_wam.adapters.oracle_replay import run
        replay_out=out/'replay'
        replay_spec=dict(spec['oracle_replay'],generated_manifest=manifest)
        run(dict(context,spec=replay_spec,output=str(replay_out),stage='oracle-replay'))
        oracle_results=read(replay_out/'result.json')['oracle_results']
    rows=json_rows(manifest);replays=json_rows(oracle_results);proofs={r['logical_slot_id']:r for r in replays}
    if len(proofs)!=len(replays):raise ValueError('Duplicate Oracle replay identity')
    client=Client(out/'vlm-requests',spec['endpoint'],spec['model_revision'],protocol.parse,
        max_attempts=spec.get('max_attempts',6),timeout=spec.get('timeout',240),api_key_env=spec.get('api_key_env'))
    decisions=[];dependencies=[manifest,oracle_results];seen=set()
    for row in rows:
        uid=row['id'];slot=row.get('logical_slot_id',uid)
        if uid in seen:raise ValueError('Duplicate candidate identity')
        seen.add(uid);reader=NativeFrames(row,dict(backbone='cosmos3-robotwin'))
        proof=proofs[slot];receipt=Path(proof['replay_directory'])/'result.json'
        if read(receipt)!=proof:raise ValueError('Oracle ledger and individual receipt differ')
        if proof['instruction']!=reader.instruction or any(row.get(k)!=proof.get(k) for k in ('episode_key','fixed_seed')):
            raise ValueError('Oracle replay belongs to a different scene or instruction')
        action_path=reader.case/'actions_qpos.npy'
        if sha(action_path)!=proof['source_actions_file_sha256']:
            raise ValueError('Oracle replay used a different action tape')
        t,reason=endpoint(proof,reader.length)
        record=dict(id=uid,source=row,case=str(reader.case),round=context['round'],task=reader.task,
            instruction=reader.instruction,selection_mode='oracle_vlm',accepted=False,reason=reason,
            oracle_result=str(receipt),oracle_result_sha256=sha(receipt),oracle=proof,
            actions_sha256=sha(action_path),generation_sha256=sha(reader.case/'generation_manifest.json'),
            target_source='imagined_rgb_and_generated_actions',history=[],vote_groups=[])
        dependencies.extend([*map(str,reader.paths),str(receipt)])
        if t is not None:
            first,current,state,actions=reader.at(t)
            gate=protocol.contract.prefix_gate(actions,state,t)
            record['gate']=gate
            if not gate['pass']:record['reason']='action_filtered'
            else:
                raw=[png(image) for image in protocol.contract.crops(first)+protocol.contract.crops(current)]
                description=None;chain={};key=hashlib.sha256(uid.encode()).hexdigest()
                for kind in ('description','judgment'):
                    payload=protocol.payload(reader.task,0,kind,reader.instruction,raw,description)
                    payload['model']=spec.get('model',payload['model'])
                    result=client.request(f'{key}/t{t}/{kind}',payload,dict(kind=kind,description=description,task=reader.task,goal=0))
                    chain[kind]=result
                    if kind=='description':description=result['prediction']
                decision=chain['judgment']['prediction']['decision'];chain['decision']=decision
                record.update(accepted=decision=='accept',reason=decision,accepted_step=t,visual_endpoint=t,
                    history=[dict(step=t,decision=decision)],
                    vote_groups=[dict(goal=0,endpoint=t,votes=[dict(id='0',endpoint=t,accepted=decision=='accept',record=chain)])])
        decisions.append(record)
    target=out/'decisions.json';write(target,dict(episodes=decisions))
    artifacts=[target]+([out/'vlm-requests'] if (out/'vlm-requests').exists() else [])
    complete(context,artifacts,decisions=str(target),dependency_paths=sorted(set(dependencies)),mode='oracle_first_success_single_vlm')
