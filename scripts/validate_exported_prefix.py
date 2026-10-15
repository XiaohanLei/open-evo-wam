"""Audit approved prefix bytes and actual native training samples, without model training."""
import argparse
from datetime import datetime,timezone
import hashlib,os
from pathlib import Path
import sys,time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import cv2,numpy as np,pyarrow.parquet as pq,torch
    from evo_wam.io import read,write,sha
    from evo_wam.verification.stage import NativeFrames
    dataset=a.dataset.resolve();manifest=read(dataset/'ADAPTATION_MANIFEST.json')
    index=[__import__('json').loads(line) for line in (dataset/'meta/index.jsonl').read_text().splitlines()]
    decisions={row['id']:row for row in read(a.decisions)['episodes']}
    report=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                training=False,manifest_sha256=sha(dataset/'ADAPTATION_MANIFEST.json'),episodes=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():write(a.output,report);a.output.chmod(0o600)
    save();start=time.monotonic()
    try:
        if not read(dataset/'READY.json')['passed']:raise ValueError('Dataset not ready')
        os.environ['ROBOTWIN_STATE_ACTION_STATS_PATH']=manifest['statistics']['path']
        os.environ['WANDB_MODE']='disabled'
        sys.path.insert(0,str(a.source_root))
        from cosmos_framework.data.vfm.action.datasets.robotwin_imagined_adaptation_dataset import get_robotwin_adaptation_dataset
        from cosmos_framework.data.vfm.action.datasets.robotwin_balanced_adaptation import get_balanced_adaptation_dataset
        for row in index:
            decision=decisions[row['source_logical_slot_id']]
            if not decision['accepted']:raise ValueError('Unapproved source')
            reader=NativeFrames(decision['source'],dict(backbone='cosmos3-robotwin'))
            t=decision['accepted_step'];ep=Path(row['video_path']).parent;marker=read(ep/'EXPORT_COMPLETE.json')
            if sha(ep/'EXPORT_COMPLETE.json')!=row['export_marker_sha256']:raise ValueError('Export marker changed')
            for name,digest in marker['file_sha256'].items():
                if sha(ep/name)!=digest:raise ValueError('Export file changed')
            table=pq.read_table(ep/'data.parquet').to_pydict()
            states=np.asarray(table['observation.state'],np.float32);actions=np.asarray(table['action'],np.float32)
            commands=reader.actions[:t].astype(np.float32)
            np.testing.assert_array_equal(states,np.concatenate([reader.state[None],commands]))
            np.testing.assert_array_equal(actions,np.concatenate([commands,commands[-1:]]))
            cap=cv2.VideoCapture(str(ep/'video.mp4'));digest=hashlib.sha256();count=0
            try:
                while True:
                    ok,bgr=cap.read()
                    if not ok:break
                    actual=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    expected=reader.initial if count==0 else reader.frames[count-1]
                    if expected.shape==(352,640,3):expected=np.pad(expected,((0,8),(0,0),(0,0)),mode='edge')
                    np.testing.assert_array_equal(actual,expected)
                    digest.update(actual.tobytes());count+=1
            finally:cap.release()
            if count!=t+1 or digest.hexdigest()!=marker['training_rgb_sha256']:raise ValueError('RGB identity differs')
            report['episodes'].append(dict(id=row['source_logical_slot_id'],prefix_steps=t,
                decoded_rgb_frames=count,lossless_rgb_equal=True,state_action_rows_equal=True,
                video_sha256=sha(ep/'video.mp4'),parquet_sha256=sha(ep/'data.parquet')))
        if len(index)!=1:raise ValueError('This limited sample audit expects one approved episode')
        raw=get_robotwin_adaptation_dataset(manifest_path=dataset/'ADAPTATION_MANIFEST.json',raw=True)
        bootstrap0=raw[0];bootstrap31=raw[31]
        assert torch.equal(bootstrap0['video'],bootstrap31['video'])
        assert torch.equal(bootstrap0['action'],bootstrap31['action'])
        sample_audit=[]
        for idx in [0,31,32,len(raw)-1]:
            sample=raw[idx];boundary=int(sample['robotwin_window_start'])
            target=np.concatenate([states[boundary:boundary+1],actions[np.minimum(np.arange(boundary,boundary+64),len(actions)-1)]])
            expected=(torch.from_numpy(target)-raw._state_action_mean)/raw._state_action_std
            torch.testing.assert_close(sample['action'],expected,rtol=0,atol=0)
            assert sample['action'].shape==(65,14) and torch.isfinite(sample['action']).all()
            bootstrap=bool(sample['robotwin_bootstrap']);assert sample['video'].shape[1]==(66 if bootstrap else 98)
            assert sample['condition_frame_indexes_vision']==list(range(1 if bootstrap else 9))
            sample_audit.append(dict(index=idx,boundary=boundary,bootstrap=bootstrap,
                terminal_hold_frames=int(sample['robotwin_terminal_hold_frames']),video_shape=list(sample['video'].shape)))
        assert sample_audit[-1]['terminal_hold_frames']==30
        balanced=get_balanced_adaptation_dataset(manifest_path=dataset/'ADAPTATION_MANIFEST.json')
        transformed=next(iter(balanced));plan=transformed['sequence_plan']
        assert int(transformed['raw_action_dim'])==14 and transformed['action'].shape==(65,64)
        assert torch.count_nonzero(transformed['action'][:,14:])==0
        assert plan.condition_frame_indexes_action==[0]
        assert plan.condition_frame_indexes_vision==list(range(1 if bool(transformed['robotwin_bootstrap']) else 9))
        assert plan.action_start_frame_offset==(0 if bool(transformed['robotwin_bootstrap']) else 32)
        assert torch.isfinite(transformed['action']).all()
        report.update(status='passed',elapsed_seconds=time.monotonic()-start,windows=len(raw),
            bootstrap_slots=32,terminal_hold_steps=30,samples=sample_audit,
            balanced_cells=[list(cell) for cell in balanced.cell_names],empty_cells=len(balanced.empty_cells),
            transformed_video_shape=list(transformed['video'].shape),transformed_action_shape=list(transformed['action'].shape),
            clean_state_action_token_indexes=plan.condition_frame_indexes_action,
            padded_action_channels_zero=True,raw_action_dim=14)
        save()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-start)
        save();raise


if __name__=='__main__':main()
