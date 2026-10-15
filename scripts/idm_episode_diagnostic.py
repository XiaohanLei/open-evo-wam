"""Score real executed actions/video with IDM without approving training prefixes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    import cv2
    import numpy as np
    from evo_wam.io import read,write,sha
    from evo_wam.verification.robotwin import load_predictor,score_prefixes
    episode = a.episode.resolve();trace=episode/'client_rollout'
    if not (trace/'COMPLETE').is_file():
        raise ValueError('Completed simulator trace required')
    execution=read(trace/'execution_manifest.json');spec=read(episode/'episode_spec.json')
    actions=np.load(trace/'executed_actions.npy',allow_pickle=False)
    states=np.load(trace/'real_qpos.npy',allow_pickle=False)
    n=execution['steps_executed']
    if n<1 or actions.shape!=(n,14) or states.shape!=(n+1,14):
        raise ValueError('Action/state layout differs')
    cap=cv2.VideoCapture(str(trace/'action_boundary_video.mp4'));frames=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if len(frames)!=n+1:raise ValueError('Video/action count differs')
    video=np.stack(frames)
    ends=sorted(set([min(64,n),min(128,n),n]))
    report=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                task=spec['task_name'],fixed_seed=spec['fixed_seed'],
                checkpoint='idm-default',official_success=execution['termination']['official_success'],
                steps_executed=n,input='decoded_real_simulator_boundary_video_and_executed_actions',
                lossless_input=False,robotwin_calibration_verified=False,
                training_prefix_approval=False,video_sha256=sha(trace/'action_boundary_video.mp4'),
                actions_sha256=sha(trace/'executed_actions.npy'),states_sha256=sha(trace/'real_qpos.npy'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        write(a.output,report);a.output.chmod(0o600)
    save();start=time.monotonic()
    try:
        predict,stats=load_predictor(dict(source_root=str(a.source_root),
            idm_checkpoint=str(a.checkpoint),tokenizer=str(a.tokenizer)))
        from evo_wam.native_idm import native_module
        data_module = native_module("data")
        canvas_views, render_views = data_module.canvas_views, data_module.render_views
        transformed=render_views(canvas_views(video,{}),False,0)
        result=score_prefixes(spec['episode_key'],ends,transformed,actions,
            [states[i] for i in range(0,n,64)],spec['instruction'],predict,stats,
            'dense','evo-wam-real-episode-diagnostic')
        for window in result['windows']:
            window['state_source']='real_observation_boundary'
        report.update(status='passed',scores=result,elapsed_seconds=time.monotonic()-start)
        save()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-start)
        save();raise


if __name__=='__main__':main()
