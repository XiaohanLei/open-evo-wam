"""Check the configured VLM API on a completed real simulator trace.

This is a post-episode diagnostic. It does not approve imagined training prefixes
or replace the simulator's official task-success condition.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--format-instructions', type=Path, required=True)
    p.add_argument('--api-key-env')
    a = p.parse_args()
    import cv2
    from evo_wam.io import read, write, sha, digest
    from evo_wam.verification.client import Client
    from evo_wam.verification.protocol import Protocol
    from evo_wam.verification.stage import png
    episode = a.episode.resolve()
    spec = read(episode/'episode_spec.json')
    trace = episode/'client_rollout'
    if not (trace/'COMPLETE').is_file():
        raise ValueError('A completed simulator trace is required')
    execution = read(trace/'execution_manifest.json')
    success = execution['termination']['official_success']
    if type(success) is not bool:
        raise ValueError('Official success must be explicit')
    video = trace/'action_boundary_video.mp4'
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count != execution['steps_executed']+1:
        raise ValueError('Video/action count differs')
    frames = []
    try:
        for index in [0,count-1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame = cap.read()
            if not ok:
                raise ValueError('Cannot decode trace boundary')
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    protocol = Protocol('robotwin_dense')
    raw = [png(x) for frame in frames for x in protocol.contract.crops(frame)]
    guidance = read(a.format_instructions)
    report = dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                  task=spec['task_name'],fixed_seed=spec['fixed_seed'],
                  model=a.model,model_revision=a.revision,official_success=success,
                  steps_executed=execution['steps_executed'],votes=[],
                  input='decoded_real_simulator_boundary_video',lossless_input=False,
                  video_sha256=sha(video),execution_sha256=sha(trace/'execution_manifest.json'),
                  format_instructions_sha256=digest(guidance),
                  training_prefix_approval=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        write(a.output,report)
        a.output.chmod(0o600)
    save()
    start = time.monotonic()
    try:
        client = Client(a.output.parent/(a.output.stem+'-requests'),a.endpoint,a.revision,
                        protocol.parse,timeout=240,max_attempts=2,api_key_env=a.api_key_env)
        for vote in range(3):
            description = None
            for kind in ['description','judgment']:
                payload = protocol.payload(spec['task_name'],'visual_success',kind,
                    spec['instruction'],raw,description)
                payload['model'] = a.model
                payload['max_tokens'] = 2048
                payload.pop('thinking_token_budget',None)
                payload['messages'][-1]['content'].append(dict(type='text',text=guidance[kind]))
                record = client.request('vote'+str(vote)+'/'+kind,payload,
                    dict(kind=kind,description=description,task=spec['task_name'],goal='visual_success'))
                if kind == 'description':
                    description = record['prediction']
                else:
                    report['votes'].append(record['prediction']['decision'])
                    save()
        report.update(status='passed',api_verified=True,elapsed_seconds=time.monotonic()-start)
        save()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__ == '__main__':
    main()
