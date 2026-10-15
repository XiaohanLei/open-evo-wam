"""Exercise six real HTTP calls through the frozen description/judgment parser."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:18881/v1/chat/completions')
    p.add_argument('--model',default='qwen-small')
    p.add_argument('--revision',default='Qwen2.5-VL-3B-Instruct-local')
    p.add_argument('--api-key-env')
    p.add_argument('--format-instructions',type=Path,required=True)
    a = p.parse_args()
    root = a.root.resolve()
    report = dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                  model=a.model,model_revision=a.revision,synthetic_initial_observation=True,
                  real_http_requests=True,parsed_votes=[],api_verified=False,
                  simulator_episode=False,external_api_verified=False)
    target = root/'.private/vlm-api-smoke.json'
    def save():
        target.write_text(json.dumps(report,indent=2)+'\n')
        target.chmod(0o600)
    save()
    start = time.monotonic()
    try:
        import numpy as np
        from evo_wam.verification.client import Client
        from evo_wam.verification.protocol import Protocol
        from evo_wam.verification.stage import png
        protocol = Protocol('robotwin_dense')
        with np.load(root/'run-output/wam-generation-smoke/synthetic-case/imagined_frames.npz',allow_pickle=False) as z:
            current = z['frames'][119]
        initial = np.full((360,640,3),127,dtype=np.uint8)
        raw = [png(x) for x in protocol.contract.crops(initial)+protocol.contract.crops(current)]
        guidance = json.loads(a.format_instructions.read_text())
        from evo_wam.io import digest
        report['format_instructions_sha256'] = digest(guidance)
        client = Client(root/'.private/vlm-api-smoke-requests'/report['format_instructions_sha256'],a.endpoint,a.revision,
                        protocol.parse,timeout=240,max_attempts=2,api_key_env=a.api_key_env)
        for vote in range(3):
            description = None
            for kind in ['description','judgment']:
                payload = protocol.payload('place_empty_cup','visual_success',kind,
                    'place the empty cup',raw,description)
                payload['model'] = a.model
                # Small-model smoke uses a bounded output budget, without changing its rubric.
                payload['max_tokens'] = 2048
                payload.pop('thinking_token_budget',None)
                payload['messages'][-1]['content'].append(dict(type='text',text=guidance[kind]))
                record = client.request('vote'+str(vote)+'/'+kind,payload,
                    dict(kind=kind,description=description,task='place_empty_cup',goal='visual_success'))
                if kind == 'description':
                    description = record['prediction']
                else:
                    report['parsed_votes'].append(record['prediction']['decision'])
                    save()
        report.update(status='passed',api_verified=True,
            accepted_votes=sum(v=='accept' for v in report['parsed_votes']),
            endpoint_step=120,elapsed_seconds=time.monotonic()-start)
        save()
        print(json.dumps(report))
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__ == '__main__':
    main()
