"""Score actual WAM-generated frames with the selected IDM; no VLM claim."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    output = root / '.private/wam-idm-scoring-smoke.json'
    report = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                  synthetic_initial_observation=True, generated_frames=True,
                  wam='wam-default', idm='idm-default',
                  fixed_threshold=0.004184869576192262, scoring=False,
                  vlm_verified=False, accepted_training_prefix=False,
                  simulator_episode=False, training=False)
    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')
        output.chmod(0o600)
    save()
    start = time.monotonic()
    try:
        import numpy as np
        from evo_wam.verification.robotwin import load_predictor, score_prefixes
        from evo_wam.io import sha
        case = root / 'run-output/wam-generation-smoke/synthetic-case'
        generation = json.loads((case / 'generation_manifest.json').read_text())
        if generation['recurrence_space'] != 'latent' or generation['decoded_rgb_used_for_recurrence'] is not False:
            raise ValueError('WAM generation did not preserve latent recurrence')
        with np.load(case / 'imagined_frames.npz', allow_pickle=False) as z:
            frames = z['frames'].copy()
        actions = np.load(case / 'actions_qpos.npy', allow_pickle=False)
        if actions.shape != (128,14) or len(frames) != 128:
            raise ValueError('WAM output is incomplete')
        states = [np.asarray(c['state'],np.float32) for c in generation['chunks']]
        if len(states) != 2 or not np.array_equal(states[1],actions[63]):
            raise ValueError('WAM boundary state differs from generated terminal action')
        predict, stats = load_predictor(dict(source_root=str(root/'native/idm_dense'),
            tokenizer=str(root/'assets/tokenizer/umt5-xxl'),
            idm_checkpoint=str(root/'weights/idm-default')))
        from evo_wam.native_idm import native_module
        data_module = native_module("data")
        canvas_views, render_views = data_module.canvas_views, data_module.render_views
        if frames.shape[1:] == (352,640,3):
            frames = np.pad(frames,((0,0),(0,8),(0,0),(0,0)),mode='edge')
        initial = np.full((1,360,640,3),127,dtype=np.uint8)
        video = render_views(canvas_views(np.concatenate([initial,frames]),{}),False,0)
        scores = score_prefixes('synthetic-smoke',[64,120,128],video,actions,states,
            'place the empty cup',predict,stats,'dense','evo-wam-robotwin-R1')
        report.update(status='passed',scoring=True,scores=scores,
                      generation_sha256=sha(case/'generation_manifest.json'),
                      generated_actions_sha256=sha(case/'actions_qpos.npy'),
                      elapsed_seconds=time.monotonic()-start)
        save()
        print(json.dumps(report))
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,
                      elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__ == '__main__':
    main()
