"""Score a frozen lossless simulator population and emit a limited success-P90 audit."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from datetime import datetime,timezone


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();import numpy as np,h5py
    from evo_wam.io import read,write,sha
    from evo_wam.verification.robotwin import load_predictor,score_prefixes
    a.output.mkdir(parents=True,exist_ok=False);target=a.output/'CALIBRATION_AUDIT.json'
    report=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='limited fresh model-executed population; no historical-threshold equivalence claim',
        frozen_registry_sha256=sha(a.evaluation/'episode-results.json'),
        historical_threshold=0.004184869576192262,historical_threshold_changed=False,episodes=[],excluded=[])
    def save():write(target,report)
    save();start=time.monotonic()
    try:
        results=read(a.evaluation/'episode-results.json')['episodes']
        predict,stats=load_predictor(dict(source_root=str(a.source_root),idm_checkpoint=str(a.checkpoint),tokenizer=str(a.tokenizer)))
        from evo_wam.native_idm import native_module
        data_module = native_module("data")
        canvas_views, render_views = data_module.canvas_views, data_module.render_views
        for index,result in enumerate(results):
            if result['status'] not in ['success','model_task_failure']:
                report['excluded'].append(dict(id=result['episode_key'],status=result['status'],reason=result.get('failure_reason')));save();continue
            ep=a.evaluation/'episodes'/f'{index:06d}';trace=ep/'client_rollout';execution=read(trace/'execution_manifest.json');spec=read(ep/'episode_spec.json')
            n=execution['steps_executed'];actions=np.load(trace/'executed_actions.npy',allow_pickle=False);states=np.load(trace/'real_qpos.npy',allow_pickle=False)
            with h5py.File(trace/'real_frames.h5','r') as h5:
                frames=h5['frames'][:];expected=h5.attrs['rgb_sha256']
            digest=hashlib.sha256(frames.tobytes()).hexdigest()
            if digest!=expected or len(frames)!=n+1 or actions.shape!=(n,14) or states.shape!=(n+1,14):raise ValueError('Lossless trajectory layout/hash differs')
            official=execution['termination']['official_success']
            if official!=(result['status']=='success'):raise ValueError('Official outcome differs')
            transformed=render_views(canvas_views(frames,{}),False,0)
            scored=score_prefixes(spec['episode_key'],[n],transformed,actions,
                [states[i] for i in range(0,n,64)],spec['instruction'],predict,stats,
                'dense','evo-wam-idm-calibration-v1')
            for window in scored['windows']:window['state_source']='real_observation_boundary'
            row=dict(id=spec['episode_key'],task=spec['task_name'],fixed_seed=spec['fixed_seed'],
                outcome='success' if official else 'fail',steps=n,mse=scored['prefixes'][str(n)]['mse'],
                lossless_rgb_sha256=digest,actions_sha256=sha(trace/'executed_actions.npy'),states_sha256=sha(trace/'real_qpos.npy'),
                windows=scored['windows'],unscored_tail_actions=0)
            write(a.output/(str(index)+'-score.json'),row);report['episodes'].append(row);save()
        successes=[r['mse'] for r in report['episodes'] if r['outcome']=='success']
        if not successes:raise ValueError('No successful trajectory for success-P90 calibration')
        threshold=float(np.quantile(successes,.9,method='linear'))
        report.update(status='passed',successes=len(successes),failures=len(report['episodes'])-len(successes),
            planned=len(results),scored=len(report['episodes']),success_p90=threshold,
            quantile_method='numpy.quantile(method="linear")',historical_threshold_equivalence_verified=False,
            historical_threshold_pass_counts={outcome:sum(r['mse']<=report['historical_threshold'] for r in report['episodes'] if r['outcome']==outcome)
                                             for outcome in ['success','fail']},elapsed_seconds=time.monotonic()-start)
        save()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-start);save();raise


if __name__=='__main__':main()
