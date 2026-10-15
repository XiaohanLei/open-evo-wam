"""One pinned RoboTwin episode, with explicit expert validity and success ledger.

Runs inside the staged simulator tree on its legacy Python environment. Task,
physics, robot and success functions remain those of the pinned upstream source.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--policy_ckpt_path')
    parser.add_argument('--expert-only',action='store_true')
    parser.add_argument('--overrides',nargs=argparse.REMAINDER,default=[])
    a = parser.parse_args()
    import yaml
    import torch
    torch.cuda.init()  # Initialize CUDA before creating the Vulkan renderer.
    options = yaml.safe_load(Path(a.config).read_text()) or {}
    if len(a.overrides)%2:
        raise ValueError('Overrides require key/value pairs')
    for key,value in zip(a.overrides[::2],a.overrides[1::2]):
        options[key.lstrip('-')] = yaml.safe_load(value)
    if int(options.get('test_num',1)) != 1:
        raise ValueError('Fixed episode runner requires exactly one episode')
    module_path = Path(__file__).with_name('_upstream_eval_policy.py')
    spec = importlib.util.spec_from_file_location('robotwin_upstream_eval',module_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    task,config,policy = [options[k] for k in ['task_name','task_config','policy_name']]
    args = yaml.safe_load(Path('task_config',config+'.yml').read_text())
    args.update(task_name=task,task_config=config,ckpt_setting=options['ckpt_setting'],
                policy_name=policy,eval_mode=True)
    camera = upstream.get_camera_config(args['camera']['head_camera_type'])
    args['head_camera_h'],args['head_camera_w'] = camera['h'],camera['w']
    mapping = yaml.safe_load(Path('task_config/_embodiment_config.yml').read_text())
    embodied = args['embodiment']
    if len(embodied) not in [1,3]:
        raise ValueError('Unsupported robot embodiment configuration')
    left = mapping[embodied[0]]['file_path']
    right = left if len(embodied)==1 else mapping[embodied[1]]['file_path']
    args.update(left_robot_file=left,right_robot_file=right,dual_arm_embodied=len(embodied)==1,
        left_embodiment_config=upstream.get_embodiment_config(left),
        right_embodiment_config=upstream.get_embodiment_config(right))
    if len(embodied)==3:
        args['embodiment_dis'] = embodied[2]
    output = Path('eval_result')/task/policy/config/options['ckpt_setting']/'fixed-episode'
    output.mkdir(parents=True,exist_ok=False)
    args['eval_video_save_dir'] = output
    options['left_arm_dim'] = len(args['left_embodiment_config']['arm_joints_name'][0])
    options['right_arm_dim'] = len(args['right_embodiment_config']['arm_joints_name'][1])
    seed = int(options['fixed_seed'])
    instruction = options['instruction_override']
    env = upstream.class_decorator(task)
    env.suc, env.test_num = 0, 0
    ledger = dict(seed=seed,instruction=instruction,attempts=[],final_counted=False,
                  final_success=False,expert_solvability_verified=False)
    ledger_path = output/'physics_retry_ledger.json'
    def save():
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    def snapshot():
        objects = {}
        for name,value in vars(env).items():
            if type(value).__module__.startswith('envs.') and hasattr(value,'get_pose'):
                pose = value.get_pose()
                objects[name] = dict(position=pose.p.tolist(),quaternion=pose.q.tolist())
        return dict(plan_success=bool(env.plan_success),task_success=bool(env.check_success()),objects=objects)
    def observation(name):
        import numpy as np
        obs = env.get_obs()
        data = {camera+'_rgb':value['rgb'] for camera,value in obs['observation'].items() if 'rgb' in value}
        data['q0'] = np.asarray(obs['joint_action']['vector'],dtype=np.float32)
        np.savez_compressed(output/(name+'.npz'),**data)
    save()
    try:
        try:
            env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args)
            observation('expert_initial_observation')
            ledger['expert_initial_state'] = snapshot()
            ledger['expert_moves'] = []
            original_move = env.move
            def recorded_move(*move_args,**move_kwargs):
                result = original_move(*move_args,**move_kwargs)
                ledger['expert_moves'].append(snapshot())
                save()
                return result
            env.move = recorded_move
            env.play_once()
            ledger['expert_plan_success'] = bool(env.plan_success)
            ledger['expert_task_success'] = bool(env.check_success())
            valid = ledger['expert_plan_success'] and ledger['expert_task_success']
            ledger['expert_final_state'] = snapshot()
            env.move = original_move
        except upstream.UnStableError:
            ledger['attempts'].append(dict(status='physics_invalid',reason='UnStableError'))
            save()
            return
        finally:
            env.close_env()
        ledger['expert_solvability_verified'] = valid
        save()
        if a.expert_only:
            ledger['diagnostic_only'] = True
            save()
            return
        if not valid:
            reason = 'expert_planning_failed' if not ledger['expert_plan_success'] else 'expert_task_condition_failed'
            ledger['attempts'].append(dict(status='expert_check_fail',reason=reason))
            save()
            return
        env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args)
        observation('policy_initial_observation')
        native_limit = int(env.step_lim)
        limit = int(options['eval_step_limit_override'])
        if limit > native_limit or limit < 1:
            raise ValueError('Episode step limit exceeds the upstream task limit')
        env.step_lim = limit
        env.set_instruction(instruction=instruction)
        model = upstream.eval_function_decorator(policy,'get_model')(options)
        reset = upstream.eval_function_decorator(policy,'reset_model')
        evaluate = upstream.eval_function_decorator(policy,'eval')
        video = None
        if args.get('eval_video_log'):
            video = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
                '-pixel_format','rgb24','-video_size',str(camera['w'])+'x'+str(camera['h']),
                '-framerate','10','-i','-','-pix_fmt','yuv420p','-vcodec','libx264',
                '-crf','23',str(output/'episode0.mp4')],stdin=subprocess.PIPE)
            env._set_eval_video_ffmpeg(video)
        try:
            reset(model)
            while env.take_action_cnt < env.step_lim and not env.eval_success:
                before = env.take_action_cnt
                evaluate(env,model,env.get_obs())
                if env.take_action_cnt <= before:
                    raise RuntimeError('Policy did not advance simulator actions')
                ledger['steps_executed'] = int(env.take_action_cnt)
                save()
            success = bool(env.eval_success)
            ledger.update(final_counted=True,final_success=success)
            ledger['attempts'].append(dict(status='success' if success else 'model_task_failure',
                                           steps_executed=int(env.take_action_cnt)))
            save()
        finally:
            if video is not None:
                env._del_eval_video_ffmpeg()
                video.wait(timeout=30)
    except Exception as error:
        ledger['infrastructure_error_type'] = type(error).__name__
        save()
        raise
    finally:
        env.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
