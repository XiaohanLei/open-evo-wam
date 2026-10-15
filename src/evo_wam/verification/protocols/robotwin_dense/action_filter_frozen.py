import numpy as np
ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
def metrics(q):
 x=q[:,ARM]*180/np.pi;d=np.diff(x,axis=0);rev=np.where(d[1:]*d[:-1]<0,np.minimum(abs(d[1:]),abs(d[:-1])),0);acc=np.diff(d,axis=0)
 # Reversal distance is a per-joint change in direction, in degrees, not a speed limit.
 w=min(14,len(rev));roll=np.stack([np.convolve(rev[:,j],np.ones(w),mode='valid') for j in range(12)],axis=1) if w else np.zeros((1,12));i,j=np.unravel_index(np.argmax(roll),roll.shape);di,dj=np.unravel_index(np.argmax(abs(d)),d.shape)
 return {'steps':len(q),'step_jump_deg':float(abs(d).max()),'step_jump_observation':int(di+2),'step_jump_joint':ARM[dj],'reversal_peak_deg':float(rev.max()),'reversal_1s_sum_deg':float(roll[i,j]),'reversal_window_start_observation':int(i+2),'reversal_joint':ARM[j],'acceleration_peak_deg_per_step2':float(abs(acc).max()),'step_delta_p99_deg':float(np.quantile(abs(d),.99)),'gripper_jump':float(abs(np.diff(q[:,[6,13]],axis=0)).max()),'chunk_boundary_jump_deg':float(abs(d[np.arange(63,len(d),64)]).max()) if len(d)>63 else 0}

POLICY = {'version':'action_quality_v1','fps':15,'jump_deg':30.0,'reversal_1s_sum_deg':25.0,'reversal_peak_deg':10.0,'arm_indices':ARM,'gripper_indices_excluded':[6,13]}
def assess(actions, initial_qpos=None):
 q=np.asarray(actions)
 if q.ndim!=2 or q.shape[1]!=14 or not np.isfinite(q).all():return {'pass':False,'reasons':['invalid_actions']}
 if initial_qpos is not None and (np.asarray(initial_qpos).shape!=(14,) or not np.isfinite(initial_qpos).all()):return {'pass':False,'reasons':['invalid_initial_state']}
 if len(q)<3:return {'pass':False,'reasons':['insufficient_actions']}
 m=metrics(q);reasons=[]
 if initial_qpos is not None:
  m['initial_jump_deg']=float(np.max(np.abs(q[0,ARM]-np.asarray(initial_qpos)[ARM]))*180/np.pi)
 if max(m['step_jump_deg'],m.get('initial_jump_deg',0))>POLICY['jump_deg']:reasons.append('joint_step_jump')
 if m['reversal_1s_sum_deg']>POLICY['reversal_1s_sum_deg']:reasons.append('sustained_direction_reversal')
 if m['reversal_peak_deg']>POLICY['reversal_peak_deg']:reasons.append('large_single_reversal')
 return {'pass':not reasons,'reasons':reasons,'metrics':m}
