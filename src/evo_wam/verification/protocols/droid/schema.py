SCHEMAS={
 'stack_bowls_v2':{
  'pink_direct_support':['blue_bowl','white_bowl','table','gripper','other','unknown'],
  'blue_direct_support':['pink_bowl','white_bowl','table','gripper','other','unknown'],
  'white_direct_support':['pink_bowl','blue_bowl','table','gripper','other','unknown']},
 'two_ducks_droidstyle_g5':{
  'pink_duck_container':['pink_bowl','blue_bowl','none','other','unknown'],
  'blue_duck_container':['pink_bowl','blue_bowl','none','other','unknown'],
  'pink_bowl_contents':['pink_duck','blue_duck','both_ducks','empty','other','unknown'],
  'blue_bowl_contents':['pink_duck','blue_duck','both_ducks','empty','other','unknown']},
 'operate_drawer_old_prompt':{
  'drawer_state':['closed','extended','unknown'],
  'handle_contact':['engaged','separate','unknown'],
  'bread_location':['drawer_interior','drawer_rim','plate','gripper','table','other','unknown'],
  'bread_support':['drawer','gripper','plate','table','other','unknown']}}

def common_fields():
 return {'grasp':['held','placed_no_active_grasp','unknown'],
 'object_shape_change':['not_observed','unexplained_change','unknown'],
 'robot_structure_change':['not_observed','clear_change','unknown']}

def required(task,goal):
 out={}
 if task=='stack_bowls_v2':
  out={'pink_direct_support':'blue_bowl'}
  if goal==1:out['blue_direct_support']='white_bowl';out['white_direct_support']='table'
 elif task=='two_ducks_droidstyle_g5':
  out={'pink_duck_container':'pink_bowl','pink_bowl_contents':'pink_duck'}
  if goal==1:out.update(blue_duck_container='blue_bowl',blue_bowl_contents='blue_duck')
 elif task=='operate_drawer_old_prompt':
  out={'drawer_state':'extended','handle_contact':'engaged'} if goal==0 else {'bread_location':'drawer_interior','bread_support':'drawer'}
 return out
