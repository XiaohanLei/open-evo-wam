import json,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('factlock_schema_contract',Path(__file__).with_name('schema.py'));schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
CHECKS=['goal_satisfied','released','objects_consistent','robot_structure_consistent']
VIEWS={'wrist','external_left','external_right'}
def parse(z,kind,description=None,task=None,goal=1):
 try:
  c=z['choices'][0]
  if c.get('finish_reason')=='length':return None,'output_truncated'
  text=c['message'].get('content')or'';p=json.loads(text[text.index('{'):text.rindex('}')+1]);assert isinstance(p,dict)
  fields=schema.SCHEMAS[task]|schema.common_fields()
  if kind=='description':
   assert set(p)=={'current_view_observations','initial_objects','current_objects','facts','robot_observation'}
   assert set(p['current_view_observations'])==VIEWS
   assert all(isinstance(x,str) and x for x in p['current_view_observations'].values())
   for k in ['initial_objects','current_objects']:
    assert isinstance(p[k],list) and p[k]
    for a in p[k]:
     assert isinstance(a['name'],str) and a['name'] and isinstance(a['features'],str) and a['features'];assert a['views'] and set(a['views'])<=VIEWS
   assert isinstance(p['robot_observation'],str) and p['robot_observation']
   assert isinstance(p['facts'],dict) and set(p['facts'])==set(fields)
   for k,values in fields.items():
    f=p['facts'][k];assert f['value'] in values;assert isinstance(f['evidence'],str) and f['evidence'];assert f['views'] and set(f['views'])<=VIEWS
  else:
   assert description is not None and set(p)=={'audit','checks'}
   assert set(p['audit'])==set(fields)
   for f in p['audit'].values():assert f['status'] in ['supported','contradicted','unresolved'] and isinstance(f['evidence'],str) and f['evidence']
   assert set(p['checks'])==set(CHECKS)
   for f in p['checks'].values():assert f['value'] in ['yes','no','uncertain'] and isinstance(f['reason'],str) and f['reason']
   model_checks=json.loads(json.dumps(p['checks']));forced=[]
   # This enforces agreement on observed propositions, not correctness of their image content.
   required=schema.required(task,goal)
   def cap(check,keys):
    bad=[k for k,v in keys.items() if description['facts'][k]['value']!=v or p['audit'][k]['status']!='supported']
    if bad and p['checks'][check]['value']=='yes':
     p['checks'][check]={'value':'uncertain','reason':'Unresolved observation/target conflict: '+', '.join(bad)};forced.extend(bad)
   cap('goal_satisfied',required)
   if not(task=='operate_drawer_old_prompt' and goal==0):cap('released',{'grasp':'placed_no_active_grasp'})
   cap('objects_consistent',{'object_shape_change':'not_observed'})
   cap('robot_structure_consistent',{'robot_structure_change':'not_observed'})
   values=[p['checks'][k]['value'] for k in CHECKS];p['decision']='reject' if 'no' in values else 'accept' if all(x=='yes' for x in values) else 'uncertain'
   p['model_checks']=model_checks;p['conflicts_preventing_accept']=sorted(set(forced))
  return p,None
 except Exception as e:return None,'invalid_json_or_evidence_rule'
