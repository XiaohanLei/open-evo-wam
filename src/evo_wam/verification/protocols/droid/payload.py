from pathlib import Path
import json,hashlib,base64,importlib.util
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('factlock_schema',ROOT/'schema.py');schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
def build_payload(task,active_goal,kind,instruction,image_bytes,image_hashes,description=None):
 assert len(image_bytes)==len(image_hashes)==6
 assert [hashlib.sha256(x).hexdigest() for x in image_bytes]==image_hashes
 prompts=json.loads((ROOT/'prompts.json').read_text());goals=json.loads((ROOT/'goals.json').read_text());focus=goals[task][0]['focus']
 if kind=='description':
  text='OBSERVATION FOCUS (not desired outcome): '+focus+'\nFields and allowed values (describe actual CURRENT independently):\n'+json.dumps(schema.SCHEMAS[task]|schema.common_fields())
 else:
  assert description is not None
  text='FROZEN OBSERVATION REPORT:\n'+json.dumps(description)+'\nACTIVE SUBGOAL RULE:\n'+goals[task][active_goal]['rule']+'\nORIGINAL INSTRUCTION (target only, not evidence): '+instruction+'\nRelease required: '+str(goals[task][active_goal]['release_required'])+'\nRequired observed relation fields:\n'+json.dumps(schema.required(task,active_goal))+'\nEXACT OUTPUT TEMPLATE. Fill values; keep exactly these audit keys (bare field names, no facts. prefix). Do not add current_view_observations/initial_objects/current_objects/robot_observation to audit:\n'+json.dumps({'audit':{k:{'status':'supported|contradicted|unresolved','evidence':'short image cue'} for k in (schema.SCHEMAS[task]|schema.common_fields())},'checks':{k:{'value':'yes|no|uncertain','reason':'short reason'} for k in ['goal_satisfied','released','objects_consistent','robot_structure_consistent']}})
 content=[{'type':'text','text':text}]
 for i,raw in enumerate(image_bytes):content.extend([{'type':'text','text':('INITIAL ' if i<3 else 'CURRENT ')+['wrist','external_left','external_right'][i%3]},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode(),'detail':'high'}}])
 return {'model':'qwen38-local','messages':[{'role':'system','content':prompts[kind]},{'role':'user','content':content}],'max_tokens':12288,'thinking_token_budget':4096,'stream':False}
