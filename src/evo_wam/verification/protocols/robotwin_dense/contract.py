"""Frozen parser, action prefix and acceptance logic; no API calls or launch side effects."""
import json
import numpy as np
from .action_filter_frozen import assess
CHECKS=['goal_satisfied', 'released', 'objects_consistent', 'robot_structure_consistent']
VIEWS=['head', 'wrist_left', 'wrist_right']

def prefix_gate(q,q0,t):
 # Preserve frozen thresholds; include the initial transition in reversal statistics too.
 return assess(np.concatenate([q0[None,:],q[:t]],axis=0))

def crops(x):
 assert x.dtype==np.uint8 and x.shape in [(360,640,3),(352,640,3)]
 return [x[:240],x[240:,:320],x[240:,320:]]

def parse(z,kind,description=None):
 try:
  c=z['choices'][0]
  if c.get('finish_reason')=='length':return None,'output_truncated'
  text=c['message'].get('content')or'';p=json.loads(text[text.index('{'):text.rindex('}')+1]);assert isinstance(p,dict)
  if kind=='description':
   assert set(p)=={'initial','current'};ids=set()
   for phase,prefix in [('initial','I'),('current','C')]:
    assert isinstance(p[phase],list) and p[phase]
    categories=set()
    for f in p[phase]:
     assert isinstance(f['id'],str) and f['id'].startswith(prefix) and f['id'] not in ids;ids.add(f['id'])
     assert f['category'] in ['object','relation','robot','occlusion'];categories.add(f['category'])
     assert isinstance(f['text'],str) and f['text'].strip() and isinstance(f['views'],list) and f['views'] and set(f['views'])<=set(VIEWS)
    assert {'object','relation','robot'}<=categories
  else:
   assert description is not None and set(p)=={'fact_checks','checks'} and isinstance(p['fact_checks'],list) and p['fact_checks']
   originals={f['id'] for phase in ['initial','current'] for f in description[phase]};facts={}
   for f in p['fact_checks']:
    assert isinstance(f['id'],str) and f['id'] not in facts;facts[f['id']]=f
    assert f['status'] in ['supported','corrected','unresolved'] and isinstance(f['source_ids'],list) and set(f['source_ids'])<=originals
    assert f['source_ids'] or f['status']=='corrected'
    assert isinstance(f['evidence'],str) and f['evidence'].strip() and isinstance(f['views'],list) and f['views'] and set(f['views'])<={phase+'_'+view for phase in ['initial','current'] for view in VIEWS}
   assert set(p['checks'])==set(CHECKS)
   for k in CHECKS:
    f=p['checks'][k];assert f['value'] in ['yes','no','uncertain'] and isinstance(f['reason'],str) and f['reason'].strip() and isinstance(f['evidence_ids'],list)
    assert set(f['evidence_ids'])<=set(facts)
    if f['value'] in ['yes','no']:assert f['evidence_ids'] and all(facts[x]['status']!='unresolved' for x in f['evidence_ids'])
   values=[p['checks'][k]['value'] for k in CHECKS];p['decision']='reject' if 'no' in values else 'accept' if all(v=='yes' for v in values) else 'uncertain'
  return p,None
 except Exception:return None,'invalid_json_or_evidence_rule'

def reduce_history(history,done=False,error=None):
 accepted=None;missing=False;action=False
 for h in history:
  if h.get('action_filtered'):action=True;break
  p=(h.get('judgment') or {}).get('prediction')
  if p is None:missing=True;continue
  if p['decision']=='accept':accepted=h['step'];break
 if accepted is not None:return {'new':{'accepted_step':accepted,'state':'accepted','valid':True,'earlier_missing':missing}}
 state='source_error' if error else 'action_filtered' if action else 'exhausted' if done else 'running'
 return {'new':{'accepted_step':None,'state':state+('_with_missing' if missing else ''),'valid':bool((done or action) and not missing and not error),'earlier_missing':missing}}
