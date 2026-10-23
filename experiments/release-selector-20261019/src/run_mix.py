"""Execute the already-frozen mixing subgroup independently of the main queue."""
import json
from common import ROOT,key,save
from learning import run

plan=json.loads((ROOT/'STUDY-SCHEDULE.json').read_text())['design']
keys={r['job'] for r in plan['roles'] if r['role']=='training-mix'}
jobs=[j for j in plan['jobs'] if key(j) in keys]
assert len(jobs)==60
for i,j in enumerate(jobs):
    if not (ROOT/'results'/(key(j)+'.json')).exists():run(j)
    save(ROOT/'MIX-STATUS.json',dict(completed=i+1,total=len(jobs)))
save(ROOT/'MIX-COMPLETE.json',dict(completed=len(jobs)))
