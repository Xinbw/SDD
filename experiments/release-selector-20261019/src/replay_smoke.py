"""Reconstruct two frozen actual-cohort cases without modifying evidence."""
import json
import learning
from common import ROOT,key,save

plan=json.loads((ROOT/'STUDY-SCHEDULE.json').read_text())['design']
names=['cifar10_10000_0.1_8501_SDD_10_matching_R_64_64',
       'femnist62_2000_1.0_8501_SDD_62_center_R_64_64']
scratch=ROOT/'reproduction-output/smoke'
for name in ['results','diagnostics']:(scratch/name).mkdir(parents=True,exist_ok=True)
learning.ROOT=scratch
reports=[]
for name in names:
    j=next(j for j in plan['jobs'] if key(j)==name)
    expected=json.loads((ROOT/'results'/(name+'.json')).read_text())
    actual=learning.run(j)
    error=max(abs(expected[field]-actual[field]) for field in ['accuracy','balanced_accuracy'])
    error=max(error,*(abs(expected['global_endpoint'][field]-actual['global_endpoint'][field]) for field in ['accuracy','balanced_accuracy']))
    assert error<2e-6,(name,error)
    assert actual['population']==expected['population']
    reports.append(dict(case=name,maximum_metric_error=error,private_and_public_endpoints_reproduced=True))
    print('PASS',name,'maximum metric difference',error,flush=True)
save(ROOT/'logs/smoke-replay.json',reports)
