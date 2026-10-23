"""Same-capacity public students for both protected vote interfaces.

The regional nearest-centroid endpoint is also retained. Neither readout is
chosen with test labels. These are extra server steps, with no private query.
"""
import json
import numpy as np
import torch
import engine_base as base
from common import ROOT,key,freeze,save,sha


def main():
    plan=json.loads((ROOT/'STUDY-SCHEDULE.json').read_text())['design']
    jobs=[j for j in plan['jobs'] if j['method'] in ['SDD','DP-FedCT']]
    freeze(ROOT/'PUBLIC-READOUT-DESIGN-FROZEN.json',dict(jobs=[key(j) for j in jobs],
        last_charged_release=3,server_steps=180,lr=.1,batch=128,initialization='independent public random head; no private state',
        all_readouts_reported=True,test_selects_readout=False,extra_private_queries=0,extra_privacy_cost=0))
    outdir=ROOT/'analysis/public-readouts';outdir.mkdir(exist_ok=True)
    for j in jobs:
        source=ROOT/'results'/(key(j)+'.json');output=outdir/(key(j)+'.json')
        if not source.exists() or output.exists():continue
        row=json.loads(source.read_text());diag=np.load(ROOT/row['diagnostic_file'])
        data=base.load_data(j['dataset']);labels=diag['r3_labels'][diag['ids']]
        xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE);yp=torch.as_tensor(labels,device=base.DEVICE)
        c=diag['truth_counts'].shape[1]
        g=torch.Generator(device=base.DEVICE).manual_seed(j['seed']+95000)
        w=torch.randn((1,xp.shape[1],c),device=base.DEVICE,generator=g)*.01;v=torch.zeros_like(w)
        idx=torch.arange(len(xp),device=base.DEVICE)[None];sizes=torch.tensor([len(xp)],device=base.DEVICE)
        for r in range(1,4):w,v=base.train_round(w,v,xp,yp,idx,sizes,j['seed']+95000,r)
        a,b=base.evaluate(w,torch.as_tensor(data['x_test'],device=base.DEVICE),torch.as_tensor(data['y_test'],device=base.DEVICE))
        result=dict(job=key(j),dataset=j['dataset'],n=j['n'],epsilon=j['epsilon'],seed=j['seed'],method=j['method'],
            accuracy=a,balanced_accuracy=b,per_class_recall=base.evaluate.last_recall,server_steps=180,
            private_steps_added=0,private_queries_added=0,private_model_state_used=False,test_selects_readout=False,
            source_sha256=sha(source),diagnostic_sha256=sha(ROOT/row['diagnostic_file']))
        save(output,result)
        print('PUBLIC STUDENT',key(j),round(a*100,4),flush=True)


if __name__=='__main__':main()
