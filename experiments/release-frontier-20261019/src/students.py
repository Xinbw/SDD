"""Fixed 180-step public students, paired independent random initializations."""
import json
import argparse
import time
import numpy as np
import torch
import engine_base as base
from run_private import decode,metrics
from common import ROOT,PARENT,CLASSES,SETTINGS,SEEDS,release_key,freeze,save,sha

def main(only=None):
    jobs={}
    for dataset,n,epsilon in SETTINGS:
        c=CLASSES[dataset]
        cfg=json.loads((PARENT/'public-selection'/f'{dataset}.json').read_text())
        selected=next(r['selected'] for r in cfg['records'] if r['n']==n and r['epsilon']==epsilon)
        for seed in SEEDS:
            for steps in [60,180]:
                for method,k,est,mode in [('regional',c,'M','center'),('regional',c,'M','coverage'),('per-example',0,'-','center'),
                    ('regional',selected['k'],selected['estimator'],'coverage' if selected['decoder']=='matching' else selected['decoder'])]:
                    source=release_key(dataset,n,epsilon,seed,steps,method,k,est);name=source+'_'+mode
                    jobs[name]=dict(source=source,decoder=mode,dataset=dataset,seed=seed,n=n,epsilon=epsilon,steps=steps)
    freeze('STUDENT-DESIGN.json',dict(jobs=jobs,steps=180,batch=128,lr=.1,initialization_seed_offset=95000,
        initialize_from_private_state=False,additional_private_queries=0,selection_by_test=False))
    outdir=ROOT/'analysis/students';outdir.mkdir(exist_ok=True)
    for dataset in CLASSES:
        if only and dataset!=only:continue
        with np.load(ROOT/'inputs'/f'{dataset}.npz') as b:data={k:b[k] for k in ['x_proxy','x_test','y_test']}
        xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE);c=CLASSES[dataset]
        todo=[(key,j) for key,j in jobs.items() if j['dataset']==dataset and not (outdir/(key+'.json')).exists()]
        random={}
        for seed in SEEDS:
            g=torch.Generator(device=base.DEVICE).manual_seed(seed+95000)
            w=torch.randn((1,128,c),device=base.DEVICE,generator=g)*.01
            picks=[]
            for r in [1,2,3]:
                g=torch.Generator(device=base.DEVICE).manual_seed((seed+95000)*100003+r*101)
                picks.extend([(torch.rand((1,128),device=base.DEVICE,generator=g)*len(xp)).long()[0] for _ in range(60)])
            random[seed]=(w,torch.stack(picks))
        for start in range(0,len(todo),64):
            chunk=todo[start:start+64];targets=[];source_rows=[]
            for name,j in chunk:
                source=ROOT/'results'/(j['source']+'.json');row=json.loads(source.read_text());d=np.load(ROOT/row['diagnostic_file'])
                labels=decode(d['scores'],np.bincount(d['ids'],minlength=len(d['scores'])),j['decoder'])
                targets.append(torch.as_tensor(labels[d['ids']],device=base.DEVICE));source_rows.append(row)
            target=torch.stack(targets);w=torch.cat([random[j['seed']][0] for name,j in chunk]).clone();v=torch.zeros_like(w)
            tapes=torch.stack([random[j['seed']][1] for name,j in chunk])
            for step in range(180):
                ix=tapes[:,step];features=xp[ix];y=target.gather(1,ix)
                residual=torch.bmm(features,w).softmax(-1)
                residual.scatter_add_(-1,y[...,None],-torch.ones_like(y[...,None],dtype=residual.dtype))
                gradient=torch.bmm(features.transpose(1,2),residual)/128+base.WEIGHT_DECAY*w
                v.mul_(base.MOMENTUM).add_(gradient);w.add_(v,alpha=-base.LR)
            xt=torch.as_tensor(data['x_test'],device=base.DEVICE)
            prediction=torch.matmul(xt,w).argmax(-1).cpu().numpy()
            for i,(name,j) in enumerate(chunk):
                row=dict(**j,**metrics(prediction[i],data['y_test'],c),public_steps=180,private_steps_added=0,
                    private_queries_added=0,private_state_used=False,source_sha256=sha(ROOT/'results'/(j['source']+'.json')))
                save(outdir/(name+'.json'),row)
            print(dataset,'students',min(start+64,len(todo)),'/',len(todo),flush=True)
    if len(list(outdir.glob('*.json')))==len(jobs):save(ROOT/'STUDENT-COMPLETE.json',dict(conditions=len(jobs),utc=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CLASSES));main(p.parse_args().dataset)
