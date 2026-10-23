"""A task-label-free compact-head comparator with declared B=1 clipping."""
import math
import time
import numpy as np
import torch
import engine_base as base
import privacy
from run_private import initial,metrics
from common import ROOT,SETTINGS,SEEDS,freeze,save,sha,release_key

def main():
    jobs=[release_key(d,n,e,s,t,'head-fixed',0,'-') for d,n,e in SETTINGS for s in SEEDS for t in [60,180]]
    freeze('FIXED-HEAD-DESIGN.json',dict(clip=1.,choice='fixed numeric default; no task labels, validation or private outcomes choose B',
        jobs=jobs,reason='Place a genuinely zero-public-label head control on the zero-label frontier, alongside the parent-tuned head control.'))
    for dataset in ['cifar10','mnist','web','femnist62']:
        data=base.load_data(dataset)
        for n in sorted({n for d,n,e in SETTINGS if d==dataset}):
            eps=[e for d,nn,e in SETTINGS if d==dataset and nn==n]
            for seed in SEEDS:
                if all((ROOT/'results'/(release_key(dataset,n,e,seed,t,'head-fixed',0,'-')+'.json')).exists() for e in eps for t in [60,180]):continue
                w,v,idx,sizes,pop=initial(data,n,seed);w,v=w.clone(),v.clone()
                x,y,index,sz=[torch.as_tensor(a,device=base.DEVICE) for a in [data['x_private'],data['y_private'],idx,sizes]]
                for steps in [60,180]:
                    if steps==180:
                        for r in [2,3]:w,v=base.train_round(w,v,x,y,index,sz,seed,r)
                    c=w.shape[-1];scale=1<<16;radius=math.ceil(scale+math.sqrt(128*c)/2)
                    norm=w.flatten(1).norm(dim=1).clamp_min(1e-12);clipped=w*(1/norm).clamp(max=1)[:,None,None]
                    q=torch.round(clipped.double()*scale).long();valid=q.double().square().sum((1,2))<=radius**2;q[~valid]=0
                    encoded=q.sum(0).cpu().numpy()
                    for e in eps:
                        key=release_key(dataset,n,e,seed,steps,'head-fixed',0,'-');out=ROOT/'results'/(key+'.json')
                        if out.exists():continue
                        cert=privacy.certificate(e,1e-5,1,n,integer_sensitivity=2*radius,dimension=128*c)
                        cert['integer_report_radius']=radius
                        noise=privacy.noise_sum(cert,128*c,seed*1000003+steps*101+20261019)
                        scores=(encoded+noise.reshape(128,c))/(n*scale);pred=(data['x_test']@scores).argmax(1)
                        diag=ROOT/'diagnostics'/(key+'.npz');np.savez_compressed(diag,scores=scores)
                        row=dict(dataset=dataset,n=n,epsilon=e,seed=seed,steps=steps,method='head-fixed',k=0,estimator='-',
                            releases=1,clip=1.,public_task_labels=0,privacy=cert,population=pop,
                            endpoints={'broadcast':metrics(pred,data['y_test'],c)},logical_report_coordinates=128*c,
                            all_training_before_release_private_only=True,private_model_published=False,
                            diagnostic_file=str(diag.relative_to(ROOT)),diagnostic_sha256=sha(diag))
                        save(out,row);print(key,round(row['endpoints']['broadcast']['accuracy']*100,2),flush=True)
    save(ROOT/'FIXED-HEAD-COMPLETE.json',dict(conditions=len(jobs),utc=time.time()))

if __name__=='__main__':main()
