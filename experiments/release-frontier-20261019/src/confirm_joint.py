"""Independent public-label and actual-cohort confirmation of joint naming."""
import argparse
import json
import math
import time
import hashlib
import numpy as np
import torch
import engine_base as base
import privacy
import run_private
from common import ROOT,CLASSES,freeze,save,sha,release_key

CONFIRM=ROOT/'confirmation'
CONFIRM_SETTINGS=[('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1)]
CONFIRM_SEEDS=list(range(8801,8811))

def freeze_design():
    freeze('JOINT-CONFIRMATION-DESIGN.json',dict(settings=CONFIRM_SETTINGS,seeds=CONFIRM_SEEDS,
        public_draws=100,budgets=[1,3,5,10,20,50],label_sampling_seed=11000001,
        private_steps=180,release_rounds=1,regional='K=C,M',head_clip=1.,
        independent_label_and_cohort_noise_tapes=True,method_frozen_before_confirmatory_outcomes=True,
        public_labels_for_all_selection_in_m=True,private_local_utility_never_selects=True,
        purpose='Independent replication after strengthening public matching controls; all development outcomes retained.'))

def samples():
    folder=CONFIRM/'samples';folder.mkdir(parents=True,exist_ok=True)
    for task in ['cifar10','mnist']:
        with np.load(ROOT/'inputs'/f'{task}.npz') as b:y=b['y_development']
        c=CLASSES[task]
        for draw in range(100):
            permutation=np.random.default_rng(11000001+list(CLASSES).index(task)*10000+draw).permutation(len(y))
            for q in [1,3,5,10,20,50]:
                idx=permutation[:q*c];labels=y[idx];file=folder/f'{task}_random_{draw}_m{q*c}.json'
                if file.exists():continue
                save(file,dict(dataset=task,scheme='random',draw=draw,q=q,m=len(idx),indices=idx.tolist(),labels=labels.tolist(),
                    missing_classes=int((np.bincount(labels,minlength=c)==0).sum()),
                    sample_sha256=hashlib.sha256(idx.tobytes()+labels.tobytes()).hexdigest(),independent_confirmation=True))

def private():
    for d in ['results','diagnostics']:(CONFIRM/d).mkdir(parents=True,exist_ok=True)
    for task,n,e in CONFIRM_SETTINGS:
        data=base.load_data(task)
        for seed in CONFIRM_SEEDS:
            key=release_key(task,n,e,seed,180,'regional',CLASSES[task],'M')
            hkey=release_key(task,n,e,seed,180,'head-fixed',0,'-')
            if (CONFIRM/'results'/(key+'.json')).exists() and (CONFIRM/'results'/(hkey+'.json')).exists():continue
            w,v,idx,sizes,pop=base.initial_state(data,n,.1,seed)
            x,y,index,sz=[torch.as_tensor(a,device='cuda') for a in [data['x_private'],data['y_private'],idx,sizes]]
            for r in [2,3]:w,v=base.train_round(w,v,x,y,index,sz,seed,r)
            acc,balanced=base.evaluate(w,torch.as_tensor(data['x_test'],device='cuda'),torch.as_tensor(data['y_test'],device='cuda'))
            local=dict(accuracy=acc,balanced_accuracy=balanced,per_class_recall=base.evaluate.last_recall)
            old=run_private.ROOT;run_private.ROOT=CONFIRM
            try:run_private.release(w,data,task,n,e,seed,180,'regional',CLASSES[task],'M',pop)
            finally:run_private.ROOT=old
            rp=CONFIRM/'results'/(key+'.json');row=json.loads(rp.read_text());row['private_local_only_endpoint']=local;save(rp,row)
            c=CLASSES[task];scale=1<<16;radius=math.ceil(scale+math.sqrt(128*c)/2)
            norm=w.flatten(1).norm(2,dim=1).clamp_min(1e-12);clipped=w*(1/norm).clamp(max=1)[:,None,None]
            quant=torch.round(clipped.double()*scale).long();valid=quant.double().square().sum((1,2))<=radius**2;quant[~valid]=0
            cert=privacy.certificate(e,1e-5,1,n,integer_sensitivity=2*radius,dimension=128*c);cert['integer_report_radius']=radius
            noise=privacy.noise_sum(cert,128*c,seed*1000003+180*101+20261019)
            score=(quant.sum(0).cpu().numpy()+noise.reshape(128,c))/(n*scale)
            diag=CONFIRM/'diagnostics'/(hkey+'.npz');np.savez_compressed(diag,scores=score)
            save(CONFIRM/'results'/(hkey+'.json'),dict(dataset=task,n=n,epsilon=e,seed=seed,steps=180,method='head-fixed',k=0,estimator='-',
                releases=1,clip=1.,public_task_labels=0,privacy=cert,population=pop,private_local_only_endpoint=local,
                endpoints={'broadcast':run_private.metrics((data['x_test']@score).argmax(1),data['y_test'],c)},
                diagnostic_file=str(diag.relative_to(CONFIRM)),diagnostic_sha256=sha(diag),private_model_published=False))
    save(ROOT/'CONFIRMATION-PRIVATE-COMPLETE.json',dict(conditions=60,actual_cohorts=30,utc=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');a=p.parse_args();freeze_design()
    if not a.freeze_only:samples();private()
