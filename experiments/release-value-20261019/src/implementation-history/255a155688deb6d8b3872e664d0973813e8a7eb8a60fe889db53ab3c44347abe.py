"""Matched T=3 private-only teachers; last versus pooled protected readouts."""
import argparse,json,sys,time
import numpy as np
import torch
from study import ROOT,PARENT,save,freeze,sha
sys.path.insert(0,str(PARENT/'src'))
import engine_base as base
import privacy
from run_private import decode,metrics
from common import release_key

SEEDS=list(range(8501,8511));OLD=PARENT.parent/'release-selector-20261019'

def main():
    freeze('NO-FEEDBACK-DESIGN.json',dict(dataset='cifar10',n=1000,epsilon=.1,delta=1e-5,seeds=SEEDS,private_records=600,
        rounds=3,private_steps_per_round=60,all_private_batches=128,proxy_training=False,regional_rule='K=10/M/center',
        round_noise='same seed tapes as preceding fixed-center T=3 protocol',postprocess=['last release','average three unprojected releases, then centered decoding'],
        role='separate budget concentration and time-varying teacher averaging; not a pure causal estimate of feedback echo',
        first_and_final_private_votes_checked_against_parent_T1=True))
    base.ROOT=ROOT;data=base.load_data('cifar10');ids,reps,centres=base.partition(data,10);sizes=np.bincount(ids,minlength=10)
    torch.backends.cuda.matmul.allow_tf32=False
    assign=torch.cdist(torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()[len(data['x_development']):]
    schedule=json.loads((OLD/'STUDY-SCHEDULE.json').read_text())['design']['roles']
    for seed in SEEDS:
        path=ROOT/'results'/f'no_feedback_s{seed}.json'
        if path.exists():continue
        torch.backends.cuda.matmul.allow_tf32=True
        w,v,idx,sz,pop=base.initial_state(data,1000,.1,seed)
        x,y,index,lens=[torch.as_tensor(a,device='cuda') for a in [data['x_private'],data['y_private'],idx,sz]]
        cert=privacy.certificate(.1,1e-5,3,1000,clusters=10,dimension=100)
        scores=[];clean=[]
        for r in [1,2,3]:
            if r>1:w,v=base.train_round(w,v,x,y,index,lens,seed,r)
            counts=base.vote_counts(w,torch.as_tensor(data['x_proxy'],device='cuda'),ids);clean.append(counts.cpu().numpy())
            if r in [1,3]:
                key=release_key('cifar10',1000,.1,seed,60 if r==1 else 180,'regional',10,'M');reference=json.loads((PARENT/'results'/(key+'.json')).read_text())
                assert reference['population']==pop and np.array_equal(np.load(PARENT/reference['diagnostic_file'])['counts'],clean[-1])
            scores.append(base.noisy_votes(counts,cert,seed*1000003+r*101).cpu().numpy())
        role=next(z for z in schedule if z['role']=='fixed-center' and z['condition']==dict(dataset='cifar10',n=1000,epsilon=.1) and z['job'].split('_')[3]==str(seed))
        source=OLD/'results'/(role['job']+'.json');original=json.loads(source.read_text());diagold=np.load(OLD/original['diagnostic_file'])
        assert np.array_equal(scores[0],diagold['r1_scores'])
        feedback=[diagold[f'r{r}_scores'] for r in [1,2,3]]
        def evaluate(a):return metrics(decode(a,sizes,'center')[assign],data['y_test'],10)
        diag=ROOT/'diagnostics'/f'no_feedback_s{seed}.npz';np.savez_compressed(diag,**{f'r{r+1}_counts':a for r,a in enumerate(clean)},**{f'r{r+1}_scores':a for r,a in enumerate(scores)})
        save(path,dict(seed=seed,population=pop,privacy=cert,protected_releases=3,steps=180,feedback=False,
            last=evaluate(scores[-1]),pooled=evaluate(np.mean(scores,axis=0)),feedback_last=evaluate(feedback[-1]),feedback_pooled=evaluate(np.mean(feedback,axis=0)),
            parent_feedback_source=role['job'],parent_feedback_sha256=sha(source),first_release_identical=True,private_initial_final_votes_identical_to_T1=True,
            pooled_noise_std=cert['residual_std']/np.sqrt(3),diagnostic_file=str(diag.relative_to(ROOT)),diagnostic_sha256=sha(diag)))
        print('No feedback completed',seed,flush=True)
    save(ROOT/'NO-FEEDBACK-COMPLETE.json',dict(cases=10,utc=time.time()))

if __name__=='__main__':main()
