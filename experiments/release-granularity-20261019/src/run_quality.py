"""Independent exact-DG validation of the Gaussian regional-label approximation.

This study estimates first-release target accuracy, not final client-model
accuracy. Private count margins are retained as offline benchmark diagnostics.
"""
import hashlib
import json
import time

import numpy as np
import torch

import engine_base as base
import privacy
from experiment_engine import ROOT, save
from public_design import select
from quality_model import labels, predict, quality, raw_lower_bound


def exact_trials(q, truth, row_weights, cert, repeats, seed):
    n=cert['actual_clients'];d=q.size
    block=max(1,min(repeats,5000000//(n*d)))
    values={mode:[] for mode in ['raw','center']}
    flips={mode:[] for mode in values}
    for first in range(0,repeats,block):
        count=min(block,repeats-first)
        shares=privacy.draw((n,count,d),cert['sigma_share_integer'],seed=seed+first)
        noise=shares.sum(0,dtype=np.int64).reshape((count,)+q.shape)/(n*cert['quantization_scale'])
        for mode in values:
            decoded=labels(q[None]+noise,row_weights,mode)
            values[mode].extend(quality(decoded,truth).tolist())
            flips[mode].extend((decoded!=labels(q,row_weights,mode)[None]).mean(1).tolist())
    return {mode:dict(exact_mean=float(np.mean(v)),exact_sem=float(np.std(v,ddof=1)/np.sqrt(repeats)),
        exact_repeats=repeats,exact_flip_fraction=float(np.mean(flips[mode])),values=v) for mode,v in values.items()}


def main():
    design=select('cifar10',[100,1000,10000],[.025,.1,1.],[5,10,20,50,100])
    plan=dict(dataset='cifar10',cohorts=[100,1000,10000],seeds=[8301,8302,8303],
        clusters=[5,10,20,50,100],budgets=[.01,.025,.05,.1,.25,.5,1.],
        gaussian_repeats=4096,exact_repeats=128,
        exact_grid=dict(seed=8301,cohorts=[100,1000,10000],clusters=[5,10,20],budgets=[.025,.1,1.]),
        target='first-release constant regional proxy-label accuracy; not downstream model accuracy',
        private_clean_counts_are_offline_only=True,public_selector_uses_private_counts=False,
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')})
    schedule=ROOT/'QUALITY-SCHEDULE.json'
    if schedule.exists():
        assert {k:v for k,v in json.loads(schedule.read_text()).items() if k not in ['created_utc','source_sha256']}=={
            k:v for k,v in plan.items() if k not in ['created_utc','source_sha256']}
    else:save(schedule,plan)
    data=base.load_data('cifar10')
    xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE)
    for n in plan['cohorts']:
        for seed in plan['seeds']:
            weights,_,_,_,population=base.initial_state(data,n,.1,seed)
            for k in plan['clusters']:
                ids,_,_=base.partition(data,k)
                counts=base.vote_counts(weights,xp,ids).cpu().numpy()
                q=counts.astype(np.float64)/n
                truth=np.stack([np.bincount(data['y_proxy'][ids==i],minlength=10) for i in range(k)])
                row_weights=np.bincount(ids,minlength=k)
                source=ROOT/'predictions'/f'counts_n{n}_s{seed}_k{k}.npz'
                if not source.exists():np.savez_compressed(source,counts=counts,truth_counts=truth,region_weights=row_weights,ids=ids)
                for epsilon in plan['budgets']:
                    output=ROOT/'predictions'/f'quality_n{n}_s{seed}_k{k}_e{epsilon}.json'
                    if output.exists():continue
                    cert=privacy.certificate(epsilon,1e-5,3,n,clusters=k,dimension=k*10)
                    modes={mode:predict(q,truth,row_weights,cert['residual_std'],mode,repeats=4096,seed=91001)
                        for mode in ['raw','center']}
                    exact=None
                    if seed==8301 and k in [5,10,20] and epsilon in [.025,.1,1.]:
                        exact=exact_trials(q,truth,row_weights,cert,128,930000+n*100+k*7+int(epsilon*1000))
                    # An independent single exact transcript accompanies every cell.
                    noise=privacy.noise_sum(cert,k*10,seed*1000003+int(epsilon*10000)+k).reshape(q.shape)/(n*(1<<16))
                    for mode in modes:
                        modes[mode]['observed_single_exact_accuracy']=float(quality(labels(q+noise,row_weights,mode),truth))
                        if exact is not None:modes[mode].update(exact[mode])
                    result=dict(dataset='cifar10',n=n,seed=seed,k=k,epsilon=epsilon,privacy=cert,modes=modes,
                        raw_subgaussian_lower_bound=raw_lower_bound(q,truth,cert['residual_std']),
                        population=population,counts_file=str(source.relative_to(ROOT)),
                        counts_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        prediction_is_offline_and_conditional_on_private_counts=True)
                    save(output,result)
                print('QUALITY',n,seed,k,flush=True)
    save(ROOT/'QUALITY-COMPLETE.json',dict(cells=315,exact_multitrial_cells=27,utc=time.time()))


if __name__=='__main__':main()
