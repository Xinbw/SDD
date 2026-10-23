"""Postprocessing of existing protected point matrices; no private queries."""
import json,sys
import numpy as np
import torch
from study import ROOT,PARENT,save,sha,freeze
sys.path.insert(0,str(PARENT/'src'))
import engine_base as base
from run_private import decode,metrics
from common import SETTINGS,SEEDS,CLASSES,release_key
CONFIRM_SETTINGS=[('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1)]

def main():
    freeze('ZERO-LABEL-POOL-AUDIT.json',dict(scope='postprocessing audit after diagnostic outcomes were known; not prospective preregistration',
        sources='110 existing protected per-example matrices at T=1/180',new_private_queries=0,
        pooling='average unprojected scores inside the same public K=C regions; then project and decode',decoders=['center','coverage'],
        public_query_assignment='TF32-disabled concatenated development/test convention'))
    torch.backends.cuda.matmul.allow_tf32=False;base.ROOT=ROOT;out=[]
    for task in CLASSES:
        with np.load(ROOT/'inputs'/f'{task}.npz') as z:data={k:z[k] for k in ['x_proxy','x_development','x_test','y_test']}
        c=CLASSES[task];ids,reps,centres=base.partition(data,c);sizes=np.bincount(ids,minlength=c)
        assignment=torch.cdist(torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()[len(data['x_development']):]
        for phase,settings,seeds,stage in [('paired',SETTINGS,SEEDS,PARENT),('confirmation',CONFIRM_SETTINGS,list(range(8801,8811)),PARENT/'confirmation')]:
            for d,n,e in settings:
                if d!=task:continue
                values={'center':[],'coverage':[]};sources=[];metric='balanced_accuracy' if task=='web' else 'accuracy'
                for seed in seeds:
                    path=stage/'results'/(release_key(d,n,e,seed,180,'per-example',0,'-')+'.json');row=json.loads(path.read_text());scores=np.load(stage/row['diagnostic_file'])['scores']
                    pooled=np.stack([scores[ids==k].mean(0) for k in range(c)])
                    for mode in values:values[mode].append(metrics(decode(pooled,sizes,mode)[assignment],data['y_test'],c)[metric])
                    sources.append(dict(path=str(path.relative_to(ROOT.parents[1])),sha256=sha(path),seed=seed))
                out.append(dict(phase=phase,dataset=d,n=n,epsilon=e,metric=metric,sizes=sizes.tolist(),sources=sources,
                    noise_std_ratio_balanced_approx=np.sqrt(len(ids)/(c*sizes)).tolist(),
                    readouts={mode:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),per_seed=v) for mode,v in values.items()}))
    save(ROOT/'analysis/zero-label-pooled-point.json',out);print('Recomputed zero-label pooled point readouts:',sum(len(r['sources']) for r in out),'sources.')
if __name__=='__main__':main()
