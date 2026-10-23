"""Matched T=1 per-example release on the independent confirmation cohorts."""
import time
import torch
import engine_base as base
import run_private
from confirm_joint import CONFIRM,CONFIRM_SETTINGS,CONFIRM_SEEDS
from common import ROOT,CLASSES,freeze,save,release_key

def main():
    freeze('CONFIRMATION-POINT-DESIGN.json',dict(settings=CONFIRM_SETTINGS,seeds=CONFIRM_SEEDS,
        steps=180,rounds=1,mechanism='same one-hot per-proxy vote release; no feedback',
        motivation='Match joint public postprocessing across private release interfaces, including regional pooling after the per-example release.',
        postprocessing_only_pooling=True,test_routes_parameters=False))
    for task,n,e in CONFIRM_SETTINGS:
        data=base.load_data(task)
        for seed in CONFIRM_SEEDS:
            key=release_key(task,n,e,seed,180,'per-example',0,'-')
            if (CONFIRM/'results'/(key+'.json')).exists():continue
            w,v,idx,sizes,pop=base.initial_state(data,n,.1,seed)
            x,y,index,sz=[torch.as_tensor(z,device='cuda') for z in [data['x_private'],data['y_private'],idx,sizes]]
            for r in [2,3]:w,v=base.train_round(w,v,x,y,index,sz,seed,r)
            old=run_private.ROOT;run_private.ROOT=CONFIRM
            try:run_private.release(w,data,task,n,e,seed,180,'per-example',0,'-',pop)
            finally:run_private.ROOT=old
    save(ROOT/'CONFIRMATION-POINT-COMPLETE.json',dict(conditions=30,utc=time.time()))

if __name__=='__main__':main()
