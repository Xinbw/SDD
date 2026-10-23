"""Public-only and Local+Public references for the natural writer task."""
import json
import numpy as np
import torch
import engine_base as base
from common import ROOT,freeze,save


def fit(data,seed,lr,steps):
    x=torch.as_tensor(data['x_development'],device=base.DEVICE);y=torch.as_tensor(data['y_development'],device=base.DEVICE)
    g=torch.Generator(device=base.DEVICE).manual_seed(seed)
    w=torch.randn((1,x.shape[1],62),device=base.DEVICE,generator=g)*.01;v=torch.zeros_like(w)
    idx=torch.arange(len(x),device=base.DEVICE)[None];sizes=torch.tensor([len(x)],device=base.DEVICE)
    original=base.LR;base.LR=lr
    try:
        for r in range(1,steps//60+1):w,v=base.train_round(w,v,x,y,idx,sizes,seed,r)
    finally:base.LR=original
    return w


def evaluate(w,data,split):
    a,b=base.evaluate(w,torch.as_tensor(data['x_'+split],device=base.DEVICE),torch.as_tensor(data['y_'+split],device=base.DEVICE))
    return dict(accuracy=a,balanced_accuracy=b,per_class_recall=base.evaluate.last_recall)


def select():
    path=ROOT/'public-selection/femnist62_public_model.json'
    if path.exists():return json.loads(path.read_text())
    freeze(ROOT/'PUBLIC-MODEL-DESIGN-FROZEN.json',dict(dataset='femnist62',lrs=[.03,.1,.3],steps=[60,180,600],
        selection_seeds=list(range(94001,94006)),criterion='public validation accuracy',private_or_test_access=False))
    with np.load(ROOT/'inputs/femnist62.npz',allow_pickle=False) as b:
        data={k:b[k] for k in ['x_development','y_development','x_validation','y_validation']}
    options=[]
    for lr in [.03,.1,.3]:
        for steps in [60,180,600]:
            values=[evaluate(fit(data,s,lr,steps),data,'validation') for s in range(94001,94006)]
            options.append(dict(lr=lr,steps=steps,score=float(np.mean([x['accuracy'] for x in values])),runs=values))
    selected=max(options,key=lambda r:(r['score'],-r['steps'],-r['lr']))
    matched=max([x for x in options if x['steps']==180],key=lambda r:(r['score'],-r['lr']))
    out=dict(candidates=options,selected=selected,matched_180=matched,configuration_frozen=True,
        development_labels=5000,validation_labels=len(data['y_validation']),private_data_access=False,test_access=False)
    save(path,out);print('PUBLIC MODEL FROZEN',selected['lr'],selected['steps'],selected['score'],flush=True)
    return out


if __name__=='__main__':select()
