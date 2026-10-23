"""Stronger public class-coverage controls without imputed label certainty."""
import hashlib
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import engine_base as base
from common import ROOT,CLASSES,freeze,save,sha

def score(labels,table):
    c=table.shape[1];total=table.sum();recall=np.array([table[labels==j,j].sum()/table[:,j].sum() for j in range(c)])
    return dict(accuracy=float(table[np.arange(c),labels].sum()/total),balanced_accuracy=float(recall.mean()),
        per_class_recall=recall.tolist(),minimum_recall=float(recall.min()))

def main():
    freeze('PUBLIC-NAMING-AUDIT.json',dict(
        reason='Known taxonomy permits assigning remaining class names without treating inherited empty-region predictions as observed labels.',
        candidates=['observed count matching','observed row frequencies with uniform empty rows, weighted by public region size'],
        no_new_labels=True,no_hyperparameters=True,class_count_free=True,keep_all_original_twelve_baselines=True,
        acquisitions=['random','stratified','geometry'],budgets=[1,3,5,10,20,50],test_routes_choice=False))
    output=[];torch.backends.cuda.matmul.allow_tf32=False
    for task,c in CLASSES.items():
        with np.load(ROOT/'inputs'/f'{task}.npz') as b:data={k:b[k] for k in ['x_proxy','x_development','x_test','y_test']}
        ids,reps,centres=base.partition(data,c);weights=np.bincount(ids,minlength=c)
        points=torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda')
        assign=torch.cdist(points,torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
        dev,test=assign[:len(data['x_development'])],assign[len(data['x_development']):]
        table=np.stack([np.bincount(data['y_test'][test==k],minlength=c) for k in range(c)])
        for path in sorted((ROOT/'public').glob(f'{task}_*_m*.json')):
            row=json.loads(path.read_text());counts=np.zeros((c,c));np.add.at(counts,(dev[row['indices']],row['labels']),1)
            denominator=counts.sum(1,keepdims=True)
            uniform=np.divide(counts,denominator,out=np.full_like(counts,1/c),where=denominator>0)*weights[:,None]
            candidates={}
            for name,value in [('count-matching',counts),('uniform-empty-matching',uniform)]:
                rr,cc=linear_sum_assignment(-value);labels=np.zeros(c,dtype=np.int64);labels[rr]=cc
                candidates[name]=dict(**score(labels,table),regional_labels=labels.tolist())
            output.append(dict(dataset=task,scheme=row['scheme'],draw=row['draw'],q=row['q'],m=row['m'],
                source=path.name,source_sha256=sha(path),baselines=candidates))
    save(ROOT/'analysis/public-naming-audit.json',output)
    for scheme in ['random','stratified']:
        r=[x for x in output if x['dataset']=='cifar10' and x['scheme']==scheme and x['q']==1]
        print('CIFAR',scheme,len(r),{name:np.mean([x['baselines'][name]['accuracy'] for x in r])*100 for name in ['count-matching','uniform-empty-matching']},flush=True)

if __name__=='__main__':main()
