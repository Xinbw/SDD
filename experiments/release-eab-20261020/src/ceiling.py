"""Partition ceilings: the best accuracy any regional naming can reach on a fixed public partition.

Offline research diagnostic computed with test labels; it never enters a selection or release
decision. Assignments are recomputed with the shared public pipeline map and must match the
recorded geometry hashes exactly.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];FRONTIER=ROOT.parent/'release-frontier-20261019'
sys.path.insert(0,str(FRONTIER/'src'))
import torch
import engine_base as base
from common import CLASSES
import privacy

TASKS=[('cifar10',10),('mnist',10),('web',6),('femnist62',31),('femnist62',62),('femnist62',124),('femnist62',248),('femnist62',500)]
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    recorded=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['geometries'];out=[]
    for task,k in TASKS:
        with np.load(FRONTIER/'inputs'/f'{task}.npz') as b:data={x:b[x] for x in ['x_proxy','x_development','x_test','y_test']}
        C=CLASSES[task];_,_,centres=base.partition(data,k)
        points=torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda')
        a=torch.cdist(points,torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()[len(data['x_development']):]
        assert hashlib.sha256(a.tobytes()).hexdigest()==recorded[f'{task}/K{k}'],(task,k)
        y=data['y_test'];M=np.zeros((k,C));np.add.at(M,(a,y),1)
        balanced=task=='web';W=M/np.maximum(M.sum(0,keepdims=True),1) if balanced else M/len(y)
        purity=float(W.max(1).sum()/(C if balanced else 1)) if balanced else float(W.max(1).sum())
        if balanced:purity=float(W.max(1).sum()/C)
        r,c=linear_sum_assignment(-W) if k>=C else linear_sum_assignment(-W.T)
        matching=float(W[r,c].sum()/(C if balanced else 1)) if k>=C else float(W.T[r,c].sum()/(C if balanced else 1))
        out.append(dict(task=task,K=k,classes=C,metric='balanced_accuracy' if balanced else 'accuracy',test_examples=int(len(y)),
            many_to_one_ceiling=purity,one_to_one_ceiling=matching if k==C else None,geometry_sha256=recorded[f'{task}/K{k}'],uses_test_labels_offline_only=True))
        print(out[-1]['task'],k,round(100*purity,2),out[-1]['one_to_one_ceiling'] and round(100*matching,2),flush=True)
    (ROOT/'analysis/partition-ceilings.json').write_text(json.dumps(dict(records=out,scope='offline diagnostic; test labels never route decisions'),indent=2)+'\n')
if __name__=='__main__':main()
