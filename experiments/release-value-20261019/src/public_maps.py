"""Record label-free canonical public inference maps for CPU verification."""
import sys,hashlib
import numpy as np
import torch
from study import ROOT,PARENT,freeze,save,sha
sys.path.insert(0,str(PARENT/'src'))
import engine_base as base

def main():
    freeze('PUBLIC-MAPS-DESIGN.json',dict(scope='label-free replay map, recorded after experiments',dataset='cifar10',k=10,
        query_order='development plus test',distance='CUDA float32 cdist with TF32 disabled',labels_used=False))
    base.ROOT=ROOT;torch.backends.cuda.matmul.allow_tf32=False
    with np.load(ROOT/'inputs/cifar10.npz') as b:data={k:b[k] for k in ['x_proxy','x_development','x_test']}
    ids,reps,centres=base.partition(data,10)
    assignment=torch.cdist(torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
    path=ROOT/'analysis/public-evaluation-maps.npz';np.savez_compressed(path,proxy=ids,development=assignment[:len(data['x_development'])],test=assignment[len(data['x_development']):],centres=centres)
    save(ROOT/'PUBLIC-MAPS.json',dict(path=str(path.relative_to(ROOT)),sha256=sha(path),feature_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()},labels_used=False))
    print('Public inference maps recorded.')
if __name__=='__main__':main()
