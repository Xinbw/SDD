"""Shared loaders for the allocation feasibility study (CPU re-releases of stored clean statistics)."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
VALUE=ROOT.parent/'release-value-20261019'
FRONTIER=ROOT.parent/'release-frontier-20261019'
sys.path.insert(0,str(FRONTIER/'src'));sys.path.insert(0,str(VALUE/'src'))
import privacy
import private_validation as pv
from linear_bands import bands
from run_private import metrics

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def load_public():
    record=json.loads((VALUE/'PUBLIC-MAPS.json').read_text());maps=np.load(VALUE/record['path'])
    assert sha(VALUE/record['path'])==record['sha256']
    with np.load(VALUE/'inputs/cifar10.npz') as z:data={k:z[k] for k in ['x_private','y_private','y_test']}
    samples=[json.loads((VALUE/'public'/f'labels_{d}.json').read_text()) for d in range(100)]
    return maps,data,samples

def private_regions(x_private,centres):
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    return torch.cdist(torch.as_tensor(x_private,device=dev),torch.as_tensor(centres,device=dev)).argmin(1).cpu().numpy()

def clean_counts(seed):
    z=np.load(VALUE/'diagnostics/naming'/f's{seed}_e0.1.npz');counts=z['counts']
    for e in ['1','3']:assert np.array_equal(np.load(VALUE/'diagnostics/naming'/f's{seed}_e{e}.npz')['counts'],counts)
    return counts
