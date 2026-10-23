"""ResNet-18 public features with the reference image identities (ENCODER-DESIGN.json)."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18,ResNet18_Weights
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;REPO=EXP.parent
sys.path.insert(0,str(ROOT/'src'));import design;design.main()
REF=EXP/'release-frontier-20261019/inputs/cifar10.npz';WEIGHTS=ROOT/'inputs/resnet18_tv_in1k.safetensors'

def load_safetensors(path):
    """Minimal reader of the safetensors format (8-byte header length, JSON header, raw little-endian tensors)."""
    raw=Path(path).read_bytes();n=int.from_bytes(raw[:8],'little');header=json.loads(raw[8:8+n]);body=raw[8+n:];out={}
    types={'F32':torch.float32,'F16':torch.float16,'I64':torch.int64}
    for k,v in header.items():
        if k=='__metadata__':continue
        a,b=v['data_offsets'];out[k]=torch.frombuffer(bytearray(body[a:b]),dtype=types[v['dtype']]).reshape(v['shape']).clone()
    return out

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def main():
    out=ROOT/'inputs/cifar10.npz'
    if out.exists():print('exists');return
    train=CIFAR10(str(REPO/'data'),train=True,download=False);test=CIFAR10(str(REPO/'data'),train=False,download=False)
    images=np.concatenate([train.data,test.data]);labels=np.array(train.targets+test.targets)
    raw=ROOT/'inputs/cifar10_resnet18_all.npy'
    if raw.exists():feats=np.load(raw)
    else:
        model=resnet18();model.load_state_dict(load_safetensors(WEIGHTS));model=model.cuda().eval();model.fc=torch.nn.Identity()
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];res=[]
        for s in range(0,len(images),256):
            x=torch.from_numpy(images[s:s+256].copy()).cuda().permute(0,3,1,2)
            x=(F.interpolate(x.float()/255,size=(224,224),mode='bicubic',align_corners=False)-mean)/std
            with torch.inference_mode():res.append(model(x).float().cpu().numpy())
        feats=np.concatenate(res);np.save(raw,feats)
        (ROOT/'inputs/resnet18_weights.json').write_text(json.dumps(dict(source='https://hf-mirror.com/timm/resnet18.tv_in1k (torchvision IMAGENET1K_V1 weights repackaged by timm)',
            file=WEIGHTS.name,sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()),indent=2)+'\n')
    with np.load(REF) as r:ref={k:r[k] for k in r.files}
    ids={k:ref[k+'_ids'] for k in ['private','development','proxy','validation','test']}
    for k,v in ids.items():assert np.array_equal(labels[v],ref['y_'+k]),k
    pca=PCA(n_components=128,svd_solver='randomized',random_state=1729).fit(feats[ids['proxy']])
    t=normalize(pca.transform(feats)).astype(np.float32)
    bundle={f'x_{k}':t[v] for k,v in ids.items()}
    bundle.update({f'y_{k}':labels[v] for k,v in ids.items()});bundle.update({f'{k}_ids':v for k,v in ids.items()})
    bundle['x_proxy_weak']=ref['x_proxy_weak']
    np.savez_compressed(out,**bundle)
    (ROOT/'inputs/cifar10_manifest.json').write_text(json.dumps(dict(dataset='cifar10',public_encoder='torchvision ResNet-18 IMAGENET1K_V1, avgpool 512-d, frozen',
        PCA_fit_data='public proxy only',public_PCA_components=128,explained_variance=float(pca.explained_variance_ratio_.sum()),
        identities_equal_reference=True,reference=str(REF.relative_to(EXP)),npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2)+'\n')
    print('prepared',t.shape,float(pca.explained_variance_ratio_.sum()))

if __name__=='__main__':main()
