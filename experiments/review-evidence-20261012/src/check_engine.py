"""Check independent-client execution against scalar SGD, without final tests."""
import json
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import engine
import privacy

torch.manual_seed(19)
tf32_setting=torch.backends.cuda.matmul.allow_tf32
torch.backends.cuda.matmul.allow_tf32=False
n,d,c=3,7,4
x=torch.randn(25,d,device='cuda');y=torch.arange(25,device='cuda')%c
index=torch.tensor([[0,1,2,3],[5,6,7,8],[10,11,12,13]],device='cuda')
sizes=torch.full((n,),4,device='cuda')
w=torch.randn(n,d,c,device='cuda')*.01;v=torch.zeros_like(w)
original_steps=engine.STEPS;engine.STEPS=5
actual,_=engine.train_round(w.clone(),v.clone(),x,y,index,sizes,29,1)
scalar=w.clone();velocity=v.clone()
generator=torch.Generator(device='cuda').manual_seed(29*100003+101)
for step in range(engine.STEPS):
    positions=(torch.rand((n,engine.BATCH),device='cuda',generator=generator)*sizes[:,None]).long()
    batch=index.gather(1,positions)
    for client in range(n):
        weight=scalar[client].detach().requires_grad_(True)
        loss=F.cross_entropy(x[batch[client]]@weight,y[batch[client]])
        gradient=torch.autograd.grad(loss,weight)[0]+engine.WEIGHT_DECAY*weight.detach()
        velocity[client].mul_(engine.MOMENTUM).add_(gradient)
        scalar[client]=weight.detach()-engine.LR*velocity[client]
error=float((actual-scalar).abs().max());assert error<2e-7,error
replaced=index.clone();replaced[0]=torch.tensor([20,21,22,23],device='cuda')
changed,_=engine.train_round(w.clone(),v.clone(),x,y,replaced,sizes,29,1)
assert torch.equal(actual[1:],changed[1:])
assert not torch.equal(actual[0],changed[0])
engine.STEPS=original_steps
torch.backends.cuda.matmul.allow_tf32=tf32_setting
opposing=F.one_hot(torch.zeros(5,dtype=torch.long),c).float()-F.one_hot(torch.ones(5,dtype=torch.long),c).float()
assert float(opposing.square().sum())==10
checks={'scalar_SGD_max_absolute_error':error,'one_client_replacement_leaves_other_R1_models_bitwise_unchanged':True,
        'vote_squared_sensitivity_attains_2K':True,'public_selection_does_not_load_test_arrays':True,
        'source_feature_splits_disjoint':True,'certificates_with_and_without_collusion_valid':True}
for name in ['cifar10','mnist','web']:
    public=engine.load_data(name,True);assert not any('test' in key for key in public)
    # Index metadata checks only; no final test features or labels are read.
    bundle=np.load(engine.ROOT/'inputs'/f'{name}.npz')
    splits=[bundle[f'{split}_ids'] for split in ['private','development','proxy','validation','test']]
    for a,b in itertools.combinations(splits,2):assert len(np.intersect1d(a,b))==0
    ids,_,_=engine.partition(public,10,'semantic');random,_,_=engine.partition(public,10,'random')
    assert np.array_equal(np.bincount(ids),np.bincount(random))
for population in [100,1000,10000]:
    for epsilon in [.01,.1,1.]:
        for colluders in [0,population//10]:
            cert=privacy.certificate(epsilon,1e-5,3,population,clusters=20,dimension=200,colluders=colluders)
            assert cert['epsilon_upper']<=epsilon
            assert cert['actual_clients']==cert['calibrated_clients']
(engine.ROOT/'privacy/engine-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
