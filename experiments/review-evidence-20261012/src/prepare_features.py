"""Public feature maps and sealed test splits for matched-population experiments."""
from pathlib import Path
import hashlib
import json
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, MNIST
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
torch.set_num_threads(8)


def normalize(x):
    return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def image_features(images, model, dataset):
    path = ROOT/'inputs'/f'{dataset}_dinov2_all.npy'
    if path.exists():
        return np.load(path)
    result=[]
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for start in range(0,len(images),128):
        x=torch.from_numpy(images[start:start+128].copy()).cuda()
        if x.ndim==3:x=x[:,None].repeat(1,3,1,1)
        else:x=x.permute(0,3,1,2)
        x=F.interpolate(x.float()/255,size=(224,224),mode='bicubic',align_corners=False)
        x=(x-mean)/std
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            y=model(x)
        result.append(y.float().cpu().numpy())
        if start%8192==0:print(dataset,'public encoder extraction',start,'/',len(images),flush=True)
    result=np.concatenate(result);np.save(path,result);return result


def image_dataset(name,model):
    if (ROOT/'inputs'/f'{name}.npz').exists():return
    if name=='cifar10':
        train=CIFAR10(str(REPO/'data'),train=True,download=False)
        test=CIFAR10(str(REPO/'data'),train=False,download=False)
        images=np.concatenate([train.data,test.data]);labels=np.array(train.targets+test.targets)
        proxy_local=np.load(REPO/'data/cifar10/proxy_cifar10_src_idx_2000.npy').astype(int)
        proxy=np.asarray(proxy_local)+len(train)
    else:
        train=MNIST(str(REPO/'data'),train=True,download=False)
        test=MNIST(str(REPO/'data'),train=False,download=False)
        images=np.concatenate([train.data.numpy(),test.data.numpy()]);labels=np.concatenate([train.targets.numpy(),test.targets.numpy()])
        proxy=np.load(REPO/'data/mnist/proxy_mnist_src_idx.npy').astype(int)
        assert len(proxy)==2000
    features=image_features(images,model,name)
    pca=PCA(n_components=128,svd_solver='randomized',random_state=1729)
    pca.fit(features[proxy])
    transformed=normalize(pca.transform(features)).astype(np.float32)
    rng=np.random.default_rng(1729)
    eligible=np.setdiff1d(np.arange(len(train)),proxy)
    development=rng.choice(eligible,5000,replace=False)
    private=np.setdiff1d(eligible,development)
    validation=np.setdiff1d(np.arange(len(train)+5000,len(images)),proxy)
    final_test=np.setdiff1d(np.arange(len(train),len(train)+5000),proxy)
    raw=images[proxy].reshape(len(proxy),-1).astype(np.float32)/255
    weak=PCA(n_components=128,svd_solver='randomized',random_state=1729).fit_transform(raw)
    np.savez_compressed(ROOT/'inputs'/f'{name}.npz',
        x_private=transformed[private],y_private=labels[private],
        x_development=transformed[development],y_development=labels[development],
        x_proxy=transformed[proxy],y_proxy=labels[proxy],
        x_validation=transformed[validation],y_validation=labels[validation],
        x_test=transformed[final_test],y_test=labels[final_test],
        x_proxy_weak=normalize(weak).astype(np.float32),
        private_ids=private,development_ids=development,proxy_ids=proxy,
        validation_ids=validation,test_ids=final_test)
    for a,b in [(private,development),(private,proxy),(proxy,final_test),(validation,final_test)]:assert not np.intersect1d(a,b).size
    manifest={'dataset':name,'public_encoder':'Meta DINOv2 ViT-S/14, frozen before client sampling',
              'encoder_source':'https://github.com/facebookresearch/dinov2',
              'encoder_weights_sha256':hashlib.sha256((ROOT/'inputs/dinov2_vits14_public_weights.pt').read_bytes()).hexdigest(),
              'public_PCA_components':128,'PCA_fit_data':'public proxy only',
              'public_development_samples':len(development),'private_pool_samples':len(private),
              'unlabelled_public_proxy_samples':len(proxy),'public_validation_samples':len(validation),
              'sealed_test_samples':len(final_test),'validation_selection_access_to_test':False,
              'weak_partition_features':'public-proxy raw-pixel PCA, 128 components'}
    (ROOT/'inputs'/f'{name}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared',manifest,flush=True)


def strip_headers(text):
    parts=re.split(r'\r?\n\r?\n',text,maxsplit=1)
    body=parts[1] if len(parts)>1 else text
    lines=[line for line in body.splitlines() if not line.lstrip().startswith('>')]
    return '\n'.join(lines).split('\n-- \n')[0]


def author(text):
    match=re.search(r'^From:\s*(.+)$',text,re.MULTILINE|re.IGNORECASE)
    value=match.group(1).strip().lower() if match else hashlib.sha256(text.encode()).hexdigest()
    email=re.search(r'[a-z0-9_.+\-]+@[a-z0-9_.\-]+',value)
    value=email.group(0) if email else value
    return hashlib.sha256(value.encode()).hexdigest()


def web_dataset():
    if (ROOT/'inputs/web.npz').exists():return
    train=fetch_20newsgroups(data_home=str(ROOT/'inputs/newsgroups'),subset='train',remove=(),shuffle=False)
    test=fetch_20newsgroups(data_home=str(ROOT/'inputs/newsgroups'),subset='test',remove=(),shuffle=False)
    texts=train.data+test.data;owners=np.array([author(t) for t in texts]);bodies=[strip_headers(t) for t in texts]
    fine=np.concatenate([train.target,test.target])
    family={'comp':0,'rec':1,'sci':2,'talk':3,'alt':4,'soc':4,'misc':5}
    labels=np.array([family[train.target_names[int(y)].split('.')[0]] for y in fine])
    train_ids=np.arange(len(train.data));rng=np.random.default_rng(1729)
    groups=np.unique(owners[train_ids]);rng.shuffle(groups)
    public=[];count=0
    for group in groups:
        public.append(group);count+=int((owners[train_ids]==group).sum())
        if count>=3000:break
    public=np.array(public);public_ids=train_ids[np.isin(owners[train_ids],public)]
    rng.shuffle(public_ids)
    proxy=public_ids[:1500];development=public_ids[1500:]
    private=train_ids[~np.isin(owners[train_ids],public)]
    # Both public calibration and evaluation use authors absent from the private client cohort.
    test_ids=np.arange(len(train.data),len(texts))
    test_ids=test_ids[~np.isin(owners[test_ids],np.unique(owners[train_ids]))]
    heldout_authors=np.unique(owners[test_ids]);rng.shuffle(heldout_authors)
    validation_authors=heldout_authors[:len(heldout_authors)//2]
    validation=test_ids[np.isin(owners[test_ids],validation_authors)]
    final_test=test_ids[~np.isin(owners[test_ids],validation_authors)]
    vectorizer=TfidfVectorizer(max_features=12000,min_df=2,stop_words='english',sublinear_tf=True)
    vectorizer.fit([bodies[int(i)] for i in np.concatenate([proxy,development])])
    sparse=vectorizer.transform(bodies)
    svd=TruncatedSVD(n_components=128,n_iter=7,random_state=1729)
    svd.fit(sparse[proxy]);x=normalize(svd.transform(sparse)).astype(np.float32)
    assert len(np.unique(owners[private]))>=1000
    assert not np.intersect1d(owners[private],owners[proxy]).size
    assert not np.intersect1d(owners[private],owners[final_test]).size
    assert not np.intersect1d(owners[proxy],owners[final_test]).size
    assert not np.intersect1d(owners[validation],owners[final_test]).size
    np.savez_compressed(ROOT/'inputs/web.npz',x_private=x[private],y_private=labels[private],
        owners_private=owners[private],x_development=x[development],y_development=labels[development],
        owners_development=owners[development],x_proxy=x[proxy],y_proxy=labels[proxy],
        x_validation=x[validation],y_validation=labels[validation],x_test=x[final_test],y_test=labels[final_test],
        x_proxy_weak=normalize(sparse[proxy,:128].toarray()).astype(np.float32),
        private_ids=private,development_ids=development,proxy_ids=proxy,validation_ids=validation,test_ids=final_test)
    manifest={'dataset':'20 Newsgroups / Usenet','labels':['comp','rec','sci','talk','alt/soc','misc'],
              'client_unit':'distinct From-header author, stored as SHA256',
              'available_private_authors':len(np.unique(owners[private])),
              'public_authors':len(public),'public_proxy_samples':len(proxy),
              'public_development_samples':len(development),'validation_samples':len(validation),
              'test_samples':len(final_test),'train_test_authors_disjoint':True,
              'proxy_test_authors_disjoint':True,'validation_test_authors_disjoint':True,
              'text_features':'TF-IDF fitted on public proxy/development only; 128-dimensional SVD fitted on proxy only',
              'headers_and_quoted_lines_removed_from_features':True,
              'private_author_record_counts':{'median':float(np.median(np.unique(owners[private],return_counts=True)[1]))}}
    (ROOT/'inputs/web_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared Web dataset',manifest,flush=True)


if __name__=='__main__':
    repo=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    model=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False)
    model.load_state_dict(torch.load(ROOT/'inputs/dinov2_vits14_public_weights.pt',weights_only=True))
    model=model.cuda().eval()
    image_dataset('cifar10',model);image_dataset('mnist',model)
    del model;torch.cuda.empty_cache();web_dataset()
