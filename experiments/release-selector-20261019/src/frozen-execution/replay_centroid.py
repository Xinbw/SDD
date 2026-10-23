"""Postprocess existing protected matrices; no new private queries or budget."""
import json
import numpy as np
import torch

import engine_base as base
from common import ROOT,save,sha


def main():
    parent=ROOT.parent/'release-granularity-20261019'
    jobs=[dict(source=str(p.relative_to(ROOT.parent)),sha256=sha(p)) for p in sorted((parent/'results').glob('*.json'))
        if json.loads(p.read_text())['method']=='SDD']
    from common import freeze
    freeze(ROOT/'CENTROID-REPLAY-FROZEN.json',dict(jobs=jobs,rule='nearest public semantic centroid; each of the three already-charged releases',
        test_labels_select_classifier=False,private_queries=0,additional_privacy_cost=0))
    output=[];cache={}
    for job in jobs:
        path=ROOT.parent/job['source'];r=json.loads(path.read_text());dataset=r['dataset'];k=r['k']
        data=base.load_data(dataset)
        if (dataset,k) not in cache:
            _,_,centres=base.partition(data,k)
            assignment=torch.cdist(torch.as_tensor(data['x_test'],device=base.DEVICE),
                torch.as_tensor(centres,device=base.DEVICE)).argmin(-1).cpu().numpy()
            cache[(dataset,k)]=assignment
        diagnostic=np.load(parent/r['diagnostic_file']);assignment=cache[(dataset,k)]
        for release in [1,2,3]:
            decoded=diagnostic[f'r{release}_decoded'];labels=decoded.argmax(1) if decoded.ndim==2 else decoded
            pred=labels[assignment]
            acc,balanced=base.score_labels(pred,data['y_test'],diagnostic['truth_counts'].shape[1])
            output.append(dict(dataset=dataset,n=r['n'],epsilon=r['epsilon'],seed=r['seed'],k=k,decoder=r['decoder'],
                group=r['group'],release=release,accuracy=acc,balanced_accuracy=balanced,
                source=job['source'],source_sha256=job['sha256'],additional_private_queries=0,
                endpoint='public nearest-centroid classifier; existing protected regional labels'))
    save(ROOT/'analysis/centroid-replays.json',output)
    print('Reused',len(jobs),'protected trajectories;',len(output),'centroid endpoints.',flush=True)


if __name__=='__main__':main()
