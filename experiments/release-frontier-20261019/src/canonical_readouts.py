"""Re-evaluate released names on one shared public distance assignment.

Private training uses the established TF32 policy. All public regional
comparisons use the public pipeline's TF32-disabled, concatenated-query map.
Raw releases/native diagnostic metrics remain byte-for-byte unchanged.
"""
import hashlib
import json
import numpy as np
import torch
import engine_base as base
from common import ROOT,PARENT,CLASSES,save,sha
from run_private import metrics

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    all_rows=[]
    for stage,method in [(PARENT,'SDD'),(ROOT,'regional'),(ROOT/'confirmation','regional')]:
        for path in (stage/'results').glob('*.json'):
            row=json.loads(path.read_text())
            if row['method']==method:all_rows.append((stage,path,row))
    records={};geometries={};changes=[]
    for task in CLASSES:
        with np.load(ROOT/'inputs'/f'{task}.npz') as b:data={k:b[k] for k in ['x_proxy','x_development','x_test','y_test']}
        points=torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda')
        assignments={}
        for k in sorted({r['k'] for s,p,r in all_rows if r['dataset']==task}):
            _,_,centres=base.partition(data,k)
            assignment=torch.cdist(points,torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()[len(data['x_development']):]
            assignments[k]=assignment;geometries[f'{task}/K{k}']=hashlib.sha256(assignment.tobytes()).hexdigest()
        for stage,path,row in all_rows:
            if row['dataset']!=task:continue
            source=str(path.relative_to(ROOT.parents[1]));assignment=assignments[row['k']];releases={}
            if row['method']=='regional':
                releases['1']={mode:metrics(np.array(ep['labels'])[assignment],data['y_test'],CLASSES[task]) for mode,ep in row['endpoints'].items()}
                for mode,ep in row['endpoints'].items():
                    delta=releases['1'][mode]['accuracy']-ep['accuracy']
                    if abs(delta)>1e-15:changes.append(dict(source=source,release=1,decoder=mode,accuracy_difference=delta))
            else:
                d=np.load(stage/row['diagnostic_file'])
                for t in [1,2,3]:
                    releases[str(t)]={'actual':metrics(d[f'r{t}_labels'][assignment],data['y_test'],CLASSES[task])}
                    delta=releases[str(t)]['actual']['accuracy']-row['trace'][t-1]['diagnostics']['global_classifier_accuracy']
                    if abs(delta)>1e-15:changes.append(dict(source=source,release=t,decoder=row['decoder'],accuracy_difference=delta))
            records[source]=dict(source_sha256=sha(path),releases=releases)
    save(ROOT/'analysis/canonical-regional-readouts.json',dict(convention='shared public pipeline map; CUDA float32 cdist, TF32 disabled, development+test query shape; no labels choose assignment',
        geometries=geometries,records=records,native_metric_changes=changes,original_raw_records_modified=False,
        maximum_absolute_accuracy_difference=max(abs(x['accuracy_difference']) for x in changes)))
    print('Canonical public readouts:',len(records),'sources;',len(changes),'native metric differences; max points',100*max(abs(x['accuracy_difference']) for x in changes))

if __name__=='__main__':main()
