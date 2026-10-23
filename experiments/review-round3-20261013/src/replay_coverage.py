"""Reproduce frozen Web runs and inspect the already protected target outputs.

Instrumentation records target histograms without changing training or releases.
Rounds 1/2 supervise later training; round 3 remains charged but is unused by
the final local classifiers. Evaluation diagnostics are offline measurements.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'review-round2-20261013'
sys.path.insert(0,str(PARENT/'src'))
import engine


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def coverage(targets,round_number):
    counts=torch.bincount(targets[targets>=0],minlength=6).cpu().tolist()
    return {'round':round_number,'target_class_counts':counts,'represented_target_classes':sum(v>0 for v in counts),
        'target_fraction':(np.array(counts)/sum(counts)).tolist(),
        'supervises_next_local_round':round_number<3,'all_proxy_items_labelled':sum(counts)==len(targets)}


def main():
    jobs=[]
    for kind,methods in [('public-selected',['SDD','representative','DP-FedCT','DP-Head']),
                         ('fixed-no-task-labels',['SDD','representative'])]:
        for seed in range(8101,8111):
            for method in methods:jobs.append({'kind':kind,'method':method,'seed':seed,'n':3000,'epsilon':1.})
    plan={'jobs':jobs,'instrumentation_only':True,'new_configuration_selection':False,
        'training_and_noise_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [PARENT/'src/engine.py',PARENT/'src/privacy.py',PARENT/'src/exact_dg.so']},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    path=ROOT/'COVERAGE-SCHEDULE.json'
    if path.exists():assert json.loads(path.read_text())['jobs']==jobs
    else:save(path,plan)
    data=engine.load_data('web');xp=torch.as_tensor(data['x_proxy'],device=engine.DEVICE)
    xt=torch.as_tensor(data['x_test'],device=engine.DEVICE);yt=torch.as_tensor(data['y_test'],device=engine.DEVICE)
    cfgpath=PARENT/'public-selection/web_aNone_n1000.json'
    cfg=json.loads(cfgpath.read_text())['budgets']['1.0']
    original_release=engine.release_labels;original_parameter=engine.parameter_release
    for i,job in enumerate(jobs):
        filename=f"coverage_{job['kind']}_s{job['seed']}_{job['method']}.json"
        output=ROOT/'diagnostics'/filename
        if output.exists():continue
        observations=[]
        def labels_hook(weights,data,method,k,epsilon,seed,round_number,mode='raw',pate=None,colluders=0):
            result=original_release(weights,data,method,k,epsilon,seed,round_number,mode,pate,colluders)
            observations.append(coverage(result[0],round_number))
            return result
        def head_hook(weights,reference,clip,epsilon,seed,round_number):
            result=original_parameter(weights,reference,clip,epsilon,seed,round_number)
            targets=torch.matmul(xp,result[0]).argmax(-1)
            q=coverage(targets,round_number)
            a,b=engine.evaluate(result[0][None],xt,yt)
            q.update(global_accuracy=a,global_balanced_accuracy=b,
                global_per_class_recall=engine.evaluate.last_recall.copy(),
                clipped_fraction=result[2]['clipped_fraction'])
            observations.append(q);return result
        engine.release_labels=labels_hook;engine.parameter_release=head_hook
        try:
            k=cfg['k'] if job['kind']=='public-selected' else 6
            result=engine.run('web',3000,None,job['seed'],job['method'],k,1.,
                mode=cfg['mode'] if job['kind']=='public-selected' else 'center',clip=cfg['clip'],pate=cfg['pate'])
        finally:
            engine.release_labels=original_release;engine.parameter_release=original_parameter
        prefix='web_3000_None_1.0_' if job['kind']=='public-selected' else 'label_free_web_3000_None_1.0_'
        prior=PARENT/'results'/f"{prefix}{job['seed']}_{job['method']}.json"
        old=json.loads(prior.read_text())
        error=max(abs(result[key]-old[key]) for key in ['accuracy','balanced_accuracy','initial_accuracy','initial_balanced_accuracy'])
        assert error<2e-6 and result['population']==old['population']
        assert np.max(np.abs(np.array(result['per_class_recall'])-old['per_class_recall']))<2e-6
        value={'job':job,'k':k,'mode':result['mode'],'observations':observations,
            'original_result_file':prior.name,'original_result_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
            'max_original_endpoint_abs_difference':error,'per_class_recall':result['per_class_recall'],
            'balanced_accuracy':result['balanced_accuracy'],'third_release_charged_but_not_used_for_local_training':True,
            'observed_target_histograms_are_postprocessing_of_existing_protected_releases':True,
            'all_diagnostics_offline':True}
        save(output,value)
        print('COVERAGE',i+1,'/',len(jobs),filename,flush=True)
    save(ROOT/'COVERAGE-COMPLETE.json',{'diagnostic_replays':len(jobs),'utc':time.time()})


if __name__=='__main__':main()
