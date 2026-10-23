"""Reconstruct source-disjoint validation and three-round actual cohorts."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from study import ROOT,PARENT,save
import private_validation as pv
from linear_bands import bands
base=pv.base;privacy=pv.privacy


def main():
    project=ROOT.parents[1];output=project/'reproduction-output/smoke-value'
    for folder in ['checkpoints','inputs/partitions']:(output/folder).mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'inputs').glob('*.npz'):
        target=output/'inputs'/source.name
        if not target.exists():target.symlink_to(source.resolve())
    # The test always reconstructs training, even if an earlier smoke cached it.
    checkpoint=output/'checkpoints/cifar1k_holdout_s8901.pt'
    if checkpoint.exists():checkpoint.unlink()
    with np.load(ROOT/'inputs/cifar10.npz') as z:data={k:z[k] for k in ['x_private','y_private','x_proxy','x_development','y_development','x_test']}
    data['num_classes']=10;old_pv=pv.ROOT;old_base=base.ROOT;pv.ROOT=output;base.ROOT=output
    checks=[]
    try:
        w,val,pop=pv.train(data,8901)
        torch.backends.cuda.matmul.allow_tf32=True
        ids,reps,centres=base.partition(data,10);sizes=np.bincount(ids,minlength=10)
        counts=base.vote_counts(w,torch.as_tensor(data['x_proxy'],device='cuda'),ids)
        torch.backends.cuda.matmul.allow_tf32=False
        assignment=torch.cdist(torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
        private_regions=torch.cdist(torch.as_tensor(data['x_private'],device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
        for e in [.1,3.]:
            named=json.loads((ROOT/'results/naming'/f's8901_e{e:g}.json').read_text());reference=np.load(ROOT/named['diagnostic_file'])
            assert pop==named['population'] and np.array_equal(counts.cpu().numpy(),reference['counts'])
            scores=base.noisy_votes(counts,named['privacy'],8901*1000003+int(e*100)*901+20261021).cpu().numpy()
            assert np.array_equal(scores,reference['scores'])
            source=ROOT/'results/validation'/f's8901_e{e:g}_m10_d0.json';r=json.loads(source.read_text());sample=json.loads((ROOT/'public'/r['label_source']).read_text())
            chosen,pub,choices=pv.select(scores,sizes,sample,assignment[:len(data['x_development'])]);assert chosen['name']==r['candidate']
            h=pv.hist_counts(chosen['labels'],pub['labels'],private_regions[val],data['y_private'][val])
            actual,diag=pv.evaluate_case(h,r['privacy_gain'],r['privacy_classes'],8901*1000000007+int(e*100)*100003+20261022)
            expected=np.load(ROOT/r['diagnostic_file'])
            for key,value in diag.items():assert np.array_equal(value,expected[key]),key
            assert actual==r['protected_validation']
            lin=json.loads((ROOT/'analysis/linear-validation'/source.name).read_text());bound=bands(diag['released_class_histogram'],r['privacy_classes']);assert bound==lin['bounds']
            accept=bool(chosen['private'] and actual['gain_lower']>0 and bound['minimum_recall_lower']>0)
            assert accept==lin['class_checked_accepted']
            checks.append(dict(cohort=8901,total_epsilon=e,source_disjoint_population_equal=True,training_and_votes_equal=True,naming_release_equal=True,
                paired_gain_and_histogram_releases_equal=True,initial_and_direct_sum_bands_equal=True,publication_decisions_equal=True))
        del w
        # Independent 600-record three-round source, with no inherited private checkpoint.
        torch.backends.cuda.matmul.allow_tf32=True
        data2=base.load_data('cifar10');index,lens,pop2=base.make_population(data2,1000,.1,8501)
        generator=torch.Generator(device='cuda').manual_seed(8501)
        w=torch.randn((1000,128,10),device='cuda',generator=generator)*.01;v=torch.zeros_like(w)
        x,y,idx,sz=[torch.as_tensor(a,device='cuda') for a in [data2['x_private'],data2['y_private'],index,lens]]
        r=json.loads((ROOT/'results/no_feedback_s8501.json').read_text());ref=np.load(ROOT/r['diagnostic_file']);assert pop2==r['population']
        for t in [1,2,3]:
            w,v=base.train_round(w,v,x,y,idx,sz,8501,t)
            clean=base.vote_counts(w,torch.as_tensor(data2['x_proxy'],device='cuda'),ids)
            assert np.array_equal(clean.cpu().numpy(),ref[f'r{t}_counts'])
            released=base.noisy_votes(clean,r['privacy'],8501*1000003+t*101).cpu().numpy();assert np.array_equal(released,ref[f'r{t}_scores'])
        checks.append(dict(cohort=8501,no_feedback_three_rounds_reconstructed=True,all_three_clean_and_protected_matrices_equal=True))
        torch.backends.cuda.matmul.allow_tf32=False
        source=json.loads((PARENT/'results/cifar10_1000_0.1_8501_T1_s180_per-example_K0_-.json').read_text());a=np.load(PARENT/source['diagnostic_file'])['scores']
        pooled=np.stack([a[ids==k].mean(0) for k in range(10)]);test=assignment[len(data['x_development']):]
        with np.load(ROOT/'inputs/cifar10.npz') as z:truth=z['y_test']
        acc=pv.metrics(pv.decode(pooled,sizes,'center')[test],truth,10)['accuracy']
        old=next(x for x in json.loads((ROOT/'analysis/zero-label-pooled-point.json').read_text()) if x['phase']=='paired' and x['dataset']=='cifar10' and x['n']==1000)
        assert acc==old['readouts']['center']['per_seed'][0]
        checks.append(dict(zero_label_existing_point_postprocessing_equal=True))
    finally:pv.ROOT=old_pv;base.ROOT=old_base
    save(output/'checks.json',checks)
    print('PASS: two actual cohorts reconstructed; source-disjoint private training, exact naming/gain/class releases, both band rules and decisions; three no-feedback releases; zero-label pooling.')

if __name__=='__main__':main()
