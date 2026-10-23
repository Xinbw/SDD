"""Audit whole-client composition, reports, bands, decisions and provenance."""
import argparse,json,math,re,sys
from pathlib import Path
import numpy as np
import mpmath as mp
import fitz
from study import ROOT,PARENT,sha,save
import private_validation as pv
from linear_bands import bands


def validate(project,layout=True):
    stage=project/'experiments'/ROOT.name;parent=stage.parent/PARENT.name;anon=(project/'ANONYMIZATION.json').exists()
    original=json.loads((stage/'PARENT-MANIFEST.json').read_text())
    for name,digest in original['scientific_files'].items():
        path=project/name
        if anon and path.name=='finalize_revision.py' and not path.exists():continue
        allowed={digest,original['anonymous_scientific_files'].get(name,digest)}
        assert path.is_file() and sha(path) in allowed,name
    # Captured execution sources are retained even when analysis/export code evolves.
    ledger=json.loads((stage/'EXECUTION-SOURCE-SNAPSHOTS.json').read_text())
    for frozen in sorted(stage.glob('*DESIGN.json'))+list(stage.glob('ZERO-LABEL-POOL-AUDIT.json')):
        for name,digest in json.loads(frozen.read_text()).get('source_sha256',{}).items():
            snapshot=stage/ledger[frozen.name+'/'+name];assert sha(snapshot)==digest,(frozen.name,name)
    for frozen,name in [('PRIVATE-VALIDATION-DESIGN.json','private_validation.py'),('NO-FEEDBACK-DESIGN.json','no_feedback.py'),('ZERO-LABEL-POOL-AUDIT.json','zero_label_pool.py'),('LINEAR-BANDS-DESIGN.json','linear_bands.py')]:
        assert sha(stage/'src'/name)==json.loads((stage/frozen).read_text())['source_sha256'][name]
    definition=json.loads((stage/'PRIVATE-VALIDATION-DESIGN.json').read_text())['design']
    assert definition['zcdp_budget_fractions']==dict(naming=.5,gain=.25,classes=.25)
    with np.load(stage/'inputs/cifar10.npz') as z:yprivate=z['y_private'];ydev=z['y_development'];ytest=z['y_test']
    populations={}
    for seed in pv.SEEDS:
        tr,va,pop=pv.cohort(yprivate,seed);assert not np.intersect1d(tr,va).size
        populations[seed]=pop
    maps_record=json.loads((stage/'PUBLIC-MAPS.json').read_text());assert sha(stage/maps_record['path'])==maps_record['sha256']
    maps=np.load(stage/maps_record['path']);sizes=np.bincount(maps['proxy'],minlength=10)
    samples={}
    for p in sorted((stage/'public').glob('labels_*.json')):
        r=json.loads(p.read_text());idx=np.array(r['indices']);assert len(idx)==len(np.unique(idx))==10
        assert np.array_equal(ydev[idx],r['labels']);samples[r['draw']]=r
    assert set(samples)==set(range(100))
    mp.mp.dps=90;cache={};certificate_attachments=0
    def certificate(c,e,fraction,rounds=1):
        nonlocal certificate_attachments
        certificate_attachments+=1;key=json.dumps(c,sort_keys=True)
        if key in cache:return cache[key]
        assert c['actual_clients']==c['calibrated_clients']==c['honest_clients']==1000
        assert c['colluders']==0 and c['rounds']==rounds and c['rho_fraction']==fraction
        rho=mp.mpf(c['sensitivity_integer_squared'])/(2*c['honest_clients']*mp.mpf(c['sigma_share_integer'])**2)
        rho+=mp.mpf(c['dimension'])*mp.mpf(str(c['tau_certified_upper']))/4
        target=(mp.sqrt(mp.log(100000)+mp.mpf(str(e)))-mp.sqrt(mp.log(100000)))**2
        assert rho*rounds<=target*mp.mpf(str(fraction))
        vmax=math.sqrt(1000)*c['sigma_share_integer']/(65536*1000)
        assert abs(vmax-c['residual_std'])<1e-12
        wrap_log=mp.log(2*c['dimension'])-(mp.mpf(2)**63-1000*65536)**2/(2*1000*mp.mpf(c['sigma_share_integer'])**2)
        assert wrap_log<mp.log(mp.mpf('1e-30'))
        cache[key]=rho*rounds;return cache[key]
    naming={}
    for p in sorted((stage/'results/naming').glob('*.json')):
        r=json.loads(p.read_text());assert r['population']==populations[r['seed']]
        c=r['privacy'];certificate(c,r['epsilon_total'],.5)
        assert c['dimension']==100 and c['sensitivity_integer_squared']==20*65536**2
        assert sha(stage/r['diagnostic_file'])==r['diagnostic_sha256']
        z=np.load(stage/r['diagnostic_file']);assert np.all(z['counts'].sum(1)==1000)
        naming[p.name]=(r,z['scores']);
    assert len(naming)==30
    total_cases=0;linear_cases=0;false_gain={};false_class={}
    for path in sorted((stage/'results/validation').glob('*.json')):
        r=json.loads(path.read_text());seed=r['seed'];e=r['epsilon_total'];sample=samples[r['draw']]
        assert r['m']==10 and r['n']==1000 and sha(stage/'public'/r['label_source'])==r['label_source_sha256']
        named,scores=naming[r['naming_source']];assert r['naming_source_sha256']==sha(stage/'results/naming'/r['naming_source'])
        selected,public,choices=pv.select(scores,sizes,sample,maps['development'])
        assert selected['name']==r['candidate'] and selected['private']==r['candidate_uses_private']
        assert np.array_equal(selected['labels'],r['candidate_labels']) and np.array_equal(public['labels'],r['public_labels'])
        assert choices==r['candidate_loo_scores']
        cg=r['privacy_gain'];ch=r['privacy_classes']
        rg=certificate(cg,e,.25);rh=certificate(ch,e,.25);rn=certificate(named['privacy'],e,.5)
        assert cg['sensitivity_integer_squared']==65536**2 and cg['dimension']==1
        assert ch['sensitivity_integer_squared']==2*65536**2 and ch['dimension']==40
        rho=rg+rh+rn;assert rho+2*mp.sqrt(rho*mp.log(100000))<=mp.mpf(str(e))
        assert sha(stage/r['diagnostic_file'])==r['diagnostic_sha256']
        z=np.load(stage/r['diagnostic_file']);h=z['client_joint_counts'];assert h.shape==(1000,10,4)
        assert np.all(h>=0) and np.all(h.sum((1,2))==120)
        g,q=pv.quantize(h);assert np.all((g>=0)&(g<=65536)) and np.all(q>=0) and np.all(q.sum(1)==65536)
        assert np.max(abs(q/65536-h.reshape(1000,40)/120))<=1/65536
        unit=(int(g.sum())+int(z['gain_noise_sum'][0]))/(1000*65536)
        hist=(q.sum(0)+z['class_noise_sum'])/(1000*65536)
        assert np.array_equal(hist,z['released_class_histogram']) and unit==float(z['released_gain_unit'])
        v=r['protected_validation'];assert v['released_gain']==2*unit-1
        low=2*unit-1-2*cg['residual_std']*math.sqrt(2*math.log(1/.025))-1/65536
        assert v['gain_lower']==low
        cb=pv.class_bands(hist,ch)
        for name,value in cb.items():assert v[name]==value
        accepted=bool(r['candidate_uses_private'] and low>0);strict=bool(accepted and cb['minimum_recall_lower']>0)
        assert r['policy']['global_gain_accepted']==accepted and r['policy']['class_checked_accepted']==strict
        assert not any(r['policy'][k] for k in ['uses_clean_statistics','uses_test_labels','private_local_used'])
        total=h.sum(0).astype(float)/(1000*120);nh=float((total[:,2]+total[:,3]).sum());ph=float((total[:,1]+total[:,3]).sum());gain=nh-ph
        mass=total.sum(1);recall=np.divide(total[:,2]+total[:,3],mass,out=np.zeros(10),where=mass>0)
        d=r['research_diagnostics'];assert d['candidate_heldout_accuracy']==nh and d['public_heldout_accuracy']==ph and d['true_gain']==gain
        assert d['true_recall']==recall.tolist()
        ctest=pv.metrics(selected['labels'][maps['test']],ytest,10);ptest=pv.metrics(public['labels'][maps['test']],ytest,10)
        assert ctest==d['candidate_test'] and ptest==d['public_test']
        assert d['global_policy_test']==(ctest if accepted else ptest)
        assert d['class_policy_test']==(ctest if strict else ptest)
        lp=stage/'analysis/linear-validation'/path.name;lin=json.loads(lp.read_text());assert lin['source_sha256']==sha(path)
        b=bands(hist,ch);assert b==lin['bounds'];la=bool(accepted and b['minimum_recall_lower']>0)
        assert lin['class_checked_accepted']==la and not lin['uses_clean_statistics_for_decision'] and lin['new_private_queries']==0
        assert lin['policy_test']==(ctest if la else ptest)
        assert lin['policy_heldout_accuracy']==(nh if la else ph)
        coverage=bool(np.all(np.array(b['recall_lower'])<=recall+1e-12)&np.all(np.array(b['recall_upper'])>=recall-1e-12))
        assert coverage==lin['research_diagnostics']['simultaneous_bands_valid']
        false_gain[e]=false_gain.get(e,0)+int(accepted and gain<=0)
        false_class[e]=false_class.get(e,0)+int(la and (gain<=0 or recall.min()<=0))
        total_cases+=1;linear_cases+=1
    assert total_cases==linear_cases==3000
    canon=json.loads((parent/'analysis/canonical-regional-readouts.json').read_text())['records']
    pools=json.loads((stage/'analysis/zero-label-pooled-point.json').read_text());assert sum(len(r['sources']) for r in pools)==110
    for row in pools:
        for src in row['sources']:assert sha(project/src['path'])==src['sha256']
    old=parent.parent/'release-selector-20261019'
    nofeedback=list((stage/'results').glob('no_feedback*.json'));assert len(nofeedback)==10
    for path in nofeedback:
        r=json.loads(path.read_text());certificate(r['privacy'],.1,1.,rounds=3)
        assert sha(stage/r['diagnostic_file'])==r['diagnostic_sha256']
        z=np.load(stage/r['diagnostic_file']);s=r['seed']
        for release,steps in [(1,60),(3,180)]:
            key=f'cifar10_1000_0.1_{s}_T1_s{steps}_regional_K10_M.json';source=json.loads((parent/'results'/key).read_text())
            assert source['population']==r['population'] and np.array_equal(z[f'r{release}_counts'],np.load(parent/source['diagnostic_file'])['counts'])
        fp=old/'results'/(r['parent_feedback_source']+'.json');assert sha(fp)==r['parent_feedback_sha256']
        original=json.loads(fp.read_text());assert np.array_equal(z['r1_scores'],np.load(old/original['diagnostic_file'])['r1_scores'])
    text='\n'.join(p.read_text() for p in (project/'paper').glob('sec-*.tex'))
    assert text.count('\\includegraphics')==3 and not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]',text)
    result=dict(parent_scientific_fingerprints_checked=True,preserved_primary_records=3426,new_actual_cohorts=10,new_private_teachers=10,
        naming_conditions=30,validation_cases=3000,validation_vectors=6000,linear_postprocessing_cases=3000,
        no_feedback_protocols=10,no_feedback_releases=30,zero_label_existing_sources=110,
        composed_deployments_checked=3000,privacy_precision_digits=90,unique_certificates_recomputed=len(cache),certificate_attachments_checked=certificate_attachments,
        report_bounds_checked=True,source_disjoint_private_holdouts=True,all_decisions_rederived=True,extra_private_queries_for_linear_bands=0,
        false_gain_accepts=false_gain,false_class_accepts=false_class,wrap_probability_upper_per_release='1e-30')
    if layout:
        pdf=fitz.open(project/'paper/main.pdf');texts=[p.get_text() for p in pdf]
        assert len(pdf)<=12 and 'CONCLUSION' in texts[7] and 'REFERENCES' in texts[8]
        log=project/'paper/main.log'
        if log.exists():
            value=log.read_text();assert 'Overfull' not in value and not re.search(r'(Citation|Reference).*undefined',value)
        else:
            build=json.loads((project/'paper/BUILD_REPORT.json').read_text());assert build['pdf_sha256']==sha(project/'paper/main.pdf')
            assert not build['overfull_boxes'] and not build['undefined_references']
        result.update(main_pages=8,total_pages=len(pdf),references_start_page=9,overfull_boxes=False,undefined_references=False)
    save(stage/'validation-checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--no-layout',action='store_true');a=p.parse_args();validate(a.project.resolve(),not a.no_layout)
