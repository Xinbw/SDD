"""Scientific record, budget accounting, pairing, regeneration and layout audit."""
import argparse
import csv
import json
import re
from pathlib import Path
import fitz
import mpmath as mp
import numpy as np
from common import ROOT,CLASSES,SETTINGS,SEEDS,release_key,sha,frozen_design

def validate(project,partial=False,layout=True):
    stage=project/'experiments'/ROOT.name;parent=stage.parent/'release-selector-20261019'
    manifest=json.loads((stage/'PARENT-MANIFEST.json').read_text())
    anon=(project/'ANONYMIZATION.json').exists();omitted=[]
    binary=json.loads((project/'experiments/review-round3-20261013/SAMPLER-ANONYMIZATION.json').read_text())['anonymous_binary_sha256']
    for name,digest in manifest['scientific_files'].items():
        path=project/name
        if anon and path.name=='finalize_revision.py' and not path.exists():omitted.append(name);continue
        allowed={digest,manifest['anonymous_parent_scientific_files'].get(name,digest)}
        if path.name=='exact_dg.so':allowed.add(binary)
        assert path.is_file() and sha(path) in allowed,name
    design=json.loads((stage/'DESIGN-FROZEN.json').read_text())
    assert design['design']==json.loads(json.dumps(frozen_design()))
    history=stage/'src/implementation-history'
    for name,freeze,snapshot in [
        ('common.py','DESIGN-FROZEN.json','common_initial.py'),
        ('run_private.py','DESIGN-FROZEN.json','run_private_before_head_guard.py'),
        ('run_private.py','PRIVATE-RUNTIME-SOURCES.json','run_private_executed.py'),
        ('public_frontier.py','PUBLIC-EXECUTION-FROZEN.json','public_frontier_executed.py'),
        ('public_frontier.py','PUBLIC-IO-OPTIMIZATION.json','public_frontier_vectorized.py'),
        ('geometry_labels.py','GEOMETRY-LABEL-DESIGN.json','geometry_labels_executed.py'),
        ('public_frontier.py','JOINT-NAMING-DESIGN.json','public_frontier_before_deterministic_prototype.py'),
        ('anchored_fusion.py','JOINT-NAMING-DESIGN.json','anchored_fusion_before_deterministic_prototype.py')]:
        expected=json.loads((stage/freeze).read_text())['source_sha256'][name]
        assert sha(history/snapshot)==expected,(freeze,snapshot)
    for name,frozen in [('head_fixed.py','FIXED-HEAD-DESIGN.json'),('students.py','STUDENT-DESIGN.json'),('confirm_joint.py','JOINT-CONFIRMATION-DESIGN.json'),
        ('confirm_point.py','CONFIRMATION-POINT-DESIGN.json'),('joint_controls.py','JOINT-RELEASE-CONTROLS.json')]:
        assert sha(stage/'src'/name)==json.loads((stage/frozen).read_text())['source_sha256'][name]
    deterministic=json.loads((stage/'DETERMINISTIC-PUBLIC-IMPLEMENTATION.json').read_text())
    for name,digest in deterministic['active_source_sha256'].items():assert sha(stage/'src'/name)==digest
    assert deterministic['superseded_partial_joint_records']==2049
    for name in ['engine_base.py','privacy.py','exact_dg.cpp']:
        assert sha(stage/'src'/name)==design['source_sha256'][name]
    schedule=json.loads((stage/'PRIVATE-SCHEDULE.json').read_text())['design']['expected_results']
    heads=json.loads((stage/'FIXED-HEAD-DESIGN.json').read_text())['design']['jobs']
    assert len(schedule)==840 and len(heads)==160
    results={p.stem:json.loads(p.read_text()) for p in (stage/'results').glob('*.json')}
    assert set(results)==set(schedule+heads)
    mp.mp.dps=90;certificates=0;populations={}
    for key,row in results.items():
        c=row['privacy'];certificates+=1
        assert c['rounds']==row['releases']==1
        assert c['actual_clients']==c['calibrated_clients']==row['n']
        rho=mp.mpf(c['sensitivity_integer_squared'])/(2*c['honest_clients']*mp.mpf(c['sigma_share_integer'])**2)
        rho+=mp.mpf(c['dimension'])*mp.mpf(str(c['tau_certified_upper']))/4
        assert rho+2*mp.sqrt(rho*mp.log(100000))<=mp.mpf(str(row['epsilon']))
        assert populations.setdefault((row['dataset'],row['n'],row['seed']),row['population'])==row['population']
        path=stage/row['diagnostic_file'];assert sha(path)==row['diagnostic_sha256']
        if row['method'] not in ['regional','per-example']:continue
        diag=np.load(path);assert np.all(diag['counts'].sum(1)==row['n'])
        assert c['sensitivity_integer_squared']==2*len(diag['counts'])*(1<<16)**2
        if 'coverage' in row['endpoints']:assert len(set(row['endpoints']['coverage']['labels']))==CLASSES[row['dataset']]
    oldplan=json.loads((parent/'STUDY-SCHEDULE.json').read_text())['design']
    matched=0
    for role in oldplan['roles']:
        if role['role']!='fixed-center':continue
        old=json.loads((parent/'results'/(role['job']+'.json')).read_text())
        new=results[release_key(old['dataset'],old['n'],old['epsilon'],old['seed'],60,'regional',CLASSES[old['dataset']],'M')]
        a=np.load(parent/old['diagnostic_file'])['r1_counts'];b=np.load(stage/new['diagnostic_file'])['counts']
        assert np.array_equal(a,b),role['job'];matched+=1
        assert old['population']==new['population']
        ratio=new['privacy']['residual_std']/old['trace'][0]['privacy']['residual_std']
        assert abs(ratio-1/np.sqrt(3))<2/old['trace'][0]['privacy']['sigma_share_integer']
    assert matched==80
    studentplan=json.loads((stage/'STUDENT-DESIGN.json').read_text())['design']['jobs']
    readers={p.stem:json.loads(p.read_text()) for p in (stage/'analysis/students').glob('*.json')}
    assert set(readers)==set(studentplan) and len(readers)==640
    for key,r in readers.items():
        assert r['source_sha256']==sha(stage/'results'/(r['source']+'.json'))
        assert r['public_steps']==180 and r['private_queries_added']==r['private_steps_added']==0
        assert r['private_state_used'] is False
    count=0;pairs=0;groups={};identity=[]
    for task,c in CLASSES.items():
        with np.load(stage/'inputs'/f'{task}.npz') as data:dev=data['y_development']
        for path in (stage/'public').glob(f'{task}_*_m*.json'):
            r=json.loads(path.read_text());idx=np.array(r['indices']);y=np.array(r['labels'])
            assert r['m']==r['label_budget']==len(idx)==len(np.unique(idx))==r['q']*c
            assert np.array_equal(y,dev[idx]);assert r['external_validation_labels']==0
            assert r['missing_classes']==int(np.sum(np.bincount(y,minlength=c)==0))
            assert len(r['baselines'])==12
            best=max(r['baselines'],key=lambda name:r['baselines'][name]['loo_score'])
            assert r['selected_public']==best and r['public_portfolio']==r['baselines'][best]
            expected={(n,e,s) for d,n,e in SETTINGS if d==task for s in SEEDS}
            assert {(x['n'],x['epsilon'],x['seed']) for x in r['hybrid']}==expected
            for h in r['hybrid']:
                source=stage/'results'/(h['protected_source']+'.json');assert h['source_sha256']==sha(source)
                g=h['gate'];accepted=(g['paired_loo_gain_lower']>0 and g['all_classes_in_proxy_predictions']
                    and g['minimum_loo_wilson_lower']>1e-12 and g['selected']!='public')
                assert accepted==g['private_augmentation_published']
                assert h['gated']==(h['ungated'] if accepted else r['public_portfolio'])
                assert g['selected']==max(g['candidate_loo_scores'],key=g['candidate_loo_scores'].get)
                if r['missing_classes']:assert not accepted
                pairs+=1
            groups.setdefault((task,r['scheme'],r['q']),set()).add(r['draw']);count+=1
            identity.append((task,r['scheme'],r['draw'],r['q']))
    assert len(identity)==len(set(identity))
    if not partial:
        assert count==3024 and pairs==57480,(count,pairs)
        for task in CLASSES:
            for q in [1,3,5,10,20,50]:
                assert groups[(task,'random',q)]==set(range(100))
                assert groups[(task,'geometry',q)]=={0}
                if task in ['cifar10','mnist']:assert groups[(task,'stratified',q)]==set(range(50))
    fresh=list((stage/'confirmation/results').glob('*.json'));assert len(fresh)==90
    for path in fresh:
        row=json.loads(path.read_text());c=row['privacy'];certificates+=1
        assert c['rounds']==row['releases']==1
        assert c['actual_clients']==c['calibrated_clients']==row['n']
        rho=mp.mpf(c['sensitivity_integer_squared'])/(2*c['honest_clients']*mp.mpf(c['sigma_share_integer'])**2)
        rho+=mp.mpf(c['dimension'])*mp.mpf(str(c['tau_certified_upper']))/4
        assert rho+2*mp.sqrt(rho*mp.log(100000))<=mp.mpf(str(row['epsilon']))
        assert populations.setdefault((row['dataset'],row['n'],row['seed']),row['population'])==row['population']
        assert sha(stage/'confirmation'/row['diagnostic_file'])==row['diagnostic_sha256']
        if row['method'] in ['regional','per-example']:
            diag=np.load(stage/'confirmation'/row['diagnostic_file']);assert np.all(diag['counts'].sum(1)==row['n'])
            assert c['sensitivity_integer_squared']==2*len(diag['counts'])*(1<<16)**2
    joint_sets=joint_pairs=control_cases=0
    CONFIRM_SETTINGS=[('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1)]
    CONFIRM_SEEDS=list(range(8801,8811))
    for phase,folder,controlfolder,settings,seeds in [('development','joint-naming','joint-controls',SETTINGS,SEEDS),
            ('confirmation','joint-confirmation','joint-controls-confirmation',CONFIRM_SETTINGS,CONFIRM_SEEDS)]:
        files=sorted((stage/'analysis'/folder).glob('*.json'));assert len(files)==(3024 if phase=='development' else 1200)
        source_root=stage if phase=='development' else stage/'confirmation'
        groups={}
        for path in files:
            r=json.loads(path.read_text());c=CLASSES[r['dataset']];idx=np.array(r['indices']);y=np.array(r['labels'])
            with np.load(stage/'inputs'/f"{r['dataset']}.npz") as data:assert np.array_equal(y,data['y_development'][idx])
            assert r['m']==r['label_budget']==len(idx)==len(np.unique(idx))==r['q']*c
            assert r['external_validation_labels']==0 and len(r['baselines'])==14
            label_source=stage/('public' if phase=='development' else 'confirmation/samples')/r['source']
            assert sha(label_source)==r['source_sha256']
            best=max(r['baselines'],key=lambda k:r['baselines'][k]['loo_score'])
            assert r['selected_public']==best and r['public_portfolio']==r['baselines'][best]
            expected={(n,e,s) for d,n,e in settings if d==r['dataset'] for s in seeds}
            assert {(h['n'],h['epsilon'],h['seed']) for h in r['hybrid']}==expected
            for h in r['hybrid']:
                assert sha(source_root/'results'/(h['protected_source']+'.json'))==h['source_sha256']
                g=h['gate'];assert not g['private_local_utility_used_for_selection']
                accepted=bool(h['selected_uses_private_votes'] and g['paired_loo_gain_lower']>0 and g['all_classes_in_proxy_predictions'] and g['minimum_loo_wilson_lower']>1e-12)
                assert accepted==g['private_augmentation_published']
                assert h['gated']==(h['ungated'] if accepted else r['public_portfolio'])
                assert g['selected']==max(g['candidate_loo_scores'],key=g['candidate_loo_scores'].get)
                if r['missing_classes']:assert not accepted and not g['public_classifier_published']
                joint_pairs+=1
            cp=stage/'analysis'/controlfolder/path.name;control=json.loads(cp.read_text())
            assert control['source_sha256']==sha(path) and len(control['controls'])==2*len(expected)
            observed=set()
            for h in control['controls']:
                assert sha(source_root/'results'/(h['source']+'.json'))==h['source_sha256']
                assert h['extra_private_queries']==0
                observed.add((h['n'],h['epsilon'],h['seed'],h['method']));control_cases+=1
            assert observed=={(*z,m) for z in expected for m in ['head-fixed','per-example']}
            groups.setdefault((r['dataset'],r['scheme'],r['q']),set()).add(r['draw']);joint_sets+=1
        if phase=='confirmation':
            for task in ['cifar10','mnist']:
                for q in [1,3,5,10,20,50]:assert groups[(task,'random',q)]==set(range(100))
    assert (joint_sets,joint_pairs,control_cases)==(4224,75480,150960)
    assert len(populations)==90 and certificates==1090
    bounds=json.loads((stage/'analysis/bound-diagnostics.json').read_text());assert len(bounds)==80
    assert {x['coupled_bound'] for x in bounds if x['coupled_bound'] is not None}=={1.}
    canonical=json.loads((stage/'analysis/canonical-regional-readouts.json').read_text())
    assert len(canonical['records'])==855 and canonical['original_raw_records_modified'] is False
    for source,row in canonical['records'].items():assert sha(project/source)==row['source_sha256'],source
    text='\n'.join(p.read_text() for p in (project/'paper').glob('sec-*.tex'))+(project/'paper/main.tex').read_text()
    assert text.count('\\includegraphics')==3
    assert not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]',text)
    report=dict(preserved_primary_records=2336,new_single_release_conditions=1090,
        distinct_actual_cohorts=90,private_teacher_states=150,unchanged_first_vote_controls=matched,
        certificates_recomputed=certificates,privacy_decimal_digits=90,public_students=640,
        legacy_label_sets=count,legacy_crossed_cases=pairs,
        public_label_sets=joint_sets,crossed_label_cohort_cases=joint_pairs,matched_release_control_cases=control_cases,all_labels_counted=True,
        public_gate_is_heuristic=True,all_adverse_seeds_retained=True,shared_public_regional_assignments=True,
        native_metric_maximum_difference_points=100*canonical['maximum_absolute_accuracy_difference'],omitted_anonymous_packaging_sources=omitted)
    if layout:
        pdf=fitz.open(project/'paper/main.pdf');texts=[p.get_text() for p in pdf]
        assert len(pdf)<=12 and 'CONCLUSION' in texts[7] and 'REFERENCES' in texts[8]
        assert not any('REFERENCES' in s for s in texts[:8])
        log_path=project/'paper/main.log'
        if log_path.exists():
            log=log_path.read_text();assert 'Overfull' not in log
            assert not re.search(r'(Citation|Reference).*undefined',log)
        else:
            build=json.loads((project/'paper/BUILD_REPORT.json').read_text())
            assert build['pdf_sha256']==sha(project/'paper/main.pdf')
            assert not build['overfull_boxes'] and not build['undefined_references']
        report.update(main_pages=8,total_pages=len(pdf),references_start_page=9,overfull_boxes=False,undefined_references=False)
    (stage/'revision-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--partial',action='store_true');p.add_argument('--no-layout',action='store_true');a=p.parse_args();validate(a.project.resolve(),a.partial,not a.no_layout)
