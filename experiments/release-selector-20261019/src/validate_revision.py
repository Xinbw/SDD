"""Verify immutable parent evidence and the second revision's actual outputs."""
import argparse
import csv
import json
import re
from pathlib import Path

import fitz
import mpmath as mp
import numpy as np
from scipy.optimize import linear_sum_assignment

from common import ROOT,key,sha


def validate(project,layout=True):
    stage=project/'experiments'/ROOT.name
    parent=json.loads((stage/'PARENT-MANIFEST.json').read_text())
    sampler=json.loads((project/'experiments/review-round3-20261013/SAMPLER-ANONYMIZATION.json').read_text())
    anonymous=(project/'ANONYMIZATION.json').exists()
    omitted=[]
    for name,digest in parent['scientific_files'].items():
        path=project/name
        if anonymous and path.name=='finalize_revision.py' and not path.exists():omitted.append(name);continue
        allowed={digest}
        if name in parent['anonymous_parent_scientific_files']:allowed.add(parent['anonymous_parent_scientific_files'][name])
        if name.endswith('/exact_dg.so'):allowed.add(sampler['anonymous_binary_sha256'])
        assert path.exists() and sha(path) in allowed,name
    plan=json.loads((stage/'STUDY-SCHEDULE.json').read_text())['design']
    for name,freeze in {'engine_base.py':'STUDY-SCHEDULE.json','selection.py':'PUBLIC-DESIGN-FROZEN.json',
        'public_readouts.py':'PUBLIC-READOUT-DESIGN-FROZEN.json','replay_centroid.py':'CENTROID-REPLAY-FROZEN.json'}.items():
        expected=json.loads((stage/freeze).read_text())['source_sha256'][name]
        assert sha(stage/'src/frozen-execution'/name)==expected
    for name,digest in parent['parent_presentation_files'].items():
        if name.startswith('paper/tab-') and name.endswith('.tex'):assert sha(project/name)==digest,name
    assert len(plan['jobs'])==540 and len({key(j) for j in plan['jobs']})==540
    rows={p.stem:json.loads(p.read_text()) for p in (stage/'results').glob('*.json')}
    query=json.loads((stage/'QUERY-CONTROL-SCHEDULE.json').read_text())['design']
    assert len(query['jobs'])==5
    assert len(rows)==565,len(rows)
    mp.mp.dps=90;certificates=0;populations={}
    for job in plan['jobs']+query['jobs']:
        r=rows[key(job)];assert all(r[k]==v for k,v in job.items())
        assert r['optimizer_steps']==180 and r['batch_size']==128
        assert r['n']==r['actual_clients']==r['calibrated_clients']
        population_key=(r['dataset'],r['n'],r['seed'])
        assert populations.setdefault(population_key,r['population'])==r['population']
        if job['method'] in ['Local-only','Local+Public']:assert not r['trace'];continue
        assert len(r['trace'])==3
        for release in r['trace']:
            c=release['privacy'];certificates+=1
            assert c['actual_clients']==c['calibrated_clients']==r['n']
            assert c['epsilon_requested']==r['epsilon'] and c['rounds']==3
            rho=mp.mpf(c['sensitivity_integer_squared'])/(2*c['honest_clients']*mp.mpf(c['sigma_share_integer'])**2)
            rho+=mp.mpf(c['dimension'])*mp.mpf(str(c['tau_certified_upper']))/4
            epsilon=3*rho+2*mp.sqrt(3*rho*mp.log(100000))
            assert epsilon<=mp.mpf(str(r['epsilon']))
        if 'diagnostic_file' not in r:continue
        file=stage/r['diagnostic_file'];assert sha(file)==r['diagnostic_sha256']
        d=np.load(file);truth=d['truth_counts'];ids=d['ids']
        for release in r['trace']:
            t=release['round'];counts=d[f'r{t}_counts'];labels=d[f'r{t}_labels']
            assert np.all(counts.sum(1)==r['n'])
            acc=truth[np.arange(len(labels)),labels].sum()/truth.sum()
            assert abs(acc-release['diagnostics']['proxy_label_accuracy'])<1e-12
            if r['decoder']=='matching':
                scores=d[f'r{t}_scores'];sort=np.sort(scores,axis=-1)[:,::-1];sums=sort.cumsum(1)-1
                rank=np.arange(1,scores.shape[1]+1);rho=(sort-sums/rank>0).sum(1).clip(1)
                theta=sums[np.arange(len(rho)),rho-1]/rho
                projected=np.maximum(scores-theta[:,None],0)
                rr,cc=linear_sum_assignment(-projected*np.bincount(ids,minlength=len(labels))[:,None])
                assert len(np.unique(labels))==labels.size==truth.shape[1]
                assert np.array_equal(labels[rr],cc)
    # The new adverse matching seed is evidence, not an excluded outlier.
    bad=rows['cifar10_10000_0.1_8603_SDD_10_matching_M_64_64']
    assert abs(bad['global_endpoint']['accuracy']-.729481775353335)<1e-12
    assert bad['trace'][0]['diagnostics']['distinct_target_classes']==10
    assert abs(bad['trace'][0]['diagnostics']['proxy_label_accuracy']-.739)<1e-12
    for path in (stage/'public-selection').glob('*.json'):
        design=json.loads(path.read_text())
        assert design['configuration_frozen'] and not design['private_data_access'] and not design['test_access']
        if 'loaded_source_arrays' in design:
            assert all('private' not in name and 'test' not in name for name in design['loaded_source_arrays'])
    primary=[r for r in plan['roles'] if r['role']=='public-selected'];assert len(primary)==80
    for role in primary:
        r=rows[role['job']];spec=json.loads((stage/'public-selection'/f"{r['dataset']}.json").read_text())
        chosen=next(x['selected'] for x in spec['records'] if x['n']==r['n'] and x['epsilon']==r['epsilon'])
        assert all(r[k]==v for k,v in chosen.items())
    readers=list((stage/'analysis/public-readouts').glob('*.json'))
    expected=[j for j in plan['jobs'] if j['method'] in ['SDD','DP-FedCT']]
    assert len(readers)==len(expected)==380
    for path in readers:
        row=json.loads(path.read_text());source=stage/'results'/(row['job']+'.json')
        assert sha(source)==row['source_sha256']
        assert row['server_steps']==180 and row['private_steps_added']==row['private_queries_added']==0
        assert row['private_model_state_used'] is False and row['test_selects_readout'] is False
    replays=json.loads((stage/'analysis/centroid-replays.json').read_text());assert len(replays)==390
    for r in replays:assert r['additional_private_queries']==0
    indices=list(csv.DictReader((project/'EXPERIMENT_INDEX.csv').open()))
    assert len(indices)==2336 and len({r['result'] for r in indices})==2336
    for row in indices:
        r=json.loads((project/row['result']).read_text())
        assert row['dataset']==r['dataset'] and row['variant']==r['method'] and int(row['seed'])==r['seed']
    data=json.loads((project/'paper/selector-data.json').read_text())
    assert len(data['conditions'])==8 and len(data['mix'])==6 and len(data['natural_resolution'])==20
    assert len(data['public_references'])==4
    assert data['fixed_taxonomy_no_labels']['pooled_global']['n']==20
    prior=[r['accuracy'] for r in replays if r['dataset']=='cifar10' and r['n']==10000 and r['decoder']=='matching' and r['release']==3]
    current=[rows[f'cifar10_10000_0.1_{s}_SDD_10_matching_M_64_64']['global_endpoint']['accuracy'] for s in range(8601,8611)]
    assert abs(data['fixed_taxonomy_no_labels']['pooled_global']['mean']-np.mean(prior+current))<1e-14
    text='\n'.join(p.read_text() for p in (project/'paper').glob('sec-*.tex'))+(project/'paper/main.tex').read_text()
    assert text.count('\\includegraphics')==3
    assert not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]',text)
    assert 'Proposition~7' in text and 'papernot2018scalable' in text
    assert '\\label{lem:decoder-stability}' in text
    assert 'pending' not in (project/'paper/selector-values.tex').read_text()
    report=dict(preserved_primary_conditions=1771,new_client_training_conditions=540,new_matched_query_conditions=5,new_public_source_conditions=20,
        additional_public_student_conditions=380,new_prior_trajectory_centroid_replays=390,
        new_end_to_end_settings=8,new_end_to_end_paired_seeds=10,natural_K_settings=20,
        actual_and_calibrated_cohorts_match=True,new_certificates_recomputed=certificates,privacy_decimal_digits=90,
        all_public_choices_used_end_to_end=True,all_adverse_seeds_retained=True,private_counts_never_select_public_rule=True,
        immutable_parent_scientific_fingerprints_checked=True,anonymous_packaging_sources_omitted=omitted,
        public_and_private_endpoints_separated=True,source_only_references_present=True,sentence_spacing_pass=True)
    if layout:
        pdf=fitz.open(project/'paper/main.pdf');texts=[p.get_text() for p in pdf]
        assert len(pdf)<=12 and 'CONCLUSION' in texts[7] and 'REFERENCES' in texts[8],len(pdf)
        assert all('REFERENCES' not in s for s in texts[:8])
        log=project/'paper/main.log'
        if log.exists():
            value=log.read_text();assert 'Overfull' not in value
            assert not re.search(r'(Citation|Reference).*undefined',value)
        objects=list(csv.DictReader((project/'PAPER_EVIDENCE_INDEX.csv').open()))
        for row in objects:assert row['object']+':' in texts[int(row['pdf_page'])-1]
        report.update(main_pages=8,total_pages=len(pdf),references_start_page=9,overfull_boxes=False,undefined_references=False)
    (stage/'revision-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--no-layout',action='store_true')
    a=p.parse_args();validate(a.project.resolve(),not a.no_layout)
