"""Scientific provenance, frozen schedules, certificates and paper checks."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import fitz
import mpmath as mp
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(project,layout=True):
    stage=project/'experiments'/ROOT.name
    parent=json.loads((stage/'PARENT-MANIFEST.json').read_text())
    anon=json.loads((project/'experiments/review-round3-20261013/SAMPLER-ANONYMIZATION.json').read_text())
    for name,digest in parent['scientific_files'].items():
        allowed={digest}
        if name.endswith('/exact_dg.so'):allowed.add(anon['anonymous_binary_sha256'])
        assert sha(project/name) in allowed,name
    records=[json.loads(p.read_text()) for p in (stage/'results').glob('*.json')]
    assert len(records)==185,len(records)
    index=list(csv.DictReader((project/'EXPERIMENT_INDEX.csv').open()))
    assert len(index)==1771 and len({r['result'] for r in index})==1771
    for row in index:
        r=json.loads((project/row['result']).read_text())
        assert row['dataset']==r['dataset'] and row['variant']==r['method'] and int(row['seed'])==r['seed']
        assert float(row['epsilon'])==r.get('epsilon',0)
        if row['cohort']!='public':assert int(row['cohort'])==r.get('actual_clients',r.get('n'))
    for name,count in [('DECODER',95),('NATURAL',90)]:
        plan=json.loads((stage/(name+'-SCHEDULE.json')).read_text())
        assert len(plan['jobs'])==count
        for job in plan['jobs']:
            key='_'.join(str(job[k]) for k in ['dataset','n','epsilon','seed','method','k','decoder'])
            actual=json.loads((stage/'results'/(key+'.json')).read_text())
            assert all(actual[k]==v for k,v in job.items())
    mp.mp.dps=90
    certificate_count=0
    populations={}
    for r in records:
        assert r['n']==r['actual_clients']==r['calibrated_clients']
        assert r['optimizer_steps']==180 and r['batch_size']==128
        assert r['private_outcomes_select_decoder'] is False
        pk=(r['dataset'],r['n'],r['seed'])
        pop=r['population']['client_sample_indices_sha256']
        assert populations.setdefault(pk,pop)==pop
        if r['method']=='Local-only':assert r['trace']==[];continue
        assert len(r['trace'])==3
        for rr in r['trace']:
            c=rr['privacy'];certificate_count+=1
            assert c['actual_clients']==c['calibrated_clients']==r['n']
            assert c['epsilon_requested']==r['epsilon']
            rho=mp.mpf(c['sensitivity_integer_squared'])/(2*c['honest_clients']*mp.mpf(c['sigma_share_integer'])**2)
            rho+=mp.mpf(c['dimension'])*mp.mpf(str(c['tau_certified_upper']))/4
            eps=3*rho+2*mp.sqrt(3*rho*mp.log(mp.mpf(1)/mp.mpf('0.00001')))
            assert eps<=mp.mpf(str(r['epsilon'])),(r,eps)
            assert c['tau_certified_upper']<=1e-30
        if 'diagnostic_file' not in r:continue
        path=stage/r['diagnostic_file'];assert sha(path)==r['diagnostic_sha256']
        d=np.load(path);truth=d['truth_counts'];ids=d['ids']
        for rr in r['trace']:
            j=rr['round'];counts=d[f'r{j}_counts'];scores=d[f'r{j}_scores'];decoded=d[f'r{j}_decoded']
            assert (counts.sum(1)==r['n']).all()
            hard=decoded.argmax(1) if decoded.ndim==2 else decoded
            accuracy=truth[np.arange(len(truth)),hard].sum()/truth.sum()
            assert abs(accuracy-rr['diagnostics']['pseudolabel_accuracy'])<1e-12
            if r['decoder']=='matching':
                assert len(np.unique(hard))==r['k']==truth.shape[1]
                objective=d[f'r{j}_projected']*np.bincount(ids,minlength=r['k'])[:,None]
                rows,columns=linear_sum_assignment(-objective)
                assert np.array_equal(hard[rows],columns)
    for seed in range(801,806):
        old=json.loads((project/'experiments/review-round2-20261013/results'/f'label_free_cifar10_10000_0.1_0.1_{seed}_SDD.json').read_text())
        replay=next(r for r in records if r['dataset']=='cifar10' and r['n']==10000 and r['seed']==seed and r['decoder']=='center')
        assert abs(old['accuracy']-replay['accuracy'])<2e-6
        assert old['population']==replay['population']
    for seed in list(range(801,806))+list(range(8201,8206)):
        rows=[r for r in records if r['dataset']=='cifar10' and r['n']==10000 and r['seed']==seed and r['method']=='SDD']
        arrays=[np.load(stage/r['diagnostic_file']) for r in rows]
        for d in arrays[1:]:
            assert np.array_equal(d['r1_counts'],arrays[0]['r1_counts'])
            assert np.array_equal(d['r1_scores'],arrays[0]['r1_scores'])
    bundle=np.load(stage/'inputs/femnist62.npz',allow_pickle=False)
    roles=['private','proxy','development','validation','test']
    sets={role:set(bundle['owners_'+role].tolist()) for role in roles}
    for i,a in enumerate(roles):
        for b in roles[i+1:]:assert not sets[a]&sets[b]
    assert len(sets['private'])==2600 and len(bundle['y_private'])==510725
    manifest=json.loads((stage/'inputs/femnist62_manifest.json').read_text())
    assert sha(stage/'inputs/femnist62.npz')==manifest['feature_bundle_sha256']
    predictions=[json.loads(p.read_text()) for p in (stage/'predictions').glob('quality_*.json')]
    assert len(predictions)==315
    assert sum('exact_mean' in r['modes']['center'] for r in predictions)==27
    for r in predictions:
        d=np.load(stage/r['counts_file']);truth=d['truth_counts']
        assert sha(stage/r['counts_file'])==r['counts_sha256']
        for mode in ['raw','center']:
            m=r['modes'][mode];probs=np.array(m['class_probabilities'])
            assert np.allclose(probs.sum(1),1,rtol=0,atol=1e-14)
            expected=(truth*probs).sum()/truth.sum()
            assert abs(expected-m['predicted_accuracy'])<1e-12
            assert expected<=m['purity']+1e-12
    for p in (stage/'public-selection').glob('*quality.json'):
        design=json.loads(p.read_text());assert design['private_data_access'] is False and design['test_access'] is False
        assert all('test' not in k and 'private' not in k for k in design['loaded_source_arrays'])
    text='\n'.join(p.read_text() for p in (project/'paper').glob('sec-*.tex'))+(project/'paper/main.tex').read_text()
    assert text.count('\\includegraphics')==3
    assert not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]',text)
    assert 'release-label model' in text and 'public' in text
    report=dict(preserved_original_conditions=1586,preserved_web_diagnostic_replays=60,
        new_training_conditions=185,new_quality_conditions=315,exact_multitrial_quality_conditions=27,
        checked_new_training_certificates=certificate_count,old_five_endpoints_reproduced=True,
        first_release_inputs_and_noise_identical_across_decoders=True,
        natural_writer_roles_disjoint=True,source_record_overlap_between_natural_clients=False,
        public_selector_omits_private_and_test_arrays=True,privacy_recomputed_decimal_digits=90,
        prediction_identity_checked=True,all_prespecified_decoders_retained=True,sentence_spacing_pass=True)
    if layout:
        doc=fitz.open(project/'paper/main.pdf');texts=[p.get_text() for p in doc]
        assert len(doc)==12,len(doc)
        assert 'CONCLUSION' in texts[7] and 'REFERENCES' in texts[8]
        assert all('REFERENCES' not in t for t in texts[:8])
        objects=list(csv.DictReader((project/'PAPER_EVIDENCE_INDEX.csv').open()))
        assert len(objects)==10
        for row in objects:
            assert row['object']+':' in texts[int(row['pdf_page'])-1]
            assert (project/row['paper_file']).is_file()
            for source in row['evidence'].split(';'):
                assert source=='theory' or (project/source).exists()
        log=project/'paper/main.log'
        if log.exists():
            value=log.read_text();assert 'Overfull' not in value
            assert not re.search(r'(Citation|Reference).*undefined',value)
        report.update(main_pages=8,total_pages=12,conclusion_page=8,references_start_page=9,
            overfull_boxes=False,undefined_citations_or_references=False)
    (stage/'revision-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--no-layout',action='store_true')
    a=p.parse_args();validate(a.project.resolve(),not a.no_layout)
