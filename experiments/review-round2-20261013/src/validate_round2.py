"""Verify data provenance, matched endpoints, certificates and key claims."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'review-evidence-20261012'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(paper):
    rows=[json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json')]
    assert len(rows)==500,len(rows)
    assert json.loads((ROOT/'ROUND2-COMPLETE.json').read_text())['completed']==400
    assert json.loads((ROOT/'LABEL-FREE-COMPLETE.json').read_text())['completed']==100
    parent=json.loads((ROOT/'PARENT-MANIFEST.json').read_text())
    for name in ['cifar10.npz','mnist.npz','web.npz']:
        sha=parent['inputs'][name]
        assert digest(PARENT/'inputs'/name)==sha
        assert digest(ROOT/'inputs'/name)==sha
    for name,sha in parent['frozen_configs'].items():
        assert digest(ROOT/'public-selection'/name)==sha
        assert digest(PARENT/'public-selection'/name)==sha
    frozen=json.loads((ROOT/'PUBLIC-RESOURCE-FROZEN.json').read_text())
    for ds in ['cifar10','mnist','web']:
        assert frozen['datasets'][ds]['public_training_examples']==(1500 if ds=='web' else 5000)
    assert not frozen['selection_uses_private_data'] and not frozen['selection_uses_final_test']
    assert digest(ROOT/'src/engine.py')==frozen['source_sha256']['engine.py']
    assert digest(ROOT/'src/attribution.py')==frozen['source_sha256']['attribution.py']
    certificates=0;replays=0;max_error=0.;label_free=0
    methods={}
    for r in rows:
        methods[r['method']]=methods.get(r['method'],0)+1
        recalls=np.array(r['per_class_recall'])
        assert len(recalls)==(6 if r['dataset']=='web' else 10)
        assert abs(recalls.mean()-r['balanced_accuracy'])<2e-7
        if r.get('parent_result_file'):
            path=PARENT/'results'/r['parent_result_file'];assert digest(path)==r['parent_result_sha256']
            old=json.loads(path.read_text())
            error=max(abs(r[k]-old[k]) for k in ['accuracy','balanced_accuracy','initial_accuracy','initial_balanced_accuracy'])
            assert error<2e-6
            assert r['population']==old['population']
            assert [q['privacy'].get('sigma_share_integer') for q in r['trace']]==[q['privacy'].get('sigma_share_integer') for q in old['trace']]
            replays+=1;max_error=max(max_error,error)
        if 'trace' in r:
            assert r['actual_clients']==r['calibrated_clients']
            assert len(r['trace'])==3 and r['optimizer_steps_per_round']==60
            for stage in r['trace']:
                cert=stage['privacy']
                parts=[cert['screen'],cert['answers']] if r['method']=='PATE-DI' else [cert]
                for q in parts:
                    assert q['actual_clients']==r['actual_clients']
                    assert q['epsilon_upper']<=r['epsilon']
                    certificates+=1
                assert cert['epsilon_upper']<=r['epsilon']
            if r['method']=='DP-Head':
                endpoint=r['local_endpoint']
                assert endpoint['optimizer_steps']==180 and endpoint['extra_adaptation_steps']==0
                assert abs(np.mean(endpoint['recall'])-endpoint['balanced_accuracy'])<2e-7
        elif r['method'].startswith('Public-only'):
            assert r['private_data_used'] is False and r['private_privacy_cost']==0
            assert r['logical_private_report_coordinates']==0
        else:
            assert r['optimizer_steps']==180 and r['batch_size']==128
            assert r['shared_private_information'] is False
        if 'configuration_rule' in r:
            label_free+=1
            assert r['k']==(6 if r['dataset']=='web' else 10)
            assert r['mode']=='center' and r['public_training_labels']==r['public_configuration_labels']==0
    assert replays==260 and label_free==100
    summary=json.loads((ROOT/'round2-summary.json').read_text())
    cf=summary['label_free'][0]
    assert round(100*cf['SDD_M']['mean'],1)==84.2
    assert round(100*cf['Local_only']['mean'],1)==42.4
    assert [round(100*cf['mean_vs_ct'][k],1) for k in ['mean_gain','ci_low','ci_high']]==[14.4,12.2,16.7]
    assert [round(100*cf['mean_vs_local'][k],1) for k in ['mean_gain','ci_low','ci_high']]==[41.8,38.0,45.6]
    small=summary['attribution'][2]
    assert [round(100*small['sdd_vs_head_local'][k],1) for k in ['mean_gain','ci_low','ci_high']]==[18.4,12.1,24.8]
    web=summary['attribution'][-1]
    assert [round(100*web['sdd_vs_head_local'][k],1) for k in ['mean_gain','ci_low','ci_high']]==[8.1,6.9,9.2]
    source='\n'.join(p.read_text() for p in paper.glob('sec-*.tex'))+(paper/'main.tex').read_text()
    assert source.count('\\includegraphics')==3
    assert not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]',source)
    assert 'tang2022labels' in source and '\\input{tab-round2-attribution}' in source
    assert '$84.2\\%$' in source and '$95.9\\%$/$89.0\\%$' in source and '$68.2\\%$' in source
    # Generated grid tables can be available in the artifact without being printed.
    assert 'tab:new-web' not in re.findall(r'\\ref\{([^}]+)\}',source)
    report={'new_recorded_conditions':500,'private_release_conditions':360,'private_release_stage_certificates':certificates,
        'public_reference_conditions':60,'local_and_public_transfer_conditions':80,'label_free_conditions':100,
        'replayed_original_endpoints':replays,'max_abs_original_endpoint_difference':max_error,
        'all_input_and_frozen_config_hashes_match_parent':True,'public_selector_omits_private_and_test_arrays':True,
        'public_training_and_validation_selection_source_unchanged_since_freeze':True,
        'all_recalls_match_balanced_accuracy':True,'dp_head_local_extra_steps':0,
        'all_private_budgets_pass':True,'methods':methods,'manuscript_key_numbers_verified':True,
        'source_sentence_spacing_pass':True,'active_figures':3}
    (ROOT/'round2-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);args=p.parse_args()
    main(args.paper)
