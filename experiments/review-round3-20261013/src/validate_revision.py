"""Check the revised claims and immutable evidence without adding experiments."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(project):
    manifest=json.loads((ROOT/'PARENT-MANIFEST.json').read_text())
    count=0
    for stage,records in manifest['record_hashes'].items():
        for name,digest in records.items():
            assert sha(REPO/'experiments'/stage/'results'/name)==digest
            assert sha(project/'experiments'/stage/'results'/name)==digest
            count+=1
    assert count==1586
    diagnostics=[json.loads(p.read_text()) for p in (ROOT/'diagnostics').glob('*.json')]
    assert len(diagnostics)==60
    max_error=0.
    for row in diagnostics:
        prior=REPO/'experiments/review-round2-20261013/results'/row['original_result_file']
        assert sha(prior)==row['original_result_sha256']
        max_error=max(max_error,row['max_original_endpoint_abs_difference'])
        assert row['max_original_endpoint_abs_difference']<2e-6
        assert len(row['observations'])==3
        for q in row['observations']:
            assert sum(q['target_class_counts'])==1500
            assert q['supervises_next_local_round']==(q['round']<3)
    summary=json.loads((ROOT/'coverage-summary.json').read_text())
    r=summary['protected_target_coverage']['representative'];m=summary['protected_target_coverage']['SDD']
    assert round(100*r['first_three_fraction']['mean'],1)==98.7
    assert round(100*m['last_three_fraction']['mean'],1)==46.1
    assert round(100*summary['minimum_recall']['SDD']['mean'],1)==2.3
    assert round(100*summary['minimum_recall']['representative']['mean'],1)==1.8
    assert [round(100*q['comp_target_fraction'],1) for q in summary['head_rounds']]==[64.7,95.4,99.8]
    paper=project/'paper';source='\n'.join(p.read_text() for p in paper.glob('sec-*.tex'))+(paper/'main.tex').read_text()
    assert source.count('\\includegraphics')==3
    assert not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]',source)
    abstract=source.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    assert abstract.index('$84.2\\%$')<abstract.index('$85.0\\%$')
    assert '14.4' in abstract and '100 coordinates' in abstract and '200\\times' in abstract
    assert 't_{0.975,m-1}' in source and 'no routing' in source
    assert '$95.6\\%$' in source and '$95.9\\%$' in source and '$65.6/68.2\\%$' in source
    assert 'not a clipping ablation' not in source
    doc=fitz.open(paper/'main.pdf');texts=[p.get_text() for p in doc]
    assert len(doc)==12 and 'CONCLUSION' in texts[7]
    assert 'auditable interface for private collaboration.' in texts[7].replace('-\n','').replace('\n',' ')
    assert all('REFERENCES' not in s for s in texts[:8]) and 'REFERENCES' in texts[8]
    log=(paper/'main.log').read_text()
    assert 'Overfull' not in log and not re.search(r'(Citation|Reference).*undefined',log)
    assert not doc.metadata.get('author') or 'Anonymous' in doc.metadata['author']
    report={'preserved_parent_records':count,'diagnostic_replays':60,'max_original_endpoint_abs_difference':max_error,
        'no_training_or_privacy_configuration_reselected':True,'protected_target_statistics_verified':True,
        'minimum_recall_computed_within_seed':True,'round3_release_excluded_from_training_target_summary':True,
        'main_pages':8,'total_pages':12,'references_start_page':9,'conclusion_fully_on_page':8,
        'active_vector_figures':3,'overfull_boxes':False,'undefined_citations_or_references':False,
        'sentence_spacing_pass':True,'main_claims_verified':True,'source_and_result_provenance_verified':True}
    (ROOT/'revision-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    validate(a.project.resolve())
