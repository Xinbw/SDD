"""Package the reviewed round-two paper and verifiable experiment records."""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import fitz

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(project):
    project=project.resolve();paper=project/'paper'
    subprocess.run([sys.executable,str(ROOT/'src/validate_round2.py'),'--paper',str(paper)],check=True)
    doc=fitz.open(paper/'main.pdf');texts=[p.get_text() for p in doc]
    assert len(doc)==12
    assert 'CONCLUSION' in texts[7] and 'certificates make that compact release auditable.' in texts[7].replace('\n',' ')
    assert all('REFERENCES' not in s for s in texts[:8]) and 'REFERENCES' in texts[8]
    log=(paper/'main.log').read_text()
    assert 'Overfull' not in log and not re.search(r'(Citation|Reference).*undefined',log)
    assert not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]', '\n'.join(texts[:8]))
    parent=ROOT.parent/'review-evidence-20261012'
    parent_dst=project/'experiments'/parent.name
    plan=json.loads((parent/'plan.json').read_text())
    for name,digest in plan['source_hashes'].items():assert sha(REPO/name)==digest
    assert sha(REPO/plan['parent_archive'])==plan['parent_archive_sha256']
    original_count=0
    for p in (parent/'results').glob('*.json'):
        assert sha(parent_dst/'results'/p.name)==sha(p);original_count+=1
    assert original_count==1086
    preserved=0
    with zipfile.ZipFile(REPO/plan['parent_archive']) as old:
        for name in old.namelist():
            if '/validation-experiments/' not in name or name.endswith('/'):continue
            relative=name.split('/validation-experiments/',1)[1]
            assert (project/'validation-experiments'/relative).read_bytes()==old.read(name)
            preserved+=1
    assert preserved==94
    dst=project/'experiments'/ROOT.name;dst.mkdir(parents=True,exist_ok=True)
    for name in ['src','results','logs','privacy','public-selection']:
        shutil.copytree(ROOT/name,dst/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for p in ROOT.iterdir():
        if p.is_file():shutil.copy2(p,dst/p.name)
    (dst/'inputs').mkdir(exist_ok=True)
    # Feature bundles are included once in the parent tree; the helper makes links.
    (dst/'inputs/PREPARE.txt').write_text('Run python ../src/prepare_replay.py from this directory, or the command in the project README.\n')
    environment={}
    for module in ['torch','numpy','scipy','sklearn','mpmath']:
        m=__import__(module);environment[module]=m.__version__
    environment['python']=sys.version
    environment['gpu']=__import__('torch').cuda.get_device_name()
    (dst/'ENVIRONMENT.json').write_text(json.dumps(environment,indent=2)+'\n')
    summary=json.loads((ROOT/'round2-summary.json').read_text())
    validation={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'main_pages':8,'total_pages':12,'conclusion_page':8,'references_start_page':9,'active_figures':3,
        'overfull_boxes':False,'undefined_references_or_citations':False,'sentence_spacing_pass':True,
        'parent_result_records_preserved':original_count,'new_condition_and_replay_records':500,
        'all_result_records':1586,'historical_validation_files_preserved':preserved,
        'original_training_sources_and_archives_unchanged':True,
        'round2_validation':json.loads((ROOT/'round2-validation.json').read_text()),
        'round2_high_precision':json.loads((ROOT/'privacy/high-precision-validation.json').read_text()),
        'parent_validation':json.loads((project/'validation.json').read_text()),
        'main_pdf_sha256':sha(paper/'main.pdf'),
        'generated_hashes':{p.name:sha(p) for p in paper.glob('tab-*.tex')},
        'figure_hashes':{p.name:sha(p) for p in (paper/'figures').glob('*.pdf')},
        'official_format':'sigconf,anonymous,review; 8 main pages and 12 total',
        'official_requirements_url':'https://www2027.thewebconf.org/research-track-papers/'}
    (project/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (dst/'FINAL-STATUS.json').write_text(json.dumps({'all_experiments_complete':True,'paper_complete':True,
        'new_recorded_conditions':500,'parent_records':1086,'summary':summary},indent=2)+'\n')
    stem='sdd-www2027-paper-3fig-compiled-8pages-round2-20261013'
    pdf=REPO/(stem+'.pdf');archive=REPO/(stem+'.zip')
    shutil.copy2(paper/'main.pdf',pdf)
    build_names={'main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt'}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for p in sorted(project.rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(project)
            if '__pycache__' in relative.parts or p.suffix=='.pyc' or relative.parts[0]=='validation-preview':continue
            if relative.parts[0]=='paper' and p.name in build_names:continue
            out.write(p,Path('sdd-www2027')/relative)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read('sdd-www2027/paper/main.pdf')==pdf.read_bytes()
        names=z.namelist()
        assert sum('/results/' in n and n.endswith('.json') for n in names)==1586
        for name in ['cifar10.npz','mnist.npz','web.npz']:
            assert f'sdd-www2027/experiments/review-evidence-20261012/inputs/{name}' in names
    receipt={'pdf':pdf.name,'pdf_sha256':sha(pdf),'zip':archive.name,'zip_sha256':sha(archive),
        'zip_bytes':archive.stat().st_size,'main_pages':8,'total_pages':12,
        'all_result_records':1586,'new_recorded_conditions_and_replays':500,'zip_crc_pass':True}
    (REPO/(stem+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    main(a.project)
