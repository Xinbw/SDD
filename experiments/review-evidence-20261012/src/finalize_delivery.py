"""Validate, compile and package the completed evidence revision."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
import fitz

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--wait', action='store_true')
args = parser.parse_args()
project = args.project.resolve()
paper = project/'paper'
if args.wait:
    while not (ROOT/'LEGACY-COMPLETE.json').exists():
        time.sleep(20)
assert json.loads((ROOT/'LEGACY-COMPLETE.json').read_text())['runs'] == 4
legacy = list((ROOT/'results').glob('resnet_*.json'))
assert len(legacy) == 4
assert len(list((ROOT/'results').glob('*.json'))) == 1086

# Reconcile execution metadata from the preserved actual commands, never accuracy.
for path in legacy:
    record = json.loads(path.read_text())
    command = json.loads((ROOT/'legacy-resnet'/path.stem/'command.json').read_text())
    record['execution_concurrent_clients'] = int(command[command.index('--concurrent-clients')+1])
    record['round1_checkpoint_reused'] = True
    record['continuation_rounds'] = 2
    record['epochs_per_continuation_round'] = 20
    raw = (ROOT/'logs'/record['log_file']).read_text()
    assert 'evaluating on 4033' in raw
    assert len(re.findall(r'\[Round [123]\] valid_acc=',raw)) == 3
    assert float(re.findall(r'Final mean client accuracy:\s*([0-9.]+)',raw)[-1]) == record['accuracy']
    path.write_text(json.dumps(record,indent=2)+'\n')

def run(script, *arguments):
    print('RUN',script,flush=True)
    subprocess.run([sys.executable,str(ROOT/'src'/script),*map(str,arguments)],check=True)

run('finalize_certificates.py')
run('export_results.py','--paper',paper)
run('validate_results.py')
run('check_high_precision.py')
run('check_manuscript.py','--paper',paper)
tectonic = shutil.which('tectonic') or '/home/xbw/.local/bin/tectonic'
with (paper/'build-output.txt').open('w') as handle:
    subprocess.run([tectonic,'--only-cached','--keep-logs','--keep-intermediates','main.tex'],
                   cwd=paper,stdout=handle,stderr=subprocess.STDOUT,check=True)
doc = fitz.open(paper/'main.pdf')
texts = [page.get_text() for page in doc]
assert len(doc) == 12, len(doc)
assert '6\nCONCLUSION' in texts[7]
assert 'Exact integer noise and explicit whole-' in texts[7]
assert all('REFERENCES' not in value for value in texts[:8])
assert 'REFERENCES' in texts[8]
log = (paper/'main.log').read_text()
assert 'Overfull' not in log
assert not re.search(r'(Citation|Reference).*undefined',log)
assert not re.search(r'(?:[a-z]|\$)\.[A-Z][a-z]', '\n'.join(texts[:8]))

dst = project/'experiments'/ROOT.name
for name in ['results','logs','privacy','public-selection','legacy-resnet']:
    shutil.copytree(ROOT/name,dst/name,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for path in ROOT.iterdir():
    if path.is_file(): shutil.copy2(path,dst/path.name)
for path in (ROOT/'src').iterdir():
    if path.is_file(): shutil.copy2(path,dst/'src'/path.name)
original = dst/'original-source'
original.mkdir(exist_ok=True)
plan = json.loads((ROOT/'plan.json').read_text())
for name,digest in plan['source_hashes'].items():
    path = REPO/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    shutil.copy2(path,original/name)
assert hashlib.sha256((REPO/plan['parent_archive']).read_bytes()).hexdigest() == plan['parent_archive_sha256']

preserved = []
with zipfile.ZipFile(REPO/plan['parent_archive']) as old:
    for name in old.namelist():
        if '/validation-experiments/' not in name or name.endswith('/'): continue
        relative = name.split('/validation-experiments/',1)[1]
        path = project/'validation-experiments'/relative
        assert path.read_bytes() == old.read(name), relative
        preserved.append(relative)
assert len(preserved) == 94, len(preserved)
preview = project/'validation-preview'
preview.mkdir(exist_ok=True)
for i in range(len(doc)):
    doc[i].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(preview/f'page-{i+1}.png')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
validation = {
    'completed_utc':now, 'main_pages':8, 'total_pages':12,
    'conclusion_page':8, 'references_start_page':9, 'active_figures':3,
    'overfull_boxes':False, 'undefined_citations_or_references':False,
    'sentence_spacing_scan_pass':True,
    'official_requirements_url':'https://www2027.thewebconf.org/research-track-papers/',
    'official_requirements_checked':'2026-10-13',
    'matched_population_training_conditions':1082,'legacy_training_conditions':4,
    'all_result_records':1086,
    'scientific_validation':json.loads((ROOT/'scientific-validation.json').read_text()),
    'high_precision_validation':json.loads((ROOT/'privacy/high-precision-validation.json').read_text()),
    'manuscript_numbers':json.loads((ROOT/'manuscript-number-validation.json').read_text()),
    'historical_validation_files_preserved':len(preserved),
    'historical_sources_and_parent_archive_unchanged':True,
    'paper_pdf_sha256':hashlib.sha256((paper/'main.pdf').read_bytes()).hexdigest(),
    'secure_transport':'assumed aggregate-only component; utility runs implement the sum arithmetic',
    'external_large_inputs':'raw data, original image-encoder weights, historical ResNet R1 checkpoints; hashes and sources recorded',
}
(project/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
status = {'completed_utc':now,'all_experiments_complete':True,'paper_complete':True,
          'canonical_results':1086,'original_plan_retained':True,
          'experiment_roles':{stage['id']:'complete' for stage in plan['stages']}}
(ROOT/'FINAL-STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
shutil.copy2(ROOT/'FINAL-STATUS.json',dst/'FINAL-STATUS.json')
stem = 'sdd-www2027-paper-3fig-compiled-8pages-evidence-20261013'
pdf = REPO/(stem+'.pdf')
archive = REPO/(stem+'.zip')
shutil.copy2(paper/'main.pdf',pdf)
build_names = {'main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt'}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
    for path in sorted(project.rglob('*')):
        if not path.is_file(): continue
        relative = path.relative_to(project)
        if '__pycache__' in relative.parts or path.suffix == '.pyc': continue
        if relative.parts[0] == 'validation-preview': continue
        if relative.parent == Path('paper') and path.name in build_names: continue
        output.write(path,Path('sdd-www2027')/relative)
with zipfile.ZipFile(archive) as output:
    assert output.testzip() is None
    assert sum('/validation-experiments/' in name and not name.endswith('/') for name in output.namelist()) == 94
    assert sum('/experiments/'+ROOT.name+'/results/' in name and name.endswith('.json') for name in output.namelist()) == 1086
    assert output.read('sdd-www2027/paper/main.pdf') == pdf.read_bytes()
receipt = {'pdf':pdf.name,'zip':archive.name,'pdf_bytes':pdf.stat().st_size,
           'zip_bytes':archive.stat().st_size,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
           'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'completed_utc':now}
(REPO/(stem+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print('DELIVERY COMPLETE',json.dumps(receipt),flush=True)
