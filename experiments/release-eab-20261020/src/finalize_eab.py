"""Compile, audit, fingerprint and package the ICDE 2027 EAB version.

Inherited scientific records are unchanged. New scientific material: the frozen
allocation feasibility study (release-allocation-20261019) and the EAB analyses
(partition ceilings, threshold post-processing). Earlier deliveries are untouched.
"""
import argparse,datetime,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path

STAGE=Path(__file__).resolve().parents[1]
EXP=STAGE.parent;REPO=EXP.parent
NEW_STAGES=['release-allocation-20261019','release-eab-20261020']
STEM='sdd-icde2027-eab-paper-3fig-compiled-12pages-eab-20261020'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sync(project):
    for name in NEW_STAGES:
        target=project/'experiments'/name
        if target.exists():shutil.rmtree(target)
        shutil.copytree(EXP/name,target,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','logs','*.partial'))
        (target/'logs').mkdir()

def compile_paper(project,log):
    binary=shutil.which('tectonic');assert binary
    with log.open('w') as stream:
        subprocess.run([binary,'--keep-logs','--keep-intermediates','main.tex'],cwd=project/'paper',stdout=stream,stderr=subprocess.STDOUT,check=True)

def manifest(project):
    path=project/'RECORD_MANIFEST.json';data=json.loads(path.read_text())
    inherited={k:v for k,v in data['scientific_files'].items() if not any(k.startswith('experiments/'+n+'/') for n in NEW_STAGES)}
    for name,digest in inherited.items():assert sha(project/name)==digest,('scientific file changed',name)
    new={}
    for name in NEW_STAGES:
        stage=project/'experiments'/name
        for p in sorted(stage.rglob('*')):
            if p.is_file() and not p.is_symlink() and 'logs' not in p.relative_to(stage).parts and p.suffix in ['.py','.json']:
                new[str(p.relative_to(project))]=sha(p)
    data['scientific_files']={**inherited,**new}
    paper=project/'paper'
    presentation=[project/'README.md',project/'reproduce.py',*project.glob('*.csv'),*paper.glob('*.tex'),paper/'main.pdf',paper/'refs.bib',paper/'BUILD_REPORT.json']
    presentation+=list(paper.glob('*-data.json'))+list((paper/'figures').glob('*.pdf'))+list((paper/'figures').glob('*-data.json'))
    presentation+=list((project/'experiments/release-icde-20261019/src').glob('*.py'))
    data['presentation_revision']=STAGE.name;data['new_scientific_stages']=NEW_STAGES
    data['presentation_files']={str(p.relative_to(project)):sha(p) for p in sorted(set(presentation))}
    path.write_text(json.dumps(data,indent=2)+'\n')

def archive(project,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(project.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            relative=p.relative_to(project)
            if any(name in relative.parts for name in ['__pycache__','checkpoints','reproduction-output','validation-preview','raw','deps','sources']):continue
            if p.suffix in ['.pyc','.png'] or p.name=='core':continue
            if relative.parts[0]=='paper' and p.name in ['main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt']:continue
            z.write(p,Path(project.name)/relative)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None

def main(project):
    project=project.resolve();logs=STAGE/'logs';logs.mkdir(exist_ok=True)
    for name in ['pdf','zip']:assert not (REPO/(STEM+'.'+name)).exists(),'refusing to overwrite '+STEM
    sync(project)
    subprocess.run([sys.executable,str(project/'experiments'/STAGE.name/'src/export_eab.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    compile_paper(project,logs/'compile.log')
    subprocess.run([sys.executable,str(project/'experiments'/STAGE.name/'src/eab_layout.py'),'--project',str(project),'--write'],check=True)
    build=json.loads((project/'paper/BUILD_REPORT.json').read_text());manifest(project)
    with (logs/'full-verify.log').open('w') as stream:
        subprocess.run([sys.executable,str(project/'reproduce.py'),'verify'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('Full numerical/scientific/EAB-layout verification passed.',flush=True)
    for name in ['compile.log','full-verify.log']:shutil.copy2(logs/name,project/'experiments'/STAGE.name/'logs'/name)
    alloc=json.loads((project/'experiments/release-allocation-20261019/results/allocation-analysis.json').read_text())
    report=dict(**{k:v for k,v in build.items() if k!='tex_source_sha256'},preserved_primary_records=3426,validation_cases=3000,
        allocation_cases=sum(len(json.loads(p.read_text())['rows']) for p in (project/'experiments/release-allocation-20261019/results').glob('alloc_s*.json')),
        allocation_preregistered_go=alloc['go'],inherited_scientific_records_changed=False,full_verification_passed=True,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (project/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    pdf=REPO/(STEM+'.pdf');full=REPO/(STEM+'.zip')
    shutil.copy2(project/'paper/main.pdf',pdf);archive(project,full);print('Archive passed CRC.',flush=True)
    receipt=dict(pdf=pdf.name,pdf_sha256=sha(pdf),full_archive=full.name,full_archive_sha256=sha(full),venue=build['venue'],format=build['format'],
        body_pages=build['body_pages'],references_start_page=build['references_start_page'],total_pages=build['total_pages'],appendix=False,main_figures=build['main_figures'],
        new_scientific_stages=NEW_STAGES,allocation_cases=report['allocation_cases'],allocation_preregistered_go=alloc['go'],
        inherited_scientific_records_changed=False,previous_deliveries_unchanged=True,all_figures_and_tables_regenerated=True,zip_crc_checks_passed=True,
        review_mode='single-blind (author block must be completed)',artifact_note='EAB papers require a public reproducible artifact link')
    (REPO/(STEM+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);main(p.parse_args().project)
