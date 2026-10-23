"""Compile, audit, fingerprint and package the thirteenth ICDE 2027 EAB round (client-weighted targets, FEMNIST role, pilot ledger, print-size figures).

Inherited scientific records are unchanged. New scientific material: the frozen
allocation feasibility study (release-allocation-20261019) and the EAB analyses
(partition ceilings, threshold post-processing). Earlier deliveries are untouched.
"""
import argparse,datetime,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path

STAGE=Path(__file__).resolve().parents[1]
EXP=STAGE.parent;REPO=EXP.parent
NEW_STAGES=['release-allocation-20261019','release-eab-20261020','release-eab2-20261020','release-eab3-20261020','release-verify-teacher-20261020','release-eab4-20261020','release-disjoint-20261020','release-eab5-20261020','release-eab6-20261020','release-eab7-20261020','release-budgetconf-20261021','release-eab8-20261021','release-eab9-20261021','release-eab10-20261022','release-eab11-20261022','release-encoder-20261022','release-femnistconf-20261022','release-eab12-20261022','release-eab13-20261023']
STEM='sdd-icde2027-eab13-paper-6fig-compiled-12pages-eab13-20261023'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sync(project):
    for name in NEW_STAGES:
        target=project/'experiments'/name
        if target.exists():shutil.rmtree(target)
        shutil.copytree(EXP/name,target,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','logs','*.partial','checkpoints','*_all.npy'))
        (target/'logs').mkdir()
        if name in ['release-verify-teacher-20261020','release-disjoint-20261020','release-budgetconf-20261021','release-encoder-20261022']:(target/'checkpoints').mkdir()

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
    presentation=[project/'README.md',project/'REVIEWER_GUIDE.md',project/'reproduce.py',*project.glob('*.csv'),*paper.glob('*.tex'),paper/'main.pdf',paper/'refs.bib',paper/'BUILD_REPORT.json']
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
    subprocess.run([sys.executable,str(project/'experiments/release-eab-20261020/src/export_eab.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab3-20261020/src/export_eab3.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab4-20261020/src/export_eab4.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab5-20261020/src/export_eab5.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab6-20261020/src/export_eab6.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab7-20261020/src/export_eab7.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab8-20261021/src/export_eab8.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab11-20261022/src/export_eab11.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab12-20261022/src/recheck_tau.py')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab12-20261022/src/export_eab12.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab13-20261023/src/ledger.py')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(project/'experiments/release-eab13-20261023/src/export_eab13.py'),'--paper',str(project/'paper')],check=True,stdout=subprocess.DEVNULL)
    compile_paper(project,logs/'compile.log')
    subprocess.run([sys.executable,str(project/'experiments'/STAGE.name/'src/eab13_layout.py'),'--project',str(project),'--write'],check=True)
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
    check=json.loads((project/'paper/eab3-data.json').read_text())['planning']['decision_check']
    pdf=REPO/(STEM+'.pdf');full=REPO/(STEM+'.zip')
    shutil.copy2(project/'paper/main.pdf',pdf);archive(project,full);print('Archive passed CRC.',flush=True)
    receipt=dict(pdf=pdf.name,pdf_sha256=sha(pdf),full_archive=full.name,full_archive_sha256=sha(full),venue=build['venue'],format=build['format'],
        body_pages=build['body_pages'],references_start_page=build['references_start_page'],total_pages=build['total_pages'],appendix=False,main_figures=build['main_figures'],
        new_scientific_stages=NEW_STAGES,allocation_cases=report['allocation_cases'],allocation_preregistered_go=alloc['go'],
        selection_rule_selected=json.loads((project/'experiments/release-eab2-20261020/analysis/selection-rules.json').read_text())['selected_rule'],selection_rule_status='exploratory: same cohorts, held-out label draws',
        planning_map_max_gap_points=max(abs(c['predicted_accept']-c['recorded_accept']) for r in check.values() for c in r.values()),audit_split='development',verify_teacher=json.loads((project/'paper/eab4-data.json').read_text())['primary'],verify_teacher_cases=6000,disjoint=json.loads((project/'paper/eab5-data.json').read_text())['disjoint_predictions'],disjoint_cases=3000,calculator_cells=24,cost_table_draws='50-99',optimized_allocation_status='retrospective sensitivity analysis (phi chosen with offline test utility)',budgetconf=json.loads((project/'paper/eab8-data.json').read_text())['primary'],budgetconf_predictions=json.loads((project/'paper/eab8-data.json').read_text())['predictions'],budgetconf_cases=54000,encoder=json.loads((project/'paper/eab12-data.json').read_text())['encoder'],encoder_predictions=json.loads((project/'paper/eab12-data.json').read_text())['encoder_predictions'],femnist=json.loads((project/'paper/eab12-data.json').read_text())['femnist'],femnist_predictions=json.loads((project/'paper/eab12-data.json').read_text())['femnist_predictions'],tau_recheck=json.loads((project/'paper/eab12-data.json').read_text())['tau'],eab13_ledger=json.loads((project/'paper/eab13-data.json').read_text())['ledger'],quick_check='python reproduce.py quick',inherited_scientific_records_changed=False,previous_deliveries_unchanged=True,all_figures_and_tables_regenerated=True,zip_crc_checks_passed=True,
        review_mode='single-blind (author block must be completed)',artifact_note='EAB papers require a public reproducible artifact link')
    (REPO/(STEM+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);main(p.parse_args().project)
