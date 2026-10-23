"""Build, verify and independently anonymize the counted-label revision."""
import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import fitz
from common import ROOT,sha

REPO=ROOT.parents[1]

def sync(project):
    stage=project/'experiments'/ROOT.name;stage.mkdir(parents=True,exist_ok=True)
    for directory in ['src','results','diagnostics','analysis','public','logs','confirmation','superseded-head-calibration','superseded-joint-numerical-replay']:
        shutil.copytree(ROOT/directory,stage/directory,dirs_exist_ok=True,symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.partial','vendor'))
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ['.json','.txt']:shutil.copy2(path,stage/path.name)
    for folder in ['inputs','src']:(stage/folder).mkdir(exist_ok=True)
    for dataset in ['cifar10','mnist','web','femnist62']:
        path=stage/'inputs'/(dataset+'.npz')
        if not path.exists():path.symlink_to('../../release-granularity-20261019/inputs/femnist62.npz' if dataset=='femnist62' else '../../review-evidence-20261012/inputs/'+dataset+'.npz')
    shutil.copytree(ROOT/'inputs/partitions',stage/'inputs/partitions',dirs_exist_ok=True)
    vendor=stage/'src/vendor'
    if not vendor.exists():vendor.symlink_to('../../review-evidence-20261012/src/vendor',target_is_directory=True)

def build_report(project):
    paper=project/'paper';log=(paper/'main.log').read_text();assert 'Overfull' not in log
    assert not re.search(r'(Citation|Reference).*undefined',log)
    pdf=fitz.open(paper/'main.pdf');texts=[p.get_text() for p in pdf]
    assert len(pdf)<=12 and 'CONCLUSION' in texts[7] and 'REFERENCES' in texts[8]
    value=dict(pdf_sha256=sha(paper/'main.pdf'),main_pages=8,total_pages=len(pdf),references_start_page=9,
        overfull_boxes=False,undefined_references=False,
        tex_source_sha256={p.name:sha(p) for p in paper.glob('*.tex')},bibliography_sha256=sha(paper/'refs.bib'))
    (paper/'BUILD_REPORT.json').write_text(json.dumps(value,indent=2)+'\n');return value

def manifest(project):
    stage=project/'experiments'/ROOT.name;parent=json.loads((stage/'PARENT-MANIFEST.json').read_text())
    scientific={name:sha(project/name) for name in parent['scientific_files'] if (project/name).is_file()}
    for directory in ['src','results','diagnostics','analysis','public','inputs/partitions','confirmation','superseded-head-calibration','superseded-joint-numerical-replay']:
        for path in (stage/directory).rglob('*'):
            if path.is_file() and not path.is_symlink() and 'vendor' not in path.parts and path.suffix in ['.py','.cpp','.so','.json','.npz']:
                scientific[str(path.relative_to(project))]=sha(path)
    for path in stage.iterdir():
        if path.suffix not in ['.json','.txt'] or not path.is_file():continue
        if path.name in ['summary.json','revision-validation.json'] or 'STATUS' in path.name or 'COMPLETE' in path.name:continue
        scientific[str(path.relative_to(project))]=sha(path)
    files=[project/name for name in ['README.md','reproduce.py','EXPERIMENT_INDEX.csv','LABEL_BUDGET_INDEX.csv',
        'LABEL_COHORT_INDEX.csv','JOINT_RELEASE_CONTROL_INDEX.csv','SINGLE_RELEASE_READOUT_INDEX.csv','SINGLE_RELEASE_SEED_RESULTS.csv','PAPER_EVIDENCE_INDEX.csv']]
    files+=list((project/'paper').glob('*.tex'))+[project/'paper/main.pdf',project/'paper/refs.bib',
        project/'paper/frontier-data.json',project/'paper/joint-data.json',project/'paper/BUILD_REPORT.json']
    files+=list((project/'paper/figures').glob('*.pdf'))+[project/'paper/figures/frontier-figure-data.json',project/'paper/figures/joint-figure-data.json']
    value=dict(revision=ROOT.name,preserved_primary_conditions=2336,new_single_release_conditions=1090,
        public_students=640,label_sets=4224,crossed_cases=75480,matched_control_cases=150960,scientific_files=scientific,
        presentation_files={str(p.relative_to(project)):sha(p) for p in files})
    (project/'RECORD_MANIFEST.json').write_text(json.dumps(value,indent=2)+'\n')

def run(project,mode,log):
    with log.open('w') as stream:subprocess.run([sys.executable,str(project/'reproduce.py'),mode],stdout=stream,stderr=subprocess.STDOUT,check=True)

def archive(project,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(project.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            relative=p.relative_to(project)
            if any(n in relative.parts for n in ['__pycache__','checkpoints','reproduction-output','validation-preview','raw','deps']):continue
            if p.suffix in ['.pyc','.png'] or p.name=='core':continue
            if relative.parts[0]=='paper' and p.name in ['main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt']:continue
            z.write(p,Path('sdd-www2027')/relative)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None

def main(project):
    project=project.resolve();sync(project);layout=build_report(project)
    stage=project/'experiments'/ROOT.name
    subprocess.run([sys.executable,str(stage/'src/build_indexes.py'),'--project',str(project)],check=True)
    manifest(project);run(project,'verify',ROOT/'logs/full-verify.log')
    print('Full evidence, generated objects and layout verified.',flush=True)
    anon=project.parent/'anonymous-supplement/sdd-www2027'
    if anon.exists():shutil.rmtree(anon)
    omitted={'historical','validation-experiments','validation-preview','reproduction-output','__pycache__','checkpoints'}
    def ignore(folder,names):
        return [n for n in names if n in omitted or n.startswith('REVIEW_INPUT') or n.endswith('DECISIONS.txt')
            or n.startswith('REVISION_NOTES') or n.endswith('.pyc') or n=='finalize_revision.py' or n.startswith('preview-page-') or n=='core']
    shutil.copytree(project,anon,symlinks=True,ignore=ignore)
    extensions={'.py','.json','.txt','.md','.tex','.log','.csv','.cpp','.h','.hpp','.c','.cmake'};redacted=[]
    for p in anon.rglob('*'):
        if not p.is_file() or p.is_symlink() or p.suffix not in extensions or 'vendor' in p.parts:continue
        before=p.read_text();after=before.replace(str(REPO),'.').replace('/home/xbw/.local/bin/tectonic','tectonic').replace('/home/xbw','<USER_HOME>')
        if before!=after:p.write_text(after);redacted.append(str(p.relative_to(anon)))
    library=project/'experiments/review-round3-20261013/anonymous-sampler/exact_dg.so'
    for name in ['review-evidence-20261012','review-round2-20261013','release-granularity-20261019','release-selector-20261019',ROOT.name]:
        shutil.copy2(library,anon/'experiments'/name/'src/exact_dg.so')
    for p in (anon/'paper').glob('main.*'):
        if p.suffix not in ['.tex','.pdf']:p.unlink()
    readme=(anon/'README.md').read_text().replace('# SDD WWW 2027: public label budgets and single-release supervision','# Anonymous SDD supplementary artifact')
    readme=readme.replace('The full local archive retains earlier drafts and delivery provenance.\n','');(anon/'README.md').write_text(readme)
    info=dict(machine_path_files_redacted=redacted,numerical_records_unchanged=True,
        exact_execution_source_snapshots_retained=True,source_fingerprints_are_provenance_not_author_identifiers=True,
        sampler_same_source_and_distribution=True,known_identity_marker_scan='pending')
    (anon/'ANONYMIZATION.json').write_text(json.dumps(info,indent=2)+'\n')
    for p in anon.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in extensions|{'.so','.pdf'}:
            value=p.read_bytes().lower();assert b'xbw' not in value and b'/root/autodl-tmp/' not in value,p
    info['known_identity_marker_scan']='passed';(anon/'ANONYMIZATION.json').write_text(json.dumps(info,indent=2)+'\n')
    manifest(anon);run(anon,'verify',ROOT/'logs/anonymous-verify.log')
    print('Anonymous numerical regeneration passed.',flush=True)
    run(anon,'smoke',ROOT/'logs/anonymous-smoke.log');run(anon,'verify',ROOT/'logs/anonymous-post-smoke-verify.log')
    print('Anonymous fresh-cohort/public-pipeline reconstruction passed.',flush=True)
    report=dict(**layout,immutable_parent_records_checked=True,exact_privacy_certificates=1090,
        full_verification_passed=True,anonymous_verification_passed=True,anonymous_smoke_passed=True,
        public_label_sets=4224,crossed_label_cohort_cases=75480,matched_control_cases=150960,public_students=640,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_scope='local artifact checks')
    for target in [project,anon]:
        (target/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        folder=target/'experiments'/ROOT.name/'logs';folder.mkdir(exist_ok=True)
        for name in ['full-verify.log','anonymous-verify.log','anonymous-smoke.log','anonymous-post-smoke-verify.log']:
            value=(ROOT/'logs'/name).read_text()
            if target==anon:value=value.replace(str(REPO),'.')
            (folder/name).write_text(value)
    stem='sdd-www2027-paper-3fig-compiled-8pages-frontier-20261019'
    pdf=REPO/(stem+'.pdf');full=REPO/(stem+'.zip');anonymous=REPO/(stem+'-anonymous.zip')
    shutil.copy2(project/'paper/main.pdf',pdf);archive(project,full);archive(anon,anonymous)
    clean=project.parent/'clean-extract-check'
    if clean.exists():shutil.rmtree(clean)
    clean.mkdir()
    with zipfile.ZipFile(anonymous) as z:z.extractall(clean)
    run(clean/'sdd-www2027','verify',ROOT/'logs/anonymous-clean-extract-verify.log')
    print('Clean extracted anonymous ZIP verified independently.',flush=True)
    receipt=dict(pdf=pdf.name,pdf_sha256=sha(pdf),full_archive=full.name,full_archive_sha256=sha(full),
        anonymous_archive=anonymous.name,anonymous_archive_sha256=sha(anonymous),main_pages=8,total_pages=layout['total_pages'],
        prior_primary_records=2336,new_single_release_conditions=1090,actual_cohorts=90,private_teacher_states=150,
        public_students=640,label_sets=4224,crossed_cases=75480,matched_control_cases=150960,all_figures_and_tables_regenerated=True,
        anonymous_reconstruction_passed=True,anonymous_clean_extract_verification_passed=True,known_identity_marker_scan_passed=True,zip_crc_checks_passed=True)
    (REPO/(stem+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);main(p.parse_args().project)
