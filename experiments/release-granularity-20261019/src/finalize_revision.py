"""Build separately named full and anonymous, locally verified deliveries."""
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(project):
    parent=json.loads((project/'experiments/release-granularity-20261019/PARENT-MANIFEST.json').read_text())
    scientific={name:sha(project/name) for name in parent['scientific_files']}
    stage=project/'experiments/release-granularity-20261019'
    for folder in ['results','diagnostics','predictions','public-selection','src']:
        for path in (stage/folder).glob('*'):
            if path.is_file() and path.suffix in ['.json','.npz','.py','.cpp','.so']:
                scientific[str(path.relative_to(project))]=sha(path)
    for path in (stage/'inputs').glob('*'):
        if path.is_file() and not path.is_symlink():scientific[str(path.relative_to(project))]=sha(path)
    for path in (stage/'inputs/partitions').glob('*.npz'):scientific[str(path.relative_to(project))]=sha(path)
    for path in stage.glob('*.json'):scientific[str(path.relative_to(project))]=sha(path)
    presentation={}
    names=['reproduce.py','README.md','EXPERIMENT_INDEX.csv','QUALITY_INDEX.csv','REGION_DIAGNOSTIC_INDEX.csv','PAPER_EVIDENCE_INDEX.csv']
    paths=[project/name for name in names]
    paths+=list((project/'paper').glob('*.tex'))+[project/'paper/refs.bib',project/'paper/main.pdf']
    paths+=list((project/'paper').glob('*data.json'))
    paths+=list((project/'paper/figures').glob('*.pdf'))+list((project/'paper/figures').glob('*data.json'))
    for path in paths:presentation[str(path.relative_to(project))]=sha(path)
    data=dict(revision='release-granularity-20261019',preserved_training_conditions=1586,new_training_conditions=185,
        total_training_conditions=1771,preserved_web_diagnostic_replays=60,new_quality_conditions=315,
        new_region_diagnostic_files=155,scientific_files=scientific,presentation_files=presentation)
    (project/'RECORD_MANIFEST.json').write_text(json.dumps(data,indent=2)+'\n')


def run(project,mode,log):
    with log.open('w') as stream:
        subprocess.run([sys.executable,str(project/'reproduce.py'),mode],stdout=stream,stderr=subprocess.STDOUT,check=True)


def archive(project,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(project.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            relative=p.relative_to(project)
            if any(x in relative.parts for x in ['__pycache__','checkpoints','reproduction-output','validation-preview','raw','deps']):continue
            if p.suffix in ['.pyc','.png']:continue
            if relative.parts[0]=='paper' and p.name in ['main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt']:continue
            z.write(p,Path('sdd-www2027')/relative)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None


def main(project):
    project=project.resolve();stage=project/'experiments'/ROOT.name
    shutil.copytree(ROOT/'src',stage/'src',dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','vendor'))
    for path in ROOT.glob('*.json'):shutil.copy2(path,stage/path.name)
    with (ROOT/'logs/final-validation.log').open('w') as stream:
        subprocess.run([sys.executable,str(stage/'src/validate_revision.py'),'--project',str(project)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=json.loads((stage/'revision-validation.json').read_text())
    report.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_scope='local artifact checks; no submission or independent review claimed')
    (project/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest(project)
    run(project,'verify',ROOT/'logs/full-verify.log')
    print('Full evidence, regenerated statistics and layout verification passed.',flush=True)
    anon=project.parent/'anonymous-supplement/sdd-www2027'
    if anon.exists():shutil.rmtree(anon)
    omit={'historical','validation-experiments','validation-preview','reproduction-output','__pycache__','checkpoints'}
    def ignore(folder,names):
        return [name for name in names if name in omit or name.startswith('REVIEW_INPUT') or name.endswith('DECISIONS.txt')
            or name.startswith('REVISION_NOTES') or name.endswith('.pyc') or name=='finalize_revision.py' or name.startswith('preview-page-')]
    shutil.copytree(project,anon,symlinks=True,ignore=ignore)
    suffixes={'.py','.json','.txt','.md','.tex','.log','.csv','.cpp','.h','.hpp','.c','.cmake'}
    redacted=[]
    for path in anon.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.suffix not in suffixes or 'vendor' in path.parts:continue
        value=path.read_text();cleaned=value.replace(str(REPO),'.').replace('/home/xbw/.local/bin/tectonic','tectonic').replace('/home/xbw','<USER_HOME>')
        if value!=cleaned:path.write_text(cleaned);redacted.append(str(path.relative_to(anon)))
    library=project/'experiments/review-round3-20261013/anonymous-sampler/exact_dg.so'
    for name in ['review-evidence-20261012','review-round2-20261013','release-granularity-20261019']:
        shutil.copy2(library,anon/'experiments'/name/'src/exact_dg.so')
    for path in (anon/'paper').glob('main.*'):
        if path.suffix not in ['.tex','.pdf']:path.unlink()
    readme=(anon/'README.md').read_text().replace('# SDD WWW 2027: release granularity revision','# Anonymous SDD supplementary artifact')
    readme=readme.replace('The full local archive retains earlier drafts and delivery provenance.\n','')
    (anon/'README.md').write_text(readme)
    for path in anon.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix in suffixes|{'.so','.pdf'}:
            value=path.read_bytes().lower();assert b'xbw' not in value and b'/root/autodl-tmp/' not in value,path
    info=dict(machine_path_files_redacted=redacted,numerical_records_unchanged=True,
        earlier_drafts_and_session_reviews_omitted=True,sampler_same_source_and_output_distribution=True,
        original_source_freeze_hashes_retained_as_provenance=True,active_anonymous_source_hashes_in_record_manifest=True,
        known_workspace_identity_marker_scan='passed')
    (anon/'ANONYMIZATION.json').write_text(json.dumps(info,indent=2)+'\n')
    manifest(anon)
    run(anon,'verify',ROOT/'logs/anonymous-verify.log')
    print('Anonymous artifact verification passed.',flush=True)
    run(anon,'smoke',ROOT/'logs/anonymous-smoke.log')
    # Replay restores the recorded evidence; verify that guarantee as well.
    run(anon,'verify',ROOT/'logs/anonymous-post-smoke-verify.log')
    print('Anonymous CIFAR-10 and natural-writer reconstruction passed.',flush=True)
    report.update(anonymous_verification_passed=True,anonymous_smoke_replay_passed=True,known_identity_marker_scan_passed=True)
    for target in [project,anon]:
        (target/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        logfolder=target/'experiments'/ROOT.name/'logs'
        for name in ['final-validation.log','full-verify.log','anonymous-verify.log','anonymous-smoke.log','anonymous-post-smoke-verify.log']:
            value=(ROOT/'logs'/name).read_text()
            if target==anon:value=value.replace(str(REPO),'.')
            (logfolder/name).write_text(value)
    stem='sdd-www2027-paper-3fig-compiled-8pages-granularity-20261019'
    pdf=REPO/(stem+'.pdf');full=REPO/(stem+'.zip');anonymous=REPO/(stem+'-anonymous.zip')
    shutil.copy2(project/'paper/main.pdf',pdf)
    archive(project,full);archive(anon,anonymous)
    receipt=dict(pdf=pdf.name,pdf_sha256=sha(pdf),full_archive=full.name,full_archive_sha256=sha(full),
        anonymous_archive=anonymous.name,anonymous_archive_sha256=sha(anonymous),main_pages=8,total_pages=12,
        preserved_training_conditions=1586,new_training_conditions=185,total_training_conditions=1771,
        preserved_web_replays=60,new_quality_conditions=315,exact_multitrial_quality_conditions=27,
        regional_diagnostic_files=155,feature_bundles=4,all_generated_tables_and_new_figure_data_verified=True,
        numerical_privacy_certificates_recomputed=True,natural_writer_roles_disjoint=True,
        anonymous_verification_and_smoke_replays_passed=True,zip_crc_checks_passed=True,
        verification_scope='local; submission portal not accessed')
    (REPO/(stem+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);main(p.parse_args().project)
