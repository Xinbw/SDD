"""Verify and package the second release-granularity revision separately."""
import argparse
import datetime
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import ROOT,sha

REPO=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('preceding_package',ROOT.parent/'release-granularity-20261019/src/finalize_revision.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def manifest(project):
    stage=project/'experiments'/ROOT.name
    parent=json.loads((stage/'PARENT-MANIFEST.json').read_text())
    files={name:sha(project/name) for name in parent['scientific_files'] if (project/name).is_file()}
    for directory in ['src','results','diagnostics','public-selection','analysis','inputs/partitions']:
        for p in (stage/directory).rglob('*'):
            if p.is_file() and not p.is_symlink() and 'vendor' not in p.parts and p.suffix in ['.py','.cpp','.so','.json','.npz']:
                files[str(p.relative_to(project))]=sha(p)
    for p in stage.glob('*.json'):
        if p.name in ['summary.json','revision-validation.json'] or 'STATUS' in p.name or 'COMPLETE' in p.name:continue
        files[str(p.relative_to(project))]=sha(p)
    paths=[project/name for name in ['README.md','reproduce.py','EXPERIMENT_INDEX.csv','PUBLIC_READOUT_INDEX.csv',
        'SELECTOR_DIAGNOSTIC_INDEX.csv','QUALITY_INDEX.csv','REGION_DIAGNOSTIC_INDEX.csv','PAPER_EVIDENCE_INDEX.csv']]
    paths+=list((project/'paper').glob('*.tex'))+[project/'paper/main.pdf',project/'paper/refs.bib',project/'paper/selector-data.json']
    paths+=list((project/'paper/figures').glob('*.pdf'))+list((project/'paper/figures').glob('*data.json'))
    data=dict(revision='release-selector-20261019',preserved_primary_conditions=1771,new_client_conditions=540,
        new_matched_query_conditions=5,new_public_reference_conditions=20,new_public_student_conditions=380,
        total_primary_conditions=2336,new_centroid_evaluations_of_existing_releases=390,
        scientific_files=files,presentation_files={str(p.relative_to(project)):sha(p) for p in paths})
    (project/'RECORD_MANIFEST.json').write_text(json.dumps(data,indent=2)+'\n')


def run(project,mode,log):
    with log.open('w') as stream:
        subprocess.run([sys.executable,str(project/'reproduce.py'),mode],stdout=stream,stderr=subprocess.STDOUT,check=True)


def main(project):
    project=project.resolve();stage=project/'experiments'/ROOT.name
    for directory in ['src','results','diagnostics','public-selection','analysis','logs','inputs/partitions']:
        shutil.copytree(ROOT/directory,stage/directory,dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','vendor'))
    for path in ROOT.glob('*.json'):shutil.copy2(path,stage/path.name)
    with (ROOT/'logs/final-validation.log').open('w') as stream:
        subprocess.run([sys.executable,str(stage/'src/validate_revision.py'),'--project',str(project)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=json.loads((stage/'revision-validation.json').read_text())
    report.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_scope='local artifact verification; no submission or independent reviewer run claimed')
    (project/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest(project);run(project,'verify',ROOT/'logs/full-verify.log')
    print('Full scientific, numerical and layout verification passed.',flush=True)
    anon=project.parent/'anonymous-supplement/sdd-www2027'
    if anon.exists():shutil.rmtree(anon)
    omitted={'historical','validation-experiments','validation-preview','reproduction-output','__pycache__','checkpoints'}
    def ignore(folder,names):
        return [n for n in names if n in omitted or n.startswith('REVIEW_INPUT') or n.endswith('DECISIONS.txt')
            or n.startswith('REVISION_NOTES') or n.endswith('.pyc') or n=='finalize_revision.py' or n.startswith('preview-page-')]
    shutil.copytree(project,anon,symlinks=True,ignore=ignore)
    suffixes={'.py','.json','.txt','.md','.tex','.log','.csv','.cpp','.h','.hpp','.c','.cmake'}
    redacted=[]
    for p in anon.rglob('*'):
        if not p.is_file() or p.is_symlink() or p.suffix not in suffixes or 'vendor' in p.parts:continue
        value=p.read_text();cleaned=value.replace(str(REPO),'.').replace('/home/xbw/.local/bin/tectonic','tectonic').replace('/home/xbw','<USER_HOME>')
        if value!=cleaned:p.write_text(cleaned);redacted.append(str(p.relative_to(anon)))
    library=project/'experiments/review-round3-20261013/anonymous-sampler/exact_dg.so'
    for name in ['review-evidence-20261012','review-round2-20261013','release-granularity-20261019','release-selector-20261019']:
        shutil.copy2(library,anon/'experiments'/name/'src/exact_dg.so')
    for p in (anon/'paper').glob('main.*'):
        if p.suffix not in ['.tex','.pdf']:p.unlink()
    readme=(anon/'README.md').read_text().replace('# SDD WWW 2027: end-to-end public release selection','# Anonymous SDD supplementary artifact')
    readme=readme.replace('The full local archive retains earlier drafts and delivery provenance.\n','')
    (anon/'README.md').write_text(readme)
    info=dict(machine_path_files_redacted=redacted,numerical_results_unchanged=True,execution_source_snapshots_retained=True,
        original_source_hashes_retained_as_provenance=True,active_anonymous_source_hashes_in_record_manifest=True,
        sampler_same_source_and_distribution=True,known_workspace_identity_marker_scan='pending')
    (anon/'ANONYMIZATION.json').write_text(json.dumps(info,indent=2)+'\n')
    for p in anon.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in suffixes|{'.so','.pdf'}:
            value=p.read_bytes().lower();assert b'xbw' not in value and b'/root/autodl-tmp/' not in value,p
    info['known_workspace_identity_marker_scan']='passed';(anon/'ANONYMIZATION.json').write_text(json.dumps(info,indent=2)+'\n')
    manifest(anon);run(anon,'verify',ROOT/'logs/anonymous-verify.log')
    print('Anonymous evidence regeneration passed.',flush=True)
    run(anon,'smoke',ROOT/'logs/anonymous-smoke.log')
    run(anon,'verify',ROOT/'logs/anonymous-post-smoke-verify.log')
    print('Anonymous fresh-cohort reconstruction and post-replay verification passed.',flush=True)
    report.update(anonymous_verification_passed=True,anonymous_smoke_replay_passed=True,known_identity_marker_scan_passed=True)
    for target in [project,anon]:
        (target/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        folder=target/'experiments'/ROOT.name/'logs'
        for name in ['final-validation.log','full-verify.log','anonymous-verify.log','anonymous-smoke.log','anonymous-post-smoke-verify.log']:
            value=(ROOT/'logs'/name).read_text()
            if target==anon:value=value.replace(str(REPO),'.')
            (folder/name).write_text(value)
    stem='sdd-www2027-paper-3fig-compiled-8pages-selector-20261019'
    pdf=REPO/(stem+'.pdf');full=REPO/(stem+'.zip');anonymous=REPO/(stem+'-anonymous.zip')
    shutil.copy2(project/'paper/main.pdf',pdf);previous.archive(project,full);previous.archive(anon,anonymous)
    receipt=dict(pdf=pdf.name,pdf_sha256=sha(pdf),full_archive=full.name,full_archive_sha256=sha(full),
        anonymous_archive=anonymous.name,anonymous_archive_sha256=sha(anonymous),main_pages=8,total_pages=report['total_pages'],
        preserved_primary_conditions=1771,new_client_conditions=540,new_matched_query_conditions=5,
        new_public_reference_conditions=20,total_primary_conditions=2336,public_student_readouts=380,
        previous_trajectory_centroid_evaluations=390,end_to_end_settings=8,paired_seeds_per_setting=10,
        actual_and_calibrated_cohorts_match=True,all_active_tables_and_figures_regenerated=True,
        adverse_class_permutation_seed_preserved=True,anonymous_reconstruction_passed=True,
        known_identity_marker_scan_passed=True,zip_crc_checks_passed=True,verification_scope='local; submission portal not accessed')
    (REPO/(stem+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);main(p.parse_args().project)
