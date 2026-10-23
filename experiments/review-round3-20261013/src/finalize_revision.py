"""Create full provenance and anonymous supplementary deliveries."""
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

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific_manifest(project):
    files={}
    for stage in ['review-evidence-20261012','review-round2-20261013']:
        folder=project/'experiments'/stage
        for p in (folder/'results').glob('*.json'):files[str(p.relative_to(project))]=sha(p)
        for p in (folder/'public-selection').glob('*.json'):files[str(p.relative_to(project))]=sha(p)
        for name in ['engine.py','privacy.py','exact_dg.cpp','exact_dg.so']:
            p=folder/'src'/name;files[str(p.relative_to(project))]=sha(p)
    for p in (project/'experiments/review-evidence-20261012/inputs').glob('*.npz'):
        files[str(p.relative_to(project))]=sha(p)
    for p in (project/'experiments/review-round3-20261013/diagnostics').glob('*.json'):
        files[str(p.relative_to(project))]=sha(p)
    p=project/'experiments/review-round2-20261013/PUBLIC-RESOURCE-FROZEN.json'
    files[str(p.relative_to(project))]=sha(p)
    (project/'RECORD_MANIFEST.json').write_text(json.dumps({'scientific_files':files,
        'condition_records':1586,'instrumentation_replays':60},indent=2)+'\n')


def archive(project,path):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(project.rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(project)
            if '__pycache__' in relative.parts or p.suffix=='.pyc' or 'reproduction-output' in relative.parts or 'checkpoints' in relative.parts:continue
            if relative.parts[:3]==('experiments','review-round2-20261013','inputs') and p.suffix=='.npz':continue
            if relative.parts[:4]==('experiments','review-round2-20261013','src','vendor'):continue
            if 'validation-preview' in relative.parts:continue
            if relative.parts[0]=='paper' and p.name in ['main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt']:continue
            z.write(p,Path('sdd-www2027')/relative)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None


def main(project):
    project=project.resolve()
    subprocess.run([sys.executable,str(ROOT/'src/validate_revision.py'),'--project',str(project)],check=True)
    dst=project/'experiments'/ROOT.name
    for name in ['src','diagnostics','logs','table-source','anonymous-sampler']:
        shutil.copytree(ROOT/name,dst/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for p in ROOT.iterdir():
        if p.is_file():shutil.copy2(p,dst/p.name)
    scientific_manifest(project)
    validation=json.loads((ROOT/'revision-validation.json').read_text())
    validation['sampler_anonymization']=json.loads((ROOT/'SAMPLER-ANONYMIZATION.json').read_text())
    validation['prior_delivery']=json.loads((ROOT/'PARENT-MANIFEST.json').read_text())['parent_delivery']
    validation['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (project/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    subprocess.run([sys.executable,str(project/'reproduce.py'),'verify'],check=True)
    anon=project.parent/'anonymous-supplement/sdd-www2027'
    if anon.exists():shutil.rmtree(anon)
    ignored={'historical','validation-experiments','validation-preview','reproduction-output','__pycache__'}
    def ignore(folder,names):
        return [name for name in names if name in ignored or name.startswith('REVIEW_INPUT')
            or name.endswith('DECISIONS.txt') or name.startswith('REVISION_NOTES') or name.endswith('.pyc')]
    shutil.copytree(project,anon,ignore=ignore)
    (anon/'experiments/review-round3-20261013/src/finalize_revision.py').unlink()
    changes=[]
    workspace=str(REPO)
    text_suffixes={'.py','.json','.txt','.md','.tex','.log','.csv','.cpp','.h','.hpp','.c','.cmake'}
    for p in anon.rglob('*'):
        if not p.is_file() or p.suffix not in text_suffixes:continue
        if 'vendor' in p.parts:continue
        text=p.read_text(errors='strict')
        cleaned=text.replace(workspace,'.').replace('/home/xbw/.local/bin/tectonic','tectonic')
        cleaned=cleaned.replace('/home/xbw','<USER_HOME>')
        if cleaned!=text:
            changes.append(str(p.relative_to(anon)));p.write_text(cleaned)
    lib=ROOT/'anonymous-sampler/exact_dg.so'
    for stage in ['review-evidence-20261012','review-round2-20261013']:
        shutil.copy2(lib,anon/'experiments'/stage/'src/exact_dg.so')
    # Historical copies may embed the original binary; omit non-active audit code.
    for p in (anon/'paper').glob('main.*'):
        if p.suffix not in ['.tex','.pdf']:p.unlink()
    (anon/'README.md').write_text('''# Anonymous SDD supplementary artifact

The paper has eight main pages and twelve pages total. All 1586 numerical
condition records, three shared feature bundles, complete generated grids,
public candidate scores, schedules, integer-noise certificates and raw training
logs are included. Sixty additional instrumented replays inspect already
protected Web targets without changing any algorithm or configuration.
`EXPERIMENT_INDEX.csv` locates seeds, roles, configurations and outputs.

From this directory:

```bash
python reproduce.py verify
python reproduce.py smoke
```

Verification checks fingerprints and regenerates current tables from records;
it requires NumPy, SciPy and PyMuPDF. Smoke replay checks the fixed no-task-label
CIFAR-10 case and the public-configured Usenet representative case against their
stored endpoints; it additionally requires CUDA PyTorch, scikit-learn and OpenSSL.
Recorded versions are in `experiments/review-round2-20261013/ENVIRONMENT.json`.
The entry point prepares shared feature links and optional cache directories.

The exact sampler is rebuilt from unchanged C++ source, using the same compiler
and flags with compile-time file paths mapped to relative names. Its replay
vectors match the original binary bit for bit; `SAMPLER-ANONYMIZATION.json`
records both fingerprints. Machine paths in command/log provenance are redacted.
Numerical records and feature/configuration fingerprints are unchanged.

The three experiment trees separate original sweeps/confirmations, public/local
source controls and fixed rules, and protected target-allocation diagnostics.
`table-source` retains pristine inputs to the current table exporter.
Releases 1/2 supervise later local training; release 3 is charged but unused by
the final local classifiers. All utility metrics are offline benchmark measures,
outside the protected deployment transcript. The deployment analysis assumes
the aggregate-only transport stated in the paper.

To replay the instrumented conditions from scratch in an isolated copy, move the
round-three `diagnostics` directory aside, create an empty replacement, and run
`python reproduce.py coverage`. No private result chooses a configuration.
To regenerate the revised tables, run the round-three `export_revision.py`
with `--paper paper`. Compile from `paper/` with Tectonic.

The main classifier checks need the supplied feature bundles. Full ResNet
diagnostics use external raw data and historical checkpoints with recorded
sources and hashes. Session reviews and earlier paper drafts are omitted.
''')
    shutil.copy2(ROOT/'SAMPLER-ANONYMIZATION.json',anon/'SAMPLER-ANONYMIZATION.json')
    scientific_manifest(anon)
    for p in anon.rglob('*'):
        if not p.is_file():continue
        if p.suffix in text_suffixes or p.suffix in ['.so','.pdf']:
            data=p.read_bytes().lower()
            assert b'xbw' not in data and b'/root/autodl-tmp/' not in data,(p,'identity marker')
    (anon/'ANONYMIZATION.json').write_text(json.dumps({'machine_path_files_redacted':changes,
        'session_review_and_historical_draft_files_omitted':True,'numerical_condition_records_unchanged':1586,
        'compiled_diagnostic_paths_mapped_to_relative_names':True,'known_workspace_identity_markers_absent':True},indent=2)+'\n')
    subprocess.run([sys.executable,str(anon/'reproduce.py'),'verify'],check=True)
    subprocess.run([sys.executable,str(anon/'reproduce.py'),'smoke'],check=True)
    stem='sdd-www2027-paper-3fig-compiled-8pages-round3-20261013'
    pdf=REPO/(stem+'.pdf');fullzip=REPO/(stem+'.zip');anonzip=REPO/(stem+'-anonymous.zip')
    shutil.copy2(project/'paper/main.pdf',pdf)
    archive(project,fullzip);archive(anon,anonzip)
    receipt={'pdf':pdf.name,'pdf_sha256':sha(pdf),'full_archive':fullzip.name,'full_archive_sha256':sha(fullzip),
        'anonymous_archive':anonzip.name,'anonymous_archive_sha256':sha(anonzip),
        'main_pages':8,'total_pages':12,'preserved_condition_records':1586,'diagnostic_replays':60,
        'anonymous_verification_and_smoke_replay_passed':True,'known_identity_marker_scan_passed':True,
        'zip_crc_checks_passed':True}
    (REPO/(stem+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    main(a.project)
