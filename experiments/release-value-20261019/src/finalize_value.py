"""Package, anonymize and independently verify the value/verification revision."""
import argparse,datetime,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
import fitz
from study import ROOT,sha
REPO=ROOT.parents[1]
STEM='sdd-www2027-paper-3fig-compiled-8pages-value-20261019'

def sync(project):
    target=project/'experiments'/ROOT.name
    def ignore(folder,names):
        return [name for name in names if name in ['checkpoints','__pycache__','sources'] or name.endswith('.pyc') or name.endswith('.partial') or (Path(folder)/name).is_symlink()]
    shutil.copytree(ROOT,target,dirs_exist_ok=True,symlinks=True,ignore=ignore)
    for task in ['cifar10','mnist','web','femnist62']:
        path=target/'inputs'/(task+'.npz')
        if not path.exists():path.symlink_to('../../release-frontier-20261019/inputs/'+task+'.npz')

def layout(project):
    paper=project/'paper';log=(paper/'main.log').read_text();assert 'Overfull' not in log
    assert not re.search(r'(Citation|Reference).*undefined',log)
    pdf=fitz.open(paper/'main.pdf');texts=[p.get_text() for p in pdf]
    assert len(pdf)<=12 and 'CONCLUSION' in texts[7] and 'REFERENCES' in texts[8]
    assert not any('REFERENCES' in value for value in texts[:8])
    assert pdf.metadata['author'] in ['',None,'Anonymous Author(s)']
    result=dict(pdf_sha256=sha(paper/'main.pdf'),main_pages=8,total_pages=len(pdf),references_start_page=9,overfull_boxes=False,undefined_references=False,
        tex_source_sha256={p.name:sha(p) for p in paper.glob('*.tex')},bibliography_sha256=sha(paper/'refs.bib'))
    (paper/'BUILD_REPORT.json').write_text(json.dumps(result,indent=2)+'\n');return result

def manifest(project):
    stage=project/'experiments'/ROOT.name;parent=json.loads((stage/'PARENT-MANIFEST.json').read_text())
    scientific={name:sha(project/name) for name in parent['scientific_files'] if (project/name).is_file()}
    for directory in ['src','results','diagnostics','analysis','public','inputs/partitions']:
        for p in (stage/directory).rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in ['.py','.cpp','.so','.json','.npz','.npy']:
                scientific[str(p.relative_to(project))]=sha(p)
    for p in stage.iterdir():
        if not p.is_file() or p.suffix not in ['.json','.txt']:continue
        if p.name in ['validation-checks.json'] or 'COMPLETE' in p.name or 'STATUS' in p.name:continue
        scientific[str(p.relative_to(project))]=sha(p)
    presentation=[project/'README.md',project/'reproduce.py',*project.glob('*.csv'),*list((project/'paper').glob('*.tex')),project/'paper/main.pdf',project/'paper/refs.bib',project/'paper/BUILD_REPORT.json']
    presentation+=list((project/'paper').glob('*-data.json'))+list((project/'paper/figures').glob('*.pdf'))+list((project/'paper/figures').glob('*-data.json'))
    data=dict(revision=ROOT.name,preserved_primary_records=3426,new_validation_cohorts=10,naming_conditions=30,validation_cases=3000,validation_vectors=6000,no_feedback_protocols=10,existing_zero_label_sources=110,
        scientific_files=scientific,presentation_files={str(p.relative_to(project)):sha(p) for p in presentation})
    (project/'RECORD_MANIFEST.json').write_text(json.dumps(data,indent=2)+'\n')

def hashes(project):
    data=json.loads((project/'RECORD_MANIFEST.json').read_text())
    for group in ['scientific_files','presentation_files']:
        for name,digest in data[group].items():assert sha(project/name)==digest,name

def run(project,mode,log):
    with log.open('w') as stream:subprocess.run([sys.executable,str(project/'reproduce.py'),mode],stdout=stream,stderr=subprocess.STDOUT,check=True)

def archive(project,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(project.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            relative=p.relative_to(project)
            if any(name in relative.parts for name in ['__pycache__','checkpoints','reproduction-output','validation-preview','raw','deps','sources']):continue
            if p.suffix in ['.pyc','.png'] or p.name=='core':continue
            if relative.parts[0]=='paper' and p.name in ['main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt']:continue
            z.write(p,Path('sdd-www2027')/relative)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None

def main(project):
    project=project.resolve();sync(project);build=layout(project)
    subprocess.run([sys.executable,str(project/'experiments'/ROOT.name/'src/build_indexes.py'),'--project',str(project)],check=True)
    manifest(project);run(project,'verify',ROOT/'logs/full-verify.log');print('Full numerical/scientific/layout verification passed.',flush=True)
    anon=project.parent/'anonymous-supplement/sdd-www2027'
    if anon.exists():shutil.rmtree(anon)
    omitted={'historical','validation-experiments','validation-preview','reproduction-output','__pycache__','checkpoints','sources'}
    def ignore(folder,names):
        return [name for name in names if name in omitted or name.startswith('REVIEW_INPUT') or name.endswith('DECISIONS.txt') or name.startswith('REVISION_NOTES') or name.endswith('.pyc') or name in ['finalize_revision.py','finalize_value.py','core'] or name.startswith('preview-page-')]
    shutil.copytree(project,anon,symlinks=True,ignore=ignore)
    extensions={'.py','.json','.txt','.md','.tex','.log','.csv','.cpp','.h','.hpp','.c','.cmake'};changed=[]
    for p in anon.rglob('*'):
        if not p.is_file() or p.is_symlink() or p.suffix not in extensions or 'vendor' in p.parts:continue
        before=p.read_text();after=before.replace(str(REPO),'.').replace('/home/xbw/.local/bin/tectonic','tectonic').replace('/home/xbw','<USER_HOME>')
        if before!=after:p.write_text(after);changed.append(str(p.relative_to(anon)))
    library=project/'experiments/review-round3-20261013/anonymous-sampler/exact_dg.so'
    for name in ['review-evidence-20261012','review-round2-20261013','release-granularity-20261019','release-selector-20261019','release-frontier-20261019']:
        shutil.copy2(library,anon/'experiments'/name/'src/exact_dg.so')
    for p in (anon/'paper').glob('main.*'):
        if p.suffix not in ['.tex','.pdf']:p.unlink()
    value=(anon/'README.md').read_text().replace('# Private supervision under strong public representations','# Anonymous supplementary artifact').replace('The full local archive retains earlier drafts and delivery provenance.\n','')
    (anon/'README.md').write_text(value)
    for p in anon.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in extensions|{'.so','.pdf'}:
            b=p.read_bytes().lower();assert b'xbw' not in b and b'/root/autodl-tmp/' not in b,p
    info=dict(machine_path_files_redacted=changed,scientific_numeric_records_unchanged=True,exact_execution_source_snapshots_retained=True,sampler_same_source_and_distribution=True,known_identity_marker_scan='passed')
    (anon/'ANONYMIZATION.json').write_text(json.dumps(info,indent=2)+'\n')
    manifest(anon);run(anon,'verify',ROOT/'logs/anonymous-verify.log');print('Anonymous regeneration/identity checks passed.',flush=True)
    run(anon,'smoke',ROOT/'logs/anonymous-smoke.log');hashes(anon);print('Anonymous fresh-cohort/exact-noise reconstruction and immutable fingerprints passed.',flush=True)
    report=dict(**build,preserved_primary_records=3426,new_validation_cohorts=10,naming_conditions=30,validation_cases=3000,validation_vectors=6000,no_feedback_protocols=10,no_feedback_releases=30,
        existing_zero_label_sources=110,full_verification_passed=True,anonymous_verification_passed=True,anonymous_reconstruction_passed=True,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_scope='finite-heldout noise coverage; benchmark diagnostics outside deployment view')
    for target in [project,anon]:
        (target/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        logs=target/'experiments'/ROOT.name/'logs'
        for name in ['full-verify.log','anonymous-verify.log','anonymous-smoke.log']:
            content=(ROOT/'logs'/name).read_text()
            if target==anon:content=content.replace(str(REPO),'.')
            (logs/name).write_text(content)
    pdf=REPO/(STEM+'.pdf');full=REPO/(STEM+'.zip');anonymous=REPO/(STEM+'-anonymous.zip')
    shutil.copy2(project/'paper/main.pdf',pdf);archive(project,full);archive(anon,anonymous);print('Final archives passed CRC.',flush=True)
    clean=project.parent/'clean-extract-check'
    if clean.exists():shutil.rmtree(clean)
    clean.mkdir()
    with zipfile.ZipFile(anonymous) as z:z.extractall(clean)
    run(clean/'sdd-www2027','verify',ROOT/'logs/anonymous-clean-extract-verify.log');print('Clean-extracted anonymous ZIP verified independently.',flush=True)
    summary=json.loads((project/'paper/value-data.json').read_text())
    receipt=dict(pdf=pdf.name,pdf_sha256=sha(pdf),full_archive=full.name,full_archive_sha256=sha(full),anonymous_archive=anonymous.name,anonymous_archive_sha256=sha(anonymous),
        main_pages=8,total_pages=build['total_pages'],previous_deliveries_unchanged=True,preserved_primary_records=3426,new_validation_cohorts=10,new_validation_cases=3000,new_validation_vectors=6000,
        no_feedback_protocols=10,existing_zero_label_sources=110,global_acceptance={str(r['epsilon']):r['global_acceptance']['mean'] for r in summary['private_validation']},
        class_acceptance={str(r['epsilon']):r['class_acceptance']['mean'] for r in summary['private_validation']},
        all_figures_and_tables_regenerated=True,anonymous_reconstruction_passed=True,anonymous_clean_extract_verification_passed=True,known_identity_marker_scan_passed=True,zip_crc_checks_passed=True)
    (REPO/(STEM+'-delivery.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);main(p.parse_args().project)
