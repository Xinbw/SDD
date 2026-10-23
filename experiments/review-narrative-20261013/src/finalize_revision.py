"""Package the narrative paper and locally verified anonymous supplement."""
import argparse
import datetime
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('previous_packaging',
    ROOT.parent / 'review-round3-20261013/src/finalize_revision.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
sha = previous.sha


def manifest(project):
    previous.scientific_manifest(project)
    path = project / 'RECORD_MANIFEST.json'
    data = json.loads(path.read_text())
    files = [project / 'reproduce.py', project / 'EXPERIMENT_INDEX.csv',
             project / 'PAPER_EVIDENCE_INDEX.csv', project / 'README.md']
    files += list((project / 'paper').glob('*.tex')) + [project / 'paper/main.pdf']
    files += list((project / 'paper/figures').glob('*.pdf'))
    files += [project / 'paper/figures/fig-mechanism-data.json', project / 'paper/core-supervision-data.json']
    files += list((project / 'experiments/review-narrative-20261013/src').glob('*.py'))
    files += [project / 'experiments/review-narrative-20261013/PARENT-MANIFEST.json']
    files += [project / 'experiments/review-round4-20261013/src/export_tables.py',
              project / 'experiments/review-round4-20261013/src/export_mechanism.py']
    data['presentation_files'] = {str(p.relative_to(project)): sha(p) for p in files}
    data['revision'] = 'narrative-20261013'
    path.write_text(json.dumps(data, indent=2) + '\n')


def run(project, mode, log):
    with log.open('w') as stream:
        subprocess.run([sys.executable, str(project / 'reproduce.py'), mode],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main(project):
    project = project.resolve()
    dst = project / 'experiments' / ROOT.name
    shutil.copytree(ROOT / 'src', dst / 'src', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'PARENT-MANIFEST.json', dst / 'PARENT-MANIFEST.json')
    (dst / 'logs').mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(dst / 'src/validate_revision.py'),
                    '--project', str(project)], check=True)
    report = json.loads((dst / 'revision-validation.json').read_text())
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['verification_scope'] = 'local artifact validation; no independent reviewer execution claimed'
    (project / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest(project)
    run(project, 'verify', ROOT / 'logs/full-verify.log')
    print('Full artifact verification passed.', flush=True)
    anon = project.parent / 'anonymous-supplement/sdd-www2027'
    if anon.exists():
        shutil.rmtree(anon)
    ignored = {'historical', 'validation-experiments', 'validation-preview',
               'reproduction-output', '__pycache__', 'checkpoints'}
    def ignore(folder, names):
        return [name for name in names if name in ignored or name.startswith('REVIEW_INPUT')
                or name.endswith('DECISIONS.txt') or name.startswith('REVISION_NOTES')
                or name.endswith('.pyc') or name == 'finalize_revision.py']
    shutil.copytree(project, anon, symlinks=True, ignore=ignore)
    suffixes = {'.py', '.json', '.txt', '.md', '.tex', '.log', '.csv', '.cpp', '.h', '.hpp', '.c', '.cmake'}
    redacted = []
    for path in anon.rglob('*'):
        if not path.is_file() or path.suffix not in suffixes or 'vendor' in path.parts:
            continue
        value = path.read_text()
        cleaned = value.replace(str(REPO), '.').replace('/home/xbw/.local/bin/tectonic', 'tectonic')
        cleaned = cleaned.replace('/home/xbw', '<USER_HOME>')
        if value != cleaned:
            redacted.append(str(path.relative_to(anon)))
            path.write_text(cleaned)
    library = project / 'experiments/review-round3-20261013/anonymous-sampler/exact_dg.so'
    for stage in ['review-evidence-20261012', 'review-round2-20261013']:
        shutil.copy2(library, anon / 'experiments' / stage / 'src/exact_dg.so')
    for path in (anon / 'paper').glob('main.*'):
        if path.suffix not in ['.tex', '.pdf']:
            path.unlink()
    shutil.copy2(project / 'experiments/review-round3-20261013/SAMPLER-ANONYMIZATION.json',
                 anon / 'SAMPLER-ANONYMIZATION.json')
    readme = (project / 'README.md').read_text()
    readme = readme.replace('# SDD WWW 2027: narrative revision', '# Anonymous SDD supplementary artifact')
    readme = readme.replace('The full local archive retains prior source history and delivery provenance.\n', '')
    (anon / 'README.md').write_text(readme)
    (anon / 'ANONYMIZATION.json').write_text(json.dumps({
        'machine_path_files_redacted': redacted,
        'session_review_and_historical_draft_files_omitted': True,
        'numerical_condition_records_unchanged': 1586,
        'diagnostic_replays_unchanged': 60,
        'compiled_diagnostic_paths_mapped_to_relative_names': True,
        'known_workspace_identity_marker_scan': 'pending'}, indent=2) + '\n')
    for path in anon.rglob('*'):
        if path.is_file() and path.suffix in suffixes | {'.so', '.pdf'}:
            value = path.read_bytes().lower()
            assert b'xbw' not in value and b'/root/autodl-tmp/' not in value, path
    info = json.loads((anon / 'ANONYMIZATION.json').read_text())
    info['known_workspace_identity_marker_scan'] = 'passed'
    (anon / 'ANONYMIZATION.json').write_text(json.dumps(info, indent=2) + '\n')
    manifest(anon)
    run(anon, 'verify', ROOT / 'logs/anonymous-verify.log')
    print('Anonymous artifact verification passed.', flush=True)
    run(anon, 'smoke', ROOT / 'logs/anonymous-smoke.log')
    print('Two representative anonymous smoke replays passed.', flush=True)
    report['anonymous_verification_passed'] = True
    report['anonymous_smoke_replay_passed'] = True
    report['known_identity_marker_scan_passed'] = True
    for destination in [project, anon]:
        (destination / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        stage = destination / 'experiments/review-narrative-20261013'
        for name in ['full-verify.log', 'anonymous-verify.log', 'anonymous-smoke.log']:
            shutil.copy2(ROOT / 'logs' / name, stage / 'logs' / name)
    # Stored replay log contents may include paths; redact the anonymous copies too.
    for path in (anon / 'experiments/review-narrative-20261013/logs').glob('*.log'):
        path.write_text(path.read_text().replace(str(REPO), '.'))
    stem = 'sdd-www2027-paper-3fig-compiled-8pages-narrative-20261013'
    pdf, full_zip, anonymous_zip = [REPO / (stem + suffix) for suffix in ['.pdf', '.zip', '-anonymous.zip']]
    shutil.copy2(project / 'paper/main.pdf', pdf)
    previous.archive(project, full_zip)
    previous.archive(anon, anonymous_zip)
    receipt = {'pdf': pdf.name, 'pdf_sha256': sha(pdf),
               'full_archive': full_zip.name, 'full_archive_sha256': sha(full_zip),
               'anonymous_archive': anonymous_zip.name, 'anonymous_archive_sha256': sha(anonymous_zip),
               'main_pages': 8, 'total_pages': 12, 'preserved_condition_records': 1586,
               'preserved_diagnostic_replays': 60, 'new_training_conditions': 0,
               'core_table_in_main_text': True, 'narrative_revision_complete': True,
               'all_generated_tables_verified': True,
               'indexed_result_identities_verified': 1586,
               'anonymous_verification_and_smoke_replay_passed': True,
               'known_identity_marker_scan_passed': True, 'zip_crc_checks_passed': True,
               'verification_scope': 'local; submission portal not accessed'}
    (REPO / (stem + '-delivery.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.copy2(dst / 'revision-validation.json', ROOT / 'revision-validation.json')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    main(parser.parse_args().project)
