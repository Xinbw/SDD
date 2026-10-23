"""Minimal entry point for the anonymous SDD supplementary artifact."""
import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    subprocess.run([sys.executable,str(ROOT/'experiments/review-round2-20261013/src/prepare_replay.py')],check=True)
    source=ROOT/'experiments/review-evidence-20261012/src/vendor'
    link=ROOT/'experiments/review-round2-20261013/src/vendor'
    if not link.exists():link.symlink_to('../../review-evidence-20261012/src/vendor',target_is_directory=True)


def verify():
    manifest=json.loads((ROOT/'RECORD_MANIFEST.json').read_text())
    for name,digest in manifest['scientific_files'].items():assert sha(ROOT/name)==digest,name
    import fitz
    pdf=fitz.open(ROOT/'paper/main.pdf')
    assert len(pdf)==12 and 'CONCLUSION' in pdf[7].get_text() and 'REFERENCES' in pdf[8].get_text()
    diagnostics=list((ROOT/'experiments/review-round3-20261013/diagnostics').glob('*.json'))
    assert len(diagnostics)==60
    for path in diagnostics:
        row=json.loads(path.read_text())
        assert row['max_original_endpoint_abs_difference']<2e-6
        for observation in row['observations']:
            assert sum(observation['target_class_counts'])==1500
            assert observation['represented_target_classes']==sum(v>0 for v in observation['target_class_counts'])
            assert observation['supervises_next_local_round']==(observation['round']<3)
    prepare()
    output=ROOT/'reproduction-output';output.mkdir(exist_ok=True)
    with (output/'table-verification.log').open('w') as stream:
        subprocess.run([sys.executable,str(ROOT/'experiments/review-round3-20261013/src/export_revision.py'),
            '--paper',str(output)],stdout=stream,check=True)
    for path in output.glob('tab-*.tex'):assert path.read_bytes()==(ROOT/'paper'/path.name).read_bytes(),path.name
    print('PASS: 1586 original condition records, 60 diagnostic replays, input/configuration hashes, tables and 8/12-page PDF.')


def smoke():
    prepare()
    sys.path.insert(0,str(ROOT/'experiments/review-round2-20261013/src'))
    import engine
    conditions=[('cifar10',10000,.1,801,'SDD',10,.1,'label_free_cifar10_10000_0.1_0.1_801_SDD.json'),
        ('web',3000,None,8101,'representative',5,1.,'web_3000_None_1.0_8101_representative.json')]
    for dataset,n,alpha,seed,method,k,epsilon,name in conditions:
        cfg=json.loads((engine.ROOT/'public-selection/web_aNone_n1000.json').read_text())['budgets']['1.0'] if dataset=='web' else {}
        result=engine.run(dataset,n,alpha,seed,method,k,epsilon,mode='center',clip=cfg.get('clip',1.),pate=cfg.get('pate'))
        old=json.loads((engine.ROOT/'results'/name).read_text())
        error=max(abs(result[key]-old[key]) for key in ['accuracy','balanced_accuracy'])
        assert error<2e-6,(name,error)
        print('PASS replay:',name,'maximum absolute accuracy difference',error)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['verify','smoke','coverage'])
    args=parser.parse_args()
    if args.mode=='verify':verify()
    elif args.mode=='smoke':smoke()
    else:
        prepare()
        subprocess.run([sys.executable,str(ROOT/'experiments/review-round3-20261013/src/replay_coverage.py')],check=True)
