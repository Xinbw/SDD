"""Independent replay of recorded verify-teacher cases from scratch (GPU).

Retrains the cohort in a temporary directory (no cached checkpoint), recomputes the naming release,
the public learners, the gated candidate, both validation releases and the decisions, and requires
the case record to equal the stored one byte for byte.
"""
import json,sys,tempfile
from pathlib import Path
VT=Path(__file__).resolve().parents[2]/'release-verify-teacher-20261020'
sys.path.insert(0,str(VT/'src'))
import run
CASES=[('mnist',9201,1.,50),('web',9401,1.,50)]

def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);(tmp/'checkpoints').mkdir();run.ROOT=tmp
        for task,seed,e,draw in CASES:
            run.run_task(task,[seed],[e],[draw],tmp)
            name=f'{task}_s{seed}_e{e:g}_d{draw}.json'
            assert (tmp/'results'/name).read_bytes()==(VT/'results'/name).read_bytes(),name
    print('PASS: retrained cohorts reproduce recorded verify-teacher cases byte for byte:',', '.join(f'{t}/{s}/eps{e:g}/draw{d}' for t,s,e,d in CASES))

if __name__=='__main__':main()
