"""Independent replay of recorded source-disjoint cases from scratch (GPU): retrain one cohort in a
temporary directory and require the case records to equal the stored ones byte for byte."""
import importlib.util,sys,tempfile
from pathlib import Path
DJ=Path(__file__).resolve().parents[2]/'release-disjoint-20261020'
spec=importlib.util.spec_from_file_location('djrun',DJ/'src/run.py');dj=importlib.util.module_from_spec(spec);sys.path.insert(0,str(DJ/'src'));spec.loader.exec_module(dj)
def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);dj.main([9601],[50],tmp)
        for e in ['0.1','1','3']:
            name=f's9601_e{e}_d50.json';assert (tmp/'results'/name).read_bytes()==(DJ/'results'/name).read_bytes(),name
    print('PASS: retrained source-disjoint cohort 9601 reproduces recorded cases (draw 50, eps .1/1/3) byte for byte')
if __name__=='__main__':main()
