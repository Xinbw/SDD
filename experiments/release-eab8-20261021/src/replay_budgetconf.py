"""Independent replay of the budget-split confirmation from scratch (GPU): retrain one development and
one confirmation cohort in a temporary directory and require their records (label draw 0 resp. 50,
all budgets and splits) to equal the stored ones."""
import importlib.util,json,sys,tempfile
from pathlib import Path
BC=Path(__file__).resolve().parents[2]/'release-budgetconf-20261021'
spec=importlib.util.spec_from_file_location('bcrun',BC/'src/run.py');bc=importlib.util.module_from_spec(spec);spec.loader.exec_module(bc)
def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);(tmp/'results').mkdir()
        for phase,seed,draw in [('dev',9701,0),('confirm',9801,50)]:
            bc.main(phase,[seed],[draw],tmp)
            new=json.loads((tmp/'results'/f'{phase}_s{seed}.json').read_text());old=json.loads((BC/'results'/f'{phase}_s{seed}.json').read_text())
            assert new['cohort']==old['cohort'] and new['certificates']==old['certificates'] and new['selection']==old['selection']
            assert new['rows']==[r for r in old['rows'] if r['draw']==draw],(phase,seed)
    print('PASS: retrained development cohort 9701 and confirmation cohort 9801 reproduce their recorded cases (54 each: 3 budgets x 18 splits) exactly')
if __name__=='__main__':main()
