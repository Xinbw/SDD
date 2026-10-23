"""Short independent reconstruction from included feature bundles."""
import json

from experiment_engine import ROOT, key, run


cases=[dict(dataset='cifar10',n=10000,epsilon=.1,seed=801,method='SDD',k=10,decoder='matching',alpha=.1,group='original-five'),
       dict(dataset='femnist62',n=2000,epsilon=1.,seed=8401,method='SDD',k=62,decoder='matching',alpha=None,group='natural-writers')]
for job in cases:
    path=ROOT/'results'/(key(job)+'.json')
    old=json.loads(path.read_text())
    # run() writes the replay. Restore the recorded evidence after comparison.
    before=path.read_bytes();diagnostic=ROOT/old['diagnostic_file'];old_diagnostic=diagnostic.read_bytes()
    try:
        fresh=run(job)
        err=max(abs(fresh[k]-old[k]) for k in ['accuracy','balanced_accuracy'])
        assert err<2e-6,(key(job),err)
        assert fresh['population']==old['population']
        print('PASS',key(job),'maximum metric difference',err,flush=True)
    finally:
        path.write_bytes(before);diagnostic.write_bytes(old_diagnostic)
