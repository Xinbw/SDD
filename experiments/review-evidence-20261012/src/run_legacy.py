"""Isolated full-ResNet corroboration with matched n and exact noise shares."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
if (ROOT/'WEB-CONFIRMATION-SCHEDULE.json').exists():
    while not (ROOT/'WEB-CONFIRMATION-COMPLETE.json').exists():time.sleep(15)
FORK=ROOT/'legacy-resnet';FORK.mkdir(exist_ok=True)
for filename in ['train_sdd.py','train_fedct.py','sdd_core.py','dp_utils.py','data.py','models.py',
                 'cluster.py','proxy_hetero.py','rdp_accounting.py','privacy_accounting.py']:
    shutil.copy2(REPO/filename,FORK/filename)
# The fork's evaluation uses the first half of test, excluding the public proxy.
path=FORK/'data.py';text=path.read_text()
text=text.replace('test_indices = list(range(len(test_set) - global_test_size, len(test_set)))',
                  'test_indices = list(range(global_test_size))')
path.write_text(text)
# Serialize local CUDA training after an interrupted concurrent run. This
# changes scheduling only; all models, batches, epochs and releases are retained.
path=FORK/'train_sdd.py';text=path.read_text()
text=text.replace('dl_workers = 0 if concurrent_clients > 1 else None','dl_workers = 0')
path.write_text(text)
path=FORK/'sdd_core.py';text=path.read_text()
text=text.replace('loss_records.append(_train_client(client))',
    'loss_records.append(_train_client(client))\n                    print(f"[Local progress] completed client {client.id}", flush=True)')
path.write_text(text)
np.save(FORK/'singleton-public-partition.npy',np.arange(2000,dtype=np.int64))
(ROOT/'LEGACY-PLAN.json').write_text(json.dumps({'fixed_before_legacy_test_access':True,
    'both_methods_use_same_local_GPU_trainer':True,'per_example_variant':'singleton public partition',
    'budgets':[.1,1.],'seeds':[42],'actual_and_calibrated_clients':100,
    'role':'Original-architecture numerical corroboration; statistical effects use five/ten-seed classifier studies.',
    'epochs_per_round':20,'T':3,'concurrent_clients':1,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in FORK.glob('*.py')}},indent=2)+'\n')
(FORK/'exact_wrapper.py').write_text('''from pathlib import Path
import json, math, runpy, sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import privacy
import dp_utils
def corrected_share(self,n):
    if self.effective_cohort(n)!=n:raise ValueError('Actual and calibrated cohort must agree')
    cert=privacy.certificate(self.epsilon,self.delta,self.num_rounds,n,clusters=self.num_clusters,
        dimension=self.num_clusters*10,colluders=self.collusion_t,bits=self.quantization_bits)
    Path('privacy-certificate.json').write_text(json.dumps(cert,indent=2)+'\\n')
    return cert['sigma_share_integer']/cert['quantization_scale']
def exact_draw(shape,sigma,device=None,generator=None):
    integer=round(sigma)
    if abs(integer-sigma)>1e-7:raise ValueError('Integer share scale required')
    replay=int(torch.randint(0,2**62,(1,),device=device,generator=generator).cpu())
    return torch.as_tensor(privacy.draw(tuple(shape),integer,seed=replay,threads=4),device=device)
dp_utils.DPConfig.client_noise_std=corrected_share
dp_utils.sample_discrete_gaussian=exact_draw
target=sys.argv.pop(1)
runpy.run_path(str(Path(__file__).resolve().parent/target),run_name='__main__')
''')
while not (ROOT/'SUITE-COMPLETE.json').exists():time.sleep(15)
for epsilon in [.1,1.]:
    for seed in [42]:
        for method in ['SDD','DP-FedCT']:
            key=f'resnet_cifar10_n100_a0.1_e{epsilon}_s{seed}_{method}'
            result_path=ROOT/'results'/f'{key}.json'
            if result_path.exists():continue
            work=FORK/key;work.mkdir(exist_ok=True)
            checkpoint=REPO/'ckpt/cifar/0.1'/('run1' if seed==42 else f'run1_seed{seed}')
            assert (checkpoint/'round1_models.pt').exists()
            command=['/root/miniconda3/bin/python','-u',str(FORK/'exact_wrapper.py'),
                'train_sdd.py',
                '--dataset','cifar10','--clients','100','--alpha','0.1','--q','1','--rounds','3',
                '--epochs','20','--batch-size','256','--samples-per-client','6000','--arch-dist','0','0','1',
                '--lr','0.1','--val-ratio','0','--concurrent-clients','1','--root',str(REPO/'data'),'--dp-cohort-size','100',
                '--epsilon-total',str(epsilon),'--seed',str(seed),'--proxy-src-idx',
                str(REPO/'data/cifar10/proxy_cifar10_src_idx_2000.npy')]
            command+=['--proxy_size','2000','--clusters','10' if method=='SDD' else '2000','--cluster-cache',
                str(REPO/'data/cifar10/cluster_result/10_C_2000.npy') if method=='SDD' else str(FORK/'singleton-public-partition.npy'),
                '--confidence-threshold','0','--report-encoding','vote','--round1-ckpt',str(checkpoint),'--no-final-ckpt']
            (work/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            log=ROOT/'logs'/f'{key}.log';started=time.time()
            print('START',key,flush=True)
            previous=log.read_text() if log.exists() else ''
            if 'Final mean client accuracy:' not in previous:
                if previous:
                    shutil.copy2(log,log.with_suffix('.interrupted.log'))
                with log.open('w') as handle:
                    process=subprocess.run(command,cwd=work,stdout=handle,stderr=subprocess.STDOUT,
                        env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
                             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
                if process.returncode:raise RuntimeError(f'Legacy run failed: {key}; see {log}')
            else:print('Recovering completed diagnostic log:',key,flush=True)
            log_text=log.read_text()
            matches=re.findall(r'Final mean client accuracy:\s*([0-9.]+)',log_text)
            if not matches:matches=re.findall(r'avg_acc[=:]\s*([0-9.]+)',log_text)
            # Keep raw log if the trainer's final print format changes; never invent accuracy.
            if not matches:raise RuntimeError(f'Cannot parse accuracy from {log}')
            accuracy=float(matches[-1]);accuracy=accuracy/100 if accuracy>1 else accuracy
            cert=json.loads((work/'privacy-certificate.json').read_text())
            result={'dataset':'cifar10','actual_clients':100,'calibrated_clients':100,'alpha':.1,
                'epsilon':epsilon,'seed':seed,'method':method,'accuracy':accuracy,'model':'full ResNet-18',
                'K':10 if method=='SDD' else 2000,'privacy':cert,'epochs_per_round':20,'rounds':3,
                'private_records_per_client':6000,'test_samples':4033,
                'seconds':None if 'Final mean client accuracy:' in previous else time.time()-started,
                'recovered_from_completed_log':'Final mean client accuracy:' in previous,
                'execution_concurrent_clients':1,
                'configuration_fixed_before_new_final_tests':True,'selection':'fixed diagnostic; no test-based tuning',
                'initial_checkpoint_sha256':hashlib.sha256((checkpoint/'round1_models.pt').read_bytes()).hexdigest(),
                'log_file':log.name,'experiment_roles':['matched original-architecture corroboration']}
            result_path.write_text(json.dumps(result,indent=2)+'\n')
            print('DONE',key,accuracy,flush=True)
(ROOT/'LEGACY-COMPLETE.json').write_text(json.dumps({'runs':4},indent=2)+'\n')
