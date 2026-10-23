"""Represent screening sensitivity exactly; verify sampled scales are unchanged."""
import json
from pathlib import Path
import privacy

ROOT=Path(__file__).resolve().parents[1]
verified=0;normalised=0
for path in (ROOT/'results').glob('*_PATE-DI.json'):
    row=json.loads(path.read_text());changed=False
    for trace in row['trace']:
        cert=trace['privacy'];old=cert['screen']
        new=privacy.certificate(row['epsilon'],row['delta'],row['rounds'],row['actual_clients'],
            dimension=old['dimension'],integer_sensitivity_squared=old['dimension']*old['quantization_scale']**2,
            rho_fraction=old['rho_fraction'],accounting='rdp',colluders=old['colluders'])
        assert new['sigma_share_integer']==old['sigma_share_integer'],(path,old,new)
        verified+=1
        if new!=old:
            normalised+=1;changed=True;cert['screen']=new
        rho=new['rho_total_upper']+cert['answers']['rho_total_upper']
        epsilon=privacy.epsilon_from_rho(rho,row['delta'],'rdp');assert epsilon<=row['epsilon']
        cert['rho_total_upper']=rho;cert['epsilon_upper']=epsilon
    if changed:path.write_text(json.dumps(row,indent=2)+'\n')
(ROOT/'CERTIFICATE-NORMALIZATION.json').write_text(json.dumps({'screening_stages_verified':verified,
    'normalised_certificates':normalised,'all_sampled_integer_scales_unchanged':True,
    'training_noise_or_accuracy_changed':False,'exact_sensitivity_squared':'|P|*s^2'},indent=2)+'\n')
print('Exact screening sensitivity certificates verified:',verified)
