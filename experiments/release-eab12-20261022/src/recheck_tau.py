"""Re-check every recorded release certificate with the corrected convolution constant (EAB round 12).

Kairouz, Liu and Steinke (2021; arXiv 2102.06387, Proposition 14, Eq. 17) give, for a sum of h
independent discrete Gaussians N_Z(0,b^2) per coordinate, 1/2 eps^2-concentrated DP with
eps^2 <= Delta^2/(h b^2) + 2 tau d, i.e. rho = Delta^2/(2 h b^2) + tau d, where
tau = 10 sum_{k=1}^{h-1} exp(-2 pi^2 b^2 k/(k+1)) <= 10 (h-1) exp(-pi^2 b^2).
The frozen accounting code (privacy.certificate) added d*tau_upper/4 instead of d*tau, with tau_upper
the bound above floored at 1e-30. Since rho is the first term plus d times the tau term, the corrected
certificate is no larger than the recorded one exactly when 10 (h-1) exp(-pi^2 b^2) <= tau_upper/4.
This script checks that inequality in 90-digit arithmetic for every certificate recorded under
experiments/ and reports the largest log10 tau. If it holds everywhere, every recorded per-release and
composed budget remains a valid upper bound under the corrected constant; no decision, noise draw or
result changes.
"""
import json,sys
from pathlib import Path
import mpmath as mp
mp.mp.dps=90
EXP=Path(__file__).resolve().parents[2]
KEYS={'sensitivity_integer_squared','sigma_share_integer','honest_clients','dimension','epsilon_requested','delta','rounds'}

def walk(x,out):
    if isinstance(x,dict):
        if KEYS<=x.keys():out.append(x)
        for v in x.values():walk(v,out)
    elif isinstance(x,list):
        for v in x:walk(v,out)

def check(c):
    h=int(c['honest_clients']);b=int(c['sigma_share_integer'])
    tau=mp.mpf(10)*(h-1)*mp.e**(-mp.pi**2*b*b) if h>1 else mp.mpf(0);recorded=mp.mpf(c['tau_certified_upper'])
    ok=(tau==0 and recorded==0) or tau<=recorded/4
    return ok,mp.log10(tau) if tau>0 else None

def main():
    files=0;certs=0;bad=0;dtau=-1e9;seen=set()
    for p in sorted(EXP.rglob('*.json')):
        if 'release-eab12-20261022' in p.parts or p.stat().st_size>200_000_000:continue
        try:data=json.loads(p.read_text())
        except Exception:continue
        found=[];walk(data,found)
        if not found:continue
        files+=1
        for c in found:
            k=(c['honest_clients'],c['sigma_share_integer'],c['tau_certified_upper'])
            certs+=1
            if k in seen:continue
            seen.add(k);ok,lt=check(c);bad+=0 if ok else 1
            if lt is not None:dtau=max(dtau,float(lt))
    res=dict(files_with_certificates=files,certificates=certs,distinct_certificates=len(seen),violations=bad,max_log10_tau=dtau,all_hold=bad==0,
        check='10 (h-1) exp(-pi^2 b^2) <= recorded tau_certified_upper / 4 for every distinct certificate',
        constant='rho = Delta^2/(2 h b^2) + d tau, tau <= 10 (h-1) exp(-pi^2 b^2) (Kairouz et al. 2021, arXiv Prop. 14)')
    out=Path(__file__).resolve().parents[1]/'analysis';out.mkdir(exist_ok=True);(out/'tau-recheck.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=1))
    assert res['all_hold']

if __name__=='__main__':main()
