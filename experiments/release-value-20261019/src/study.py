"""Frozen resource/value and private-validation study."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'release-frontier-20261019'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def freeze(name,design):
    design=json.loads(json.dumps(design));path=ROOT/name
    if path.exists():assert json.loads(path.read_text())['design']==design,name
    else:save(path,dict(design=design,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}))
