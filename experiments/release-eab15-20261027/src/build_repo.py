"""Build the public GitHub working tree from the delivered project (same file filter as the ZIP archive).

Relative symlinks are kept (git stores them). Files above 50 MB are written as <name>.part00, ... of
45 MB each; reproduce.py joins them and RECORD_MANIFEST.json checks the joined SHA-256.
"""
import argparse,hashlib,json,os,shutil
from pathlib import Path
LIMIT=50*10**6;PART=45*10**6
SKIP_DIRS={'__pycache__','checkpoints','reproduction-output','validation-preview','raw','deps','sources'}
SKIP_PAPER={'main.aux','main.log','main.out','main.bbl','main.blg','build-output.txt'}

def wanted(rel):
    if any(p in SKIP_DIRS for p in rel.parts):return False
    if rel.suffix in ['.pyc','.png'] or rel.name=='core':return False
    if rel.parts[0]=='paper' and rel.name in SKIP_PAPER:return False
    return True

def main(project,repo):
    project=project.resolve();repo.mkdir(parents=True,exist_ok=True)
    for p in repo.iterdir():
        if p.name!='.git':shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    split=[];files=0;links=0
    for p in sorted(project.rglob('*')):
        rel=p.relative_to(project)
        if p.is_dir() and not p.is_symlink():continue
        if not wanted(rel):continue
        dst=repo/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if p.is_symlink():
            target=os.readlink(p)
            if os.path.isabs(target):continue
            dst.symlink_to(target);links+=1;continue
        size=p.stat().st_size
        if size>LIMIT:
            data=p.read_bytes()
            for i in range(0,len(data),PART):(dst.parent/f'{dst.name}.part{i//PART:02d}').write_bytes(data[i:i+PART])
            split.append(dict(file=str(rel),bytes=size,parts=-(-size//PART),sha256=hashlib.sha256(data).hexdigest()))
        else:shutil.copy2(p,dst)
        files+=1
    ignore=['reproduction-output/','__pycache__/','*.pyc','checkpoints/','paper/main.aux','paper/main.log','paper/main.out','paper/main.bbl','paper/main.blg','paper/build-output.txt']+[s['file'] for s in split]
    (repo/'.gitignore').write_text('\n'.join(ignore)+'\n')
    (repo/'SPLIT_FILES.json').write_text(json.dumps(split,indent=2)+'\n')
    print(json.dumps(dict(files=files,symlinks=links,split=split),indent=1))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--project',type=Path,required=True);a.add_argument('--repo',type=Path,required=True);x=a.parse_args();main(x.project,x.repo)
