"""Reproduce the public pilot in an isolated directory using its source snapshot."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True,
                    help='New directory for the replay; existing paths are refused.')
parser.add_argument('--prepare-only', action='store_true')
args = parser.parse_args()
output = args.output.resolve()
if output.exists():
    raise SystemExit(f'Replay directory already exists: {output}')
(output/'src').mkdir(parents=True)
(output/'inputs').mkdir()
(output/'public-selection').mkdir()
for path in (ROOT/'public-selection/source-snapshot').iterdir():
    if path.is_file(): shutil.copy2(path,output/'src'/path.name)
for name in ['cifar10.npz','mnist.npz','web.npz']:
    shutil.copy2(ROOT/'inputs'/name,output/'inputs'/name)
shutil.copytree(ROOT/'src/vendor',output/'src/vendor')
licence = ROOT/'src/BOOST-COPYRIGHT.txt'
if licence.exists(): shutil.copy2(licence,output/'src'/licence.name)
command = [sys.executable,str(output/'src/select_public.py')]
print('Public pilot source snapshot and feature bundles prepared:',output,flush=True)
if args.prepare_only:
    print('Run:',*command,flush=True)
else:
    subprocess.run(command,cwd=output,check=True)
