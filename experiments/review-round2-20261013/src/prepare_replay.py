"""Prepare shared feature inputs for a clean packaged round-two reproduction."""
import hashlib
import json
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'review-evidence-20261012'
manifest=json.loads((ROOT/'PARENT-MANIFEST.json').read_text())
(ROOT/'inputs/partitions').mkdir(parents=True,exist_ok=True)
(ROOT/'checkpoints').mkdir(exist_ok=True)
for name in ['cifar10.npz','mnist.npz','web.npz']:
    source=PARENT/'inputs'/name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['inputs'][name]
    target=ROOT/'inputs'/name
    if not target.exists():target.symlink_to(os.path.relpath(source,target.parent))
print('Feature hashes verified; shared inputs and client-checkpoint directory ready.')
