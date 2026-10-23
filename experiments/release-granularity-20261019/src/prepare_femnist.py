"""Official writer-partitioned EMNIST with disjoint public/private/evaluation writers."""
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'deps'))
import h5py

URL = 'https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2'
torch.set_num_threads(8)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download():
    archive = ROOT / 'raw/fed_emnist.tar.bz2'
    if not archive.exists():
        temporary = archive.with_suffix('.partial')
        with requests.get(URL, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for block in response.iter_content(1 << 20):
                    stream.write(block)
        assert temporary.stat().st_size == 169808360
        assert hashlib.md5(temporary.read_bytes()).hexdigest() == 'f2d29187fb207080dbda71b07dedc0ce'
        temporary.replace(archive)
    if not (ROOT / 'raw/fed_emnist_train.h5').exists():
        with tarfile.open(archive) as tar:
            assert all(Path(item.name).name in ['fed_emnist_train.h5', 'fed_emnist_test.h5']
                       for item in tar.getmembers() if item.isfile())
            tar.extractall(ROOT / 'raw', filter='data')
    return archive


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def main():
    destination = ROOT / 'inputs/femnist62.npz'
    if destination.exists():
        print('Natural-writer feature bundle already exists.', flush=True)
        return
    archive = download()
    train = h5py.File(ROOT / 'raw/fed_emnist_train.h5', 'r')['examples']
    test = h5py.File(ROOT / 'raw/fed_emnist_test.h5', 'r')['examples']
    writers = np.array(sorted(set(train.keys()) & set(test.keys())))
    assert len(writers) == 3400
    rng = np.random.default_rng(1729)
    rng.shuffle(writers)
    rosters = {'proxy': writers[:200], 'development': writers[200:400],
               'validation': writers[400:600], 'test': writers[600:800],
               'private': writers[800:]}
    # Writer roles and sampling are fixed before model fitting or private evaluation.
    freeze = {'seed': 1729, 'classes': 62, 'proxy_samples': 2000, 'development_samples': 5000,
              'validation_samples_cap': 5000, 'test_samples_cap': 5000,
              'writer_rosters': {key: value.tolist() for key, value in rosters.items()},
              'natural_private_records': 'all original training records of each selected private writer',
              'private_records_shared_across_clients': False, 'source': URL,
              'source_sha256': digest(archive),
              'documentation': 'https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data'}
    manifest = ROOT / 'inputs/FEMNIST-SPLIT-FROZEN.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == freeze
    else:
        manifest.write_text(json.dumps(freeze, indent=2) + '\n')
    images, labels, owners, offsets = [], [], [], {}
    for role in ['proxy', 'development', 'validation', 'test', 'private']:
        source = test if role in ['validation', 'test'] else train
        pixels = np.concatenate([source[str(writer)]['pixels'][:] for writer in rosters[role]])
        target = np.concatenate([source[str(writer)]['label'][:] for writer in rosters[role]]).astype(np.int64)
        owner = np.concatenate([np.repeat(str(writer), len(source[str(writer)]['label'])) for writer in rosters[role]])
        if role != 'private':
            limit = {'proxy': 2000, 'development': 5000, 'validation': 5000, 'test': 5000}[role]
            if len(target) > limit:
                picked = np.sort(rng.choice(len(target), limit, replace=False))
                pixels, target, owner = pixels[picked], target[picked], owner[picked]
        start = sum(len(y) for y in labels)
        offsets[role] = np.arange(start, start + len(target))
        images.append(np.round(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
        labels.append(target)
        owners.append(owner)
        print(role, 'samples', len(target), 'writers', len(np.unique(owner)), flush=True)
    images, labels, owners = np.concatenate(images), np.concatenate(labels), np.concatenate(owners)
    # Official FEMNIST pixels are black ink on a white background, kept as supplied.
    feature_path = ROOT / 'inputs/femnist62_dinov2.npy'
    weights = REPO / 'experiments/review-evidence-20261012/inputs/dinov2_vits14_public_weights.pt'
    code = Path('/home/xbw/.cache/torch/hub/facebookresearch_dinov2_main')
    if not code.exists():
        code = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    model = torch.hub.load(str(code), 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(weights, weights_only=True))
    model = model.cuda().eval()
    features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float32, shape=(len(images), 384))
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    started = time.time()
    for first in range(0, len(images), 256):
        batch = torch.from_numpy(images[first:first + 256].copy()).cuda()[:, None].repeat(1, 3, 1, 1)
        batch = F.interpolate(batch.float() / 255, size=(224, 224), mode='bicubic', align_corners=False)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            encoded = model((batch - mean) / std)
        features[first:first + len(batch)] = encoded.float().cpu().numpy()
        if first % 8192 == 0:
            features.flush()
            print('DINOv2', first, '/', len(images), 'elapsed', round(time.time() - started, 1), flush=True)
    features.flush()
    pca = PCA(n_components=128, svd_solver='randomized', random_state=1729).fit(features[offsets['proxy']])
    transformed = normalize(pca.transform(features)).astype(np.float32)
    data = {}
    for role, indices in offsets.items():
        data['x_' + role], data['y_' + role] = transformed[indices], labels[indices]
        data['owners_' + role] = owners[indices]
        data[role + '_ids'] = indices
    data['x_proxy_weak'] = normalize(PCA(n_components=128, svd_solver='randomized', random_state=1729)
        .fit_transform(images[offsets['proxy']].reshape(2000, -1).astype(np.float32) / 255)).astype(np.float32)
    np.savez_compressed(destination, **data)
    manifest = {'dataset': 'Federated EMNIST, 62 classes, official writer partition',
                'source': URL, 'source_sha256': digest(archive), 'natural_clients_available': 2600,
                'classes': 62, 'writer_roles_disjoint': True, 'private_records_overlap_across_writers': False,
                'private_samples': len(offsets['private']), 'proxy_samples': 2000,
                'development_samples': 5000, 'validation_samples': len(offsets['validation']),
                'test_samples': len(offsets['test']),
                'encoder': 'DINOv2 ViT-S/14, same frozen checkpoint as original image experiments',
                'encoder_weights_sha256': digest(weights), 'PCA_fit': 'public proxy only',
                'PCA_components': 128, 'feature_bundle_sha256': digest(destination),
                'pixels': 'official white-background grayscale, 224 RGB bicubic, ImageNet normalization',
                'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (ROOT / 'inputs/femnist62_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
