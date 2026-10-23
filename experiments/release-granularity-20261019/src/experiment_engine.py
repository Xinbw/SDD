"""Frozen-decoder study. Clean counts and benchmark labels are offline diagnostics.

Private inputs never select a decoder or a public configuration. Every variant
uses the unchanged exact-noise implementation and the original training streams.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import engine_base as base
import privacy

ROOT = base.ROOT


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def key(job):
    return '_'.join(str(job[x]) for x in ['dataset', 'n', 'epsilon', 'seed', 'method', 'k', 'decoder'])


def uniform_partition(data, k):
    """K uniformly chosen queries, nearest-query public propagation."""
    x = data['x_proxy']
    queries = np.random.default_rng(1729 + k).choice(len(x), k, replace=False)
    centres = x[queries]
    ids = ((x[:, None] - centres[None]) ** 2).sum(2).argmin(1)
    return ids.astype(np.int64), queries, centres


def decode(scores, row_weights, decoder):
    a = base.project_rows(scores)
    prior = (a * row_weights[:, None]).sum(0) / row_weights.sum()
    if decoder == 'center':
        return (a - prior).argmax(-1)
    if decoder == 'raw':
        return a.argmax(-1)
    if decoder in ['matching', 'matching-unweighted']:
        assert a.shape[0] == a.shape[1], 'Bijection is prespecified only for K=C.'
        objective = a if decoder == 'matching-unweighted' else a * row_weights[:, None]
        rows, columns = linear_sum_assignment(-objective.cpu().numpy())
        labels = np.empty(len(a), dtype=np.int64)
        labels[rows] = columns
        return torch.as_tensor(labels, device=base.DEVICE)
    if decoder == 'soft':
        return a
    if decoder == 'center-soft':
        # Fixed temperature, declared before examining private outcomes.
        return ((a - prior) / .1).softmax(-1)
    raise ValueError(decoder)


def train_round(weights, velocity, x, y, index, sizes, seed, r, proxy_x, targets):
    if targets is None or targets.ndim == 1:
        return base.train_round(weights, velocity, x, y, index, sizes, seed, r, proxy_x, targets)
    n = len(weights)
    generator = torch.Generator(device=base.DEVICE).manual_seed(seed * 100003 + r * 101)
    generator_proxy = torch.Generator(device=base.DEVICE).manual_seed(seed * 100019 + r * 103)
    for step in range(base.STEPS):
        local = (torch.rand((n, 64), device=base.DEVICE, generator=generator) * sizes[:, None]).long()
        global_ids = index.gather(1, local)
        for first in range(0, n, 1024):
            last = min(n, first + 1024)
            picks = torch.randint(len(targets), (last-first, 64), device=base.DEVICE, generator=generator_proxy)
            features = torch.cat([x[global_ids[first:last]], proxy_x[picks]], dim=1)
            w = weights[first:last]
            residual = torch.bmm(features, w).softmax(-1)
            labels = y[global_ids[first:last]]
            residual[:, :64].scatter_add_(-1, labels[..., None], -torch.ones_like(labels[..., None], dtype=residual.dtype))
            residual[:, 64:] -= targets[picks]
            grad = torch.bmm(features.transpose(1, 2), residual) / 128 + base.WEIGHT_DECAY * w
            velocity[first:last].mul_(base.MOMENTUM).add_(grad)
            w.add_(velocity[first:last], alpha=-base.LR)
    return weights, velocity


def release(weights, data, job, r, diagnostic):
    n, d, c = weights.shape
    xp = torch.as_tensor(data['x_proxy'], device=base.DEVICE)
    if job['method'] == 'DP-FedCT':
        ids = np.arange(len(xp))
        counts = base.vote_counts(weights, xp)
    else:
        if job['method'] == 'uniform-query':
            ids, reps, _ = uniform_partition(data, job['k'])
        else:
            ids, reps, _ = base.partition(data, job['k'])
        counts = base.vote_counts(weights, xp, ids, reps if job['method'] == 'uniform-query' else None)
    k = len(counts)
    cert = privacy.certificate(job['epsilon'], 1e-5, 3, n, clusters=k, dimension=k*c)
    score = base.noisy_votes(counts, cert, job['seed']*1000003+r*101)
    row_weights = torch.bincount(torch.as_tensor(ids, device=base.DEVICE), minlength=k).float()
    label = decode(score, row_weights, job['decoder'])
    clean = decode(counts.float()/n, row_weights, job['decoder'])
    hard = label.argmax(-1) if label.ndim == 2 else label
    clean_hard = clean.argmax(-1) if clean.ndim == 2 else clean
    truth = np.stack([np.bincount(data['y_proxy'][ids == i], minlength=c) for i in range(k)])
    a = base.project_rows(score)
    diagnostic.update({f'r{r}_counts': counts.cpu().numpy(), f'r{r}_scores': score.cpu().numpy(),
                       f'r{r}_projected': a.cpu().numpy(), f'r{r}_decoded': label.cpu().numpy(),
                       f'r{r}_clean_decoded': clean.cpu().numpy(), 'ids': ids, 'truth_counts': truth})
    h = hard.cpu().numpy()
    diagnostics = dict(logical_report_coordinates=k*c, pseudolabel_accuracy=float((h[ids] == data['y_proxy']).mean()),
        public_cluster_purity=float(truth.max(1).sum()/len(ids)),
        clean_pseudolabel_accuracy=float((clean_hard.cpu().numpy()[ids] == data['y_proxy']).mean()),
        cluster_flip_fraction=float((hard != clean_hard).float().mean()),
        distinct_released_classes=int(len(np.unique(h))), repeated_class_regions=int(k-len(np.unique(h))),
        pseudolabel_coverage=1., supervises_final_local_model=(r < 3))
    return label[torch.as_tensor(ids, device=base.DEVICE)], cert, diagnostics


def run(job):
    start = time.time()
    data = base.load_data(job['dataset'])
    weights, velocity, idx, sizes, population = base.initial_state(data, job['n'], job.get('alpha'), job['seed'])
    weights, velocity = weights.clone(), velocity.clone()
    x, y, xp, xt, yt, index, sizes_t = [torch.as_tensor(value, device=base.DEVICE) for value in
        [data['x_private'], data['y_private'], data['x_proxy'], data['x_test'], data['y_test'], idx, sizes]]
    initial, initial_balanced = base.evaluate(weights, xt, yt)
    trace, diagnostic = [], {}
    targets = None
    reference = torch.zeros_like(weights[0])
    local_endpoint = None
    for r in range(1, 4):
        if r > 1:
            weights, velocity = train_round(weights, velocity, x, y, index, sizes_t, job['seed'], r, xp, targets)
        if job['method'] == 'Local-only':
            continue
        if job['method'] == 'DP-Head':
            if r == 3:
                a, b = base.evaluate(weights, xt, yt)
                local_endpoint = dict(accuracy=a, balanced_accuracy=b, optimizer_steps=180, extra_adaptation_steps=0)
            reference, cert, info = base.parameter_release(weights, reference, job['clip'], job['epsilon'], job['seed'], r)
            weights = reference[None].expand(job['n'], -1, -1).clone()
            velocity.zero_()
            targets = (xp @ reference).argmax(-1)
        else:
            targets, cert, info = release(weights, data, job, r, diagnostic)
        trace.append(dict(round=r, privacy=cert, diagnostics=info))
    acc, balanced = base.evaluate(weights, xt, yt)
    result = dict(**job, actual_clients=job['n'], calibrated_clients=job['n'], delta=1e-5, accuracy=acc,
        balanced_accuracy=balanced, initial_accuracy=initial, initial_balanced_accuracy=initial_balanced,
        local_endpoint=local_endpoint, population=population, trace=trace, per_class_recall=base.evaluate.last_recall,
        optimizer_steps=180, batch_size=128, third_label_release_used_by_final_local_model=False,
        diagnostics_are_offline_benchmark_measurements=True,
        public_configuration_labels=(len(data['y_development'])+len(data['y_validation'])) if job['method']=='DP-Head' else 0,
        public_development_labels_for_configuration=len(data['y_development']) if job['method']=='DP-Head' else 0,
        public_validation_labels_for_configuration=len(data['y_validation']) if job['method']=='DP-Head' else 0,
        public_training_labels=0, private_outcomes_select_decoder=False, seconds=time.time()-start)
    if diagnostic:
        file = ROOT / 'diagnostics' / (key(job) + '.npz')
        np.savez_compressed(file, **diagnostic)
        result['diagnostic_file'] = str(file.relative_to(ROOT))
        result['diagnostic_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    save(ROOT/'results'/(key(job)+'.json'), result)
    print(key(job), round(acc*100, 4), round(balanced*100, 4), round(result['seconds'], 1), flush=True)
    return result
