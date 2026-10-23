"""Conditional regional-label model, distinct from downstream model accuracy."""
import numpy as np


def project(values):
    order = np.sort(values, axis=-1)[..., ::-1]
    cssv = np.cumsum(order, axis=-1)-1
    ranks = np.arange(1, values.shape[-1]+1)
    active = order-cssv/ranks > 0
    rho = np.maximum(active.sum(-1), 1)
    theta = np.take_along_axis(cssv, (rho-1)[..., None], axis=-1)[..., 0]/rho
    return np.maximum(values-theta[..., None], 0)


def labels(values, weights, mode):
    a = project(values)
    if mode == 'center':
        a = a-(a*weights[..., None]).sum(-2, keepdims=True)/weights.sum()
    return a.argmax(-1)


def quality(decoded, truth_counts):
    """Exact accuracy of constant regional labels on the specified labelled set."""
    return truth_counts[np.arange(len(truth_counts)), decoded].sum(-1)/truth_counts.sum()


def predict(q, truth_counts, public_weights, sigma, mode='center', repeats=4096, seed=91001):
    rng = np.random.default_rng(seed)
    sample = q[None]+sigma*rng.standard_normal((repeats,)+q.shape)
    decoded = labels(sample, public_weights, mode)
    values = quality(decoded, truth_counts)
    prob = np.eye(q.shape[1])[decoded].mean(0)
    clean = labels(q, public_weights, mode)
    return dict(predicted_accuracy=float(values.mean()), gaussian_mc_sem=float(values.std(ddof=1)/np.sqrt(repeats)),
        purity=float(truth_counts.max(1).sum()/truth_counts.sum()),
        clean_accuracy=float(quality(clean, truth_counts)),
        expected_flip_fraction=float((decoded!=clean[None]).mean()),
        class_probabilities=prob.tolist())


def raw_lower_bound(q, truth, sigma):
    """Sub-Gaussian competitor bound, valid for uncentered coordinate argmax."""
    top=q.argmax(1)
    fractions=truth/np.maximum(truth.sum(1,keepdims=True),1)
    margins=q[np.arange(len(q)),top,None]-q
    tail=np.exp(-np.maximum(margins,0)**2/(4*sigma*sigma))
    loss=np.maximum(fractions[np.arange(len(q)),top,None]-fractions,0)*tail
    per_region=np.maximum(fractions[np.arange(len(q)),top]-loss.sum(1),0)
    return float((per_region*truth.sum(1)).sum()/truth.sum())
