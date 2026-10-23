"""Paired numerical-noise experiments on frozen round-1 reports; no retraining.

Replays the implementation's rounded continuous Gaussian shares at its actual
16-bit integer grid. Results measure pseudo-labels conditional on these fixed
reports, separately from end-to-end client accuracy and training-seed tests.
"""
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dp_utils import DPConfig, project_rows_to_simplex, quantize

HERE = Path(__file__).resolve().parent
TRIALS = 20000
CHUNK = 200
REPLAY_SEED = 20261009


def summarise(values):
    a = np.asarray(values)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    return {"mean": mean, "std": sd,
            "mean_ci95": [mean - 1.96 * sd / np.sqrt(len(a)),
                          mean + 1.96 * sd / np.sqrt(len(a))]}


def main():
    torch.set_num_threads(4)
    cache = ROOT / "logs/dp_v2/round1_clean_reports_K10.npy"
    report = torch.from_numpy(np.load(cache)).float()
    n, k, c = report.shape
    assert (n, k, c) == (100, 10, 10)
    votes = torch.nn.functional.one_hot(report.argmax(-1), c).float()
    clean = votes.mean(0)
    clean_labels = clean.argmax(-1)
    ordered_vote = votes.double().mean(0).sort(-1, descending=True).values
    vote_margins = (ordered_vote[:, 0] - ordered_vote[:, 1]).numpy()
    ordered_posterior = report.double().mean(0).sort(-1, descending=True).values
    posterior_margins = (ordered_posterior[:, 0] - ordered_posterior[:, 1]).numpy()
    reference_sigma = DPConfig(epsilon=1.0, delta=1e-5, q=1.0,
                              num_clusters=k, num_rounds=3, cohort_size=10000).sigma_aggregate
    clusters_path = ROOT / "data/cifar10/cluster_result/10_C_2000.npy"
    indices_path = ROOT / "data/cifar10/proxy_cifar10_src_idx_2000.npy"
    cluster_ids = np.load(clusters_path)
    indices = np.load(indices_path)
    with (ROOT / "data/cifar-10-batches-py/test_batch").open("rb") as stream:
        dataset = pickle.load(stream, encoding="latin1")
    true_labels = np.asarray(dataset["labels"])[indices]
    assert np.array_equal(true_labels, np.load(ROOT / "data/cifar10/proxy_cifar10_orig_labels_2000.npy"))
    counts = np.zeros((k, c), dtype=np.int64)
    np.add.at(counts, (cluster_ids, true_labels), 1)
    counts_t = torch.from_numpy(counts)
    configs = [(f"cohort{cohort}", 1.0, cohort) for cohort in [100, 300, 1000, 3000, 10000]]
    configs += [("eps0_25", .25, 10000), ("eps0_5", .5, 10000)]
    scale = 1 << 16
    encoded_sum = quantize(votes, scale).sum(0)
    accum = {name: {"accuracy": [], "flips": [], "predictions": []}
             for name, _, _ in configs}
    calibration = {}
    generator = torch.Generator().manual_seed(REPLAY_SEED)
    for start in range(0, TRIALS, CHUNK):
        batch = min(CHUNK, TRIALS - start)
        # The shared normal variates couple the conditions within each repeat.
        normal = torch.randn((batch, n, k, c), dtype=torch.float64, generator=generator)
        for name, eps, cohort in configs:
            dp = DPConfig(epsilon=eps, delta=1e-5, q=1.0,
                          num_clusters=k, num_rounds=3, cohort_size=cohort,
                          quantization_bits=16)
            share_std = dp.client_noise_std(n)
            integer_noise = torch.round(normal * (share_std * scale)).to(torch.int64).sum(1)
            total = encoded_sum.unsqueeze(0) + integer_noise
            avg = (total.to(torch.float64) / scale).float() / n
            projected = project_rows_to_simplex(avg.reshape(-1, c)).reshape(batch, k, c)
            pred = projected.argmax(-1)
            correct = counts_t[torch.arange(k).unsqueeze(0), pred].sum(-1)
            accum[name]["accuracy"].extend((correct.double() / len(cluster_ids)).tolist())
            accum[name]["flips"].extend((pred != clean_labels).float().mean(-1).tolist())
            accum[name]["predictions"].extend(pred.tolist())
            calibration[name] = {"epsilon": eps, "cohort": cohort,
                                 "aggregate_sigma": dp.sigma_aggregate,
                                 "residual_std": dp.noise_std_on_average(n)}
    result = {
        "scope": "fixed seed-42 round-1 reports, paired rounded-Gaussian noise replay",
        "trials": TRIALS, "replay_seed": REPLAY_SEED,
        "trained_clients": n, "K": k, "C": c, "T": 3, "quantization_bits": 16,
        "sampler": "independent float64 Gaussian draws rounded to integer noise per client",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [cache, clusters_path, indices_path]},
        "clean_vote_pseudo_label_accuracy": float(counts_t[torch.arange(k), clean_labels].sum()) / len(cluster_ids),
        "clean_vote_margins": (clean.sort(-1, descending=True).values[:, 0] - clean.sort(-1, descending=True).values[:, 1]).tolist(),
        "margin_summary": {
            "median_convention": "arithmetic mean of the two middle observations for even K",
            "vote_median": float(np.median(vote_margins)),
            "mean_posterior_median": float(np.median(posterior_margins)),
            "legacy_posterior_lower_median": float(np.sort(posterior_margins)[(k-1)//2]),
            "sigma": reference_sigma,
            "vote_operating_cohort": reference_sigma / float(np.median(vote_margins)),
            "vote_operating_cohort_rounded": round(reference_sigma / float(np.median(vote_margins)), -1),
        },
        "conditions": {}, "paired_comparisons": {},
    }
    for name, _, _ in configs:
        result["conditions"][name] = dict(calibration[name],
            pseudo_label_accuracy=summarise(accum[name]["accuracy"]),
            cluster_flip_fraction=summarise(accum[name]["flips"]))
    for left, right in [("cohort1000", "cohort3000"), ("eps0_25", "eps0_5")]:
        a, b = accum[left], accum[right]
        differences = 100 * (np.array(b["accuracy"]) - np.array(a["accuracy"]))
        identical = np.all(np.array(a["predictions"]) == np.array(b["predictions"]), axis=1)
        result["paired_comparisons"][f"{right} minus {left}"] = {
            "pseudo_label_gain_points": summarise(differences),
            "fraction_all_cluster_labels_identical": float(identical.mean()),
            "fraction_accuracy_lower": float((differences < 0).mean()),
            "fraction_accuracy_higher": float((differences > 0).mean()),
            "fraction_accuracy_equal": float((differences == 0).mean()),
        }
    (HERE / "fixed_report_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(HERE / "fixed_report_replay_samples.npz",
                        **{f"{name}_accuracy": np.array(a["accuracy"]) for name, a in accum.items()},
                        **{f"{name}_labels": np.array(a["predictions"], dtype=np.int8) for name, a in accum.items()})
    print(json.dumps({"trials": TRIALS, "conditions": result["conditions"],
                      "paired_comparisons": result["paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
