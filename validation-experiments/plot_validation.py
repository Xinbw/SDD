"""Plot source-derived noise replays and completed paired training results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    r = json.loads((HERE / "fixed_report_replay.json").read_text())
    cohorts = np.array([100, 300, 1000, 3000, 10000])
    rows = [r["conditions"][f"cohort{n}"] for n in cohorts]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    for ax, key, label, color in zip(axes,
        ["pseudo_label_accuracy", "cluster_flip_fraction"],
        ["Mean pseudo-label accuracy", "Mean fraction of clusters changed"],
        ["#0072B2", "#D55E00"]):
        mean = np.array([row[key]["mean"] for row in rows])
        ci = np.array([row[key]["mean_ci95"] for row in rows])
        ax.errorbar(cohorts, mean, yerr=np.stack([mean-ci[:, 0], ci[:, 1]-mean]),
                    fmt="o-", color=color, capsize=3, lw=1.5)
        ax.set_xscale("log")
        ax.set_xticks(cohorts, ["100", "300", "1k", "3k", "10k"])
        ax.set_xlabel("Cohort used for residual-noise calibration")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.25)
    axes[0].axhline(r["clean_vote_pseudo_label_accuracy"], color="gray", ls=":", lw=1,
                   label="Fixed clean vote consensus")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("20,000 paired noise repeats on fixed round-1 reports", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "validation-noise-replay.pdf", bbox_inches="tight")
    fig.savefig(HERE / "validation-noise-replay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    a = json.loads((HERE / "analysis_results.json").read_text())
    if "cohort_3000_minus_1000" not in a:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    seeds = a["cohort_validation"]["1000"]["seeds"]
    left, right = [a["cohort_validation"][str(n)]["values"] for n in [1000, 3000]]
    for i, seed in enumerate(seeds):
        ax.plot([0, 1], [100*left[i], 100*right[i]], "o-", label=f"seed {seed}")
    ax.set_xticks([0, 1], ["1k", "3k"])
    ax.set_xlabel("Cohort used for residual-noise calibration")
    ax.set_ylabel("Final mean client accuracy (%)")
    ax.set_xlim(-.15, 1.15)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Matched full-training repeats")
    fig.tight_layout()
    fig.savefig(HERE / "validation-paired-training.pdf", bbox_inches="tight")
    fig.savefig(HERE / "validation-paired-training.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
