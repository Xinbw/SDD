#!/usr/bin/env python3
"""Regenerate the three paper figures from frozen result snapshots.

Main values are parsed from results/SSD_COMPARISON.md. Mechanism and diagnostic
arrays preserve the packaged cohort, FedAvg and round-sweep snapshots. Learning
payload counts and deployment-noise calibration are identified explicitly.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
PAPER_OUT = OUT.parent / "paper" / "figures"
EPS = np.array([.01, .025, .05, .1, .25, .5, 1.0])
COLORS = {"SDD": "#0072B2", "DP-FedCT": "#D55E00", "PATE": "#009E73"}
MARKERS = {"SDD": "o", "DP-FedCT": "s", "PATE": "^"}
DISPLAY_NAMES = {"SDD": "SDD", "DP-FedCT": "DP-FedCT", "PATE": "PATE-DI"}

def read_main():
    import re
    result = {}
    current = None
    for line in (OUT.parent / "results" / "SSD_COMPARISON.md").read_text().splitlines():
        heading = re.match(r"## (CIFAR-10|MNIST), α=(0\.1|1\.0)", line)
        if heading:
            current = (heading[1], f"$\\alpha={heading[2]}$")
            result[current] = {name: ([], []) for name in COLORS}
        if current and line.startswith("|"):
            cols = [x.strip().replace("**", "") for x in line.strip("|").split("|")]
            if not cols or not re.fullmatch(r"0?\.\d+|1\.0", cols[0]):
                continue
            for name, cell in zip(COLORS, cols[2:5]):
                mean, std = map(float, cell.split("±"))
                result[current][name][0].append(mean)
                result[current][name][1].append(std)
    # Preserve the panel order in the original three-figure package.
    order = [("CIFAR-10", "$\\alpha=0.1$"), ("MNIST", "$\\alpha=0.1$"),
             ("CIFAR-10", "$\\alpha=1.0$"), ("MNIST", "$\\alpha=1.0$")]
    return {key: result[key] for key in order}


MAIN = read_main()


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem):
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main_curves():
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.3), sharex=True)
    panels = ["(a)", "(b)", "(c)", "(d)"]
    for idx, (ax, ((dataset, alpha), methods)) in enumerate(zip(axes.flat, MAIN.items())):
        for name, (mean, std) in methods.items():
            mean, std = np.asarray(mean), np.asarray(std)
            ax.plot(EPS, mean, label=DISPLAY_NAMES[name], color=COLORS[name], marker=MARKERS[name], lw=1.6, ms=4)
            ax.fill_between(EPS, mean-std, mean+std, color=COLORS[name], alpha=.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(EPS, [".01", ".025", ".05", ".1", ".25", ".5", "1"])
        ax.grid(axis="y", color="#dddddd", lw=.6)
        ax.set_title(f"{panels[idx]} {dataset}, {alpha}")
        ax.set_ylabel("Mean client accuracy")
    axes[0,0].annotate("+14.2 points vs DP-FedCT", xy=(.1,.4229), xytext=(.013,.466),
                       fontsize=7, color=COLORS["SDD"],
                       arrowprops={"arrowstyle":"-", "color":COLORS["SDD"], "lw":.8})
    axes[0,0].set_ylim(.23,.50)
    axes[0,1].annotate("+45.1 points vs DP-FedCT", xy=(.05,.8617), xytext=(.011,1.005),
                       fontsize=7, color=COLORS["SDD"],
                       arrowprops={"arrowstyle":"-", "color":COLORS["SDD"], "lw":.8})
    axes[0,1].set_ylim(.33,1.075)
    axes[1,1].axhline(.9, color="#777777", ls=":", lw=.8)
    axes[1,1].text(.029,.753,"90% target: 10x smaller budget\nSDD .025; DP-FedCT .25", fontsize=7,
                   color=COLORS["SDD"], bbox={"facecolor":"white","alpha":.88,"edgecolor":"none","pad":2})
    for ax in axes[-1]:
        ax.set_xlabel(r"Whole-run privacy budget $\varepsilon$")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "fig-main-curves")


def mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5))
    n = np.array([100, 300, 1000, 3000, 10000])
    import csv, json
    validation = OUT.parent / "validation-experiments"
    with (validation / "cohort_curve.csv").open() as stream:
        cohort_rows = list(csv.DictReader(stream))
    acc = np.array([float(row["mean_accuracy"]) for row in cohort_rows])
    std = np.array([float(row["std_across_seeds"]) if row["std_across_seeds"] else np.nan for row in cohort_rows])
    cohort_counts = np.array([int(row["n_seeds"]) for row in cohort_rows])
    operating_n = json.loads((validation / "fixed_report_replay.json").read_text())["margin_summary"]["vote_operating_cohort_rounded"]
    ax = axes[0]
    ax.plot(n, acc, color=COLORS["SDD"], marker="o", lw=1.8)
    multi_seed = cohort_counts > 1
    ax.errorbar(n[multi_seed], acc[multi_seed], yerr=std[multi_seed], fmt="none",
                color=COLORS["SDD"], capsize=3, lw=.9)
    ax.axvline(operating_n, color="#555555", ls="--", lw=1,
               label=f"vote-margin estimate: n≈{operating_n:.0f}")
    ax.set_xscale("log"); ax.set_xticks(n, ["100", "300", "1k", "3k", "10k"])
    ax.set_xlabel(r"Cohort $n$ (noise calibration)"); ax.set_ylabel("Mean client accuracy")
    ax.set_title("(a) Consensus operating scale")
    ax.legend(frameon=True, facecolor="white", edgecolor="white", framealpha=.97,
              loc="lower right", fontsize=7); ax.grid(axis="y", color="#dddddd", lw=.6)

    eps = np.array([.01, .1, 1, 4, 10])
    # One fixed schedule for the curve and noise ratio: T=3, E=20.
    fedavg = np.array([.1028, .1028, .0998, .1510, .2764])
    ratio = np.array([556.0, 55.7, 5.7, 1.5, .66])
    ax = axes[1]
    ax.plot(eps, fedavg, color="#CC79A7", marker="o", lw=1.8, label=r"DP-FedAvg, $T=3$")
    ax.scatter([1,10], [.1028,.3408], color="#882255", marker="^", s=26,
               label=r"DP-FedAvg, $T=20$", zorder=4)
    ax.axhline(.1, color="#777777", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel(r"Whole-run privacy budget $\varepsilon$")
    ax.set_ylabel("Accuracy"); ax.set_title("(b) Model-update noise scale")
    ax.set_ylim(.085,.38)
    ax.grid(axis="y", color="#dddddd", lw=.6)
    twin = ax.twinx()
    twin.plot(eps, ratio, color="#E69F00", marker="s", ls="--", lw=1.5, label=r"Noise / signal, $T=3$")
    twin.set_yscale("log"); twin.axhline(1, color="#E69F00", ls=":", lw=1)
    twin.set_ylabel("Noise / clipped-signal ratio")
    lines, labels = ax.get_legend_handles_labels()
    more, more_labels = twin.get_legend_handles_labels()
    ax.legend(lines+more, labels+more_labels, frameon=True, facecolor="white", framealpha=.9, edgecolor="white", loc="center left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig-mechanism")


def ablations():
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.5))
    ax = axes[0]
    labels=[r"$\varepsilon=.1$", r"$\varepsilon=1$"]
    hard=[.4202,.4280]; soft=[.3664,.3953]
    x=np.arange(2); w=.34
    ax.bar(x-w/2, hard, w, color=COLORS["SDD"], label="Hard CE")
    ax.bar(x+w/2, soft, w, color="#BBBBBB", label="Soft KL")
    for j, gap in enumerate([5.4,3.3]):
        ax.text(j,.455,f"+{gap:.1f} points",ha="center",fontsize=7,color=COLORS["SDD"])
    ax.set_xticks(x, labels); ax.set_ylim(0,.56); ax.set_ylabel("Accuracy")
    ax.set_title("(a) Extracting consensus")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.3,
              handlelength=1.2, columnspacing=.8)
    ax.grid(axis="y", color="#dddddd", lw=.6); ax.set_axisbelow(True)

    ax = axes[1]
    k = np.array([5,10,20,50,100]); vals=np.array([.3903,.4280,.4064,.4261,.4348])
    ax.plot(k, vals, marker="o", color=COLORS["SDD"], lw=1.6)
    ax.scatter([10],[.4280],s=65,facecolors="white",edgecolors=COLORS["SDD"],zorder=4)
    ax.annotate("100 report coordinates",xy=(10,.4280),xytext=(5.1,.442),fontsize=6.8,
                arrowprops={"arrowstyle":"-","lw":.6,"color":"#555555"})
    ax.set_ylim(.38,.452)
    ax.set_xscale("log"); ax.set_xticks(k, ["5","10","20","50","100"])
    ax.set_xlabel("Clusters $K$"); ax.set_ylabel("Accuracy")
    ax.set_title(r"(b) Compact granularity ($\varepsilon=1$)")
    ax.grid(axis="y", color="#dddddd", lw=.6)

    ax = axes[2]
    T = np.array([2,3,5,8])
    series=[(r"CIFAR, $\varepsilon=.1$", [.3382,.4229,.4631,.4557], [.0094,.0337], "#0072B2"),
            (r"CIFAR, $\varepsilon=1$", [.3425,.4473,.4991,.4790], [.0257,.0419], "#D55E00"),
            (r"MNIST, $\varepsilon=.1$", [.8819,.9176,.8283,.7787], [.0183], "#009E73")]
    for label, vals, std, color in series:
        ax.plot(T, vals, marker="o", lw=1.4, label=label, color=color, ms=3.5)
        multi_seed_t=[3,5] if len(std)==2 else [3]
        multi_seed_v=[vals[list(T).index(t)] for t in multi_seed_t]
        ax.errorbar(multi_seed_t,multi_seed_v,yerr=std,fmt="none",color=color,capsize=2,lw=.8)
    ax.set_xlabel("Release rounds $T$"); ax.set_ylabel("Accuracy")
    ax.set_title("(c) Allocating a fixed budget")
    ax.legend(frameon=False, fontsize=6.5); ax.grid(axis="y", color="#dddddd", lw=.6)
    fig.tight_layout()
    save(fig, "fig-ablations")


if __name__ == "__main__":
    style(); main_curves(); mechanism(); ablations()
