"""Export the SSD results as a paper-ready table.

Columns (client-dimension reading, chosen by the user 2026-08-21):
  epsilon_total  the whole-run central budget
  acc_min        worst-client accuracy, averaged over seeds
  acc_avg        mean client accuracy, averaged over seeds
  acc_max        best-client accuracy, averaged over seeds
  min_std        std ACROSS SEEDS of the worst-client accuracy
  avg_std        std ACROSS SEEDS of the mean client accuracy

Note that min_std is over seeds, not over clients -- the worst client is a
single client per run, so its seed-to-seed spread is large (it can be a
different client each time).  The client-level spread within one run is a
different quantity and is not in this table.

K per epsilon: MNIST uses the argmax found on seed 42 (notes section 36b);
CIFAR-10 uses a fixed K=10 (section 33: the recommended value, and the only
epsilon row whose argmax survived n=3).
"""

import os
import re
import statistics as st

MEAN = re.compile(r"Final mean client accuracy: ([\d.]+)")
BEST = re.compile(r"Best  client(?: accuracy)?[:=]?\s*(?:id=\d+, accuracy=)?([\d.]+)")
WORST = re.compile(r"Worst client(?: accuracy)?[:=]?\s*(?:id=\d+, accuracy=)?([\d.]+)")

SEEDS = [42, 43, 44]
EPS = ["0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1.0"]

# MNIST: argmax K per epsilon (section 36b).  CIFAR-10: fixed K=10 (section 33).
MNIST_K = {"0.01": 5, "0.025": 20, "0.05": 20, "0.1": 50,
           "0.25": 100, "0.5": 500, "1.0": 1000}
CIFAR_K = {e: 10 for e in EPS}
# alpha=1.0: K is a flat region (spread 0.4pt over K in {10,20,50,100}, notes
# 39a), so it is fixed at 20 rather than screened per epsilon.
CIFAR_A1_K = {e: 20 for e in EPS}
# MNIST alpha=1.0: K must be tuned per epsilon -- unlike CIFAR, the clustering
# ceiling climbs steeply with K (0.6120 at K=10 to 0.9850 at K=1000), so K keeps
# binding through the ceiling even where noise no longer bites (notes 40a).
# These are the per-epsilon argmaxes; rows 0.25/0.5/1.0 are flat regions where
# the argmax is nominal (spread 0.7-1.3pt, inside the 2.5pt chaos threshold).
MNIST_A1_K = {"0.01": 5, "0.025": 20, "0.05": 100, "0.1": 100,
              "0.25": 2000, "0.5": 500, "1.0": 500}


def tag(e):
    return e.replace(".", "_")


def parse(path):
    if not os.path.isfile(path):
        return None
    t = open(path, errors="ignore").read()
    m, b, w = MEAN.search(t), BEST.search(t), WORST.search(t)
    if not (m and b and w):
        return None
    return float(m.group(1)), float(b.group(1)), float(w.group(1))


def mnist_paths(e, K, s):
    return [f"logs/dp_v2/mnist/ours_K{K}_eps{tag(e)}_s{s}.log"]


def mnist_a1_paths(e, K, s):
    return [f"logs/dp_v2/alpha1_mnist/ours_K{K}_eps{tag(e)}_s{s}.log"]


def cifar_a1_paths(e, K, s):
    """alpha=1.0 CIFAR runs all live in one directory."""
    return [f"logs/dp_v2/alpha1/ours_K{K}_eps{tag(e)}_s{s}.log"]


def cifar_paths(e, K, s):
    """CIFAR-10 K=10 runs are spread across several sweep directories."""
    return [f"logs/dp_v2/ultralow/ours_K{K}_eps{tag(e)}_s{s}.log",
            f"logs/dp_v2/k_eps_grid/ours_K{K}_eps{tag(e)}_s{s}.log",
            f"logs/dp_v2/followups/ours_K{K}_eps{tag(e)}_s{s}.log",
            f"logs/dp_v2/k_at_low_eps/ours_K{K}_eps{tag(e)}_s{s}.log"]


def collect(e, K, pathfn):
    means, bests, worsts, n = [], [], [], 0
    for s in SEEDS:
        for p in pathfn(e, K, s):
            r = parse(p)
            if r:
                means.append(r[0]); bests.append(r[1]); worsts.append(r[2]); n += 1
                break
    if not means:
        return None
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
    return dict(n=n, acc_min=st.mean(worsts), acc_avg=st.mean(means),
                acc_max=st.mean(bests), min_std=sd(worsts), avg_std=sd(means))


def table(name, kmap, pathfn, note):
    lines = [f"## {name}", "", note, "",
             "| epsilon_total | acc_min | acc_avg | acc_max | min_std | avg_std | K | seeds |",
             "|---|---|---|---|---|---|---|---|"]
    for e in EPS:
        K = kmap[e]
        r = collect(e, K, pathfn)
        if not r:
            lines.append(f"| {e} | — | — | — | — | — | {K} | 0 |")
            continue
        sd = (lambda v: f"{v:.4f}" if r["n"] > 1 else "—")
        lines.append(f"| {e} | {r['acc_min']:.4f} | {r['acc_avg']:.4f} | {r['acc_max']:.4f} | "
                     f"{sd(r['min_std'])} | {sd(r['avg_std'])} | {K} | {r['n']} |")
    return "\n".join(lines)


def main():
    out = ["# SSD 实验结果（导出于 2026-09-01）", "",
           "100 客户端，3 轮 × 20 epoch，cohort 10000（模拟），δ=1e-5，",
           "vote 上报 + 硬标签 CE，置信度阈值 0.0，干净评测口径。",
           "Dirichlet α 见各表标题（α 越小越 non-IID）。", "",
           "**列定义**（客户端维度）：", "",
           "- `acc_min` / `acc_avg` / `acc_max` — 单次 run 内**最差客户端 / 全体均值 / 最好客户端**"
           "的准确率，再对 3 个 seed 取平均",
           "- `min_std` / `avg_std` — 上述 `acc_min` / `acc_avg` 在 **3 个 seed 之间**的标准差",
           "",
           "> `min_std` 通常远大于 `avg_std`：最差客户端每次 run 可能是不同的那一个，",
           "> 因此它在 seed 间的波动天然大。单次 run 内部的客户端间标准差是另一个量，不在本表。",
           ""]
    out.append(table(
        "MNIST（α=0.1，强 non-IID）", MNIST_K, mnist_paths,
        "代理集为 EMNIST-digits 风格的 2000 张（取自 MNIST train split，与评测 test split 不相交）。"
        "K 取各 ε 行上的最优值（seed 42 上的 argmax）。"))
    out.append("")
    out.append(table(
        "CIFAR-10（α=0.1，强 non-IID）", CIFAR_K, cifar_paths,
        "K 固定为 10（推荐值：在 ε=0.1 处 3/3 seed 同向胜出，且免去用测试集选 K 的事后选择）。"))
    out.append("")
    out.append(table(
        "CIFAR-10（α=1.0，温和 non-IID）", CIFAR_A1_K, cifar_a1_paths,
        "K 固定为 20：α=1.0 下 K 是平区（K∈{10,20,50,100} 极差仅 0.4 pt，见 notes §39a），"
        "逐 ε 筛 argmax 只会筛出噪声。"))
    out.append("")
    out.append(table(
        "MNIST（α=1.0，温和 non-IID）", MNIST_A1_K, mnist_a1_paths,
        "K 逐 ε 取最优（见 notes §40a/§40f）。**ε≥0.25 三行是平区**（极差 0.7~1.3 pt，在 §28a 的 "
        "2.5 pt 阈值内），其 K 只是名义 argmax，不应当作真最优引用。ε=0.05 行的最优区间是 K∈[50,100]（两者仅差 0.11 pt），沿用 K=100。"))
    out.append("")
    out.append("---")
    out.append("")
    out.append("`K` 列为该行使用的簇数；`seeds` 列为实际参与统计的 seed 数。")
    out.append("")
    out.append("**`seeds`=1 的表为单 seed 筛查口径**，其 `min_std` / `avg_std` 无法计算（记为 —），")
    out.append("不代表方差为零。要写进论文需补 seed 43/44。")
    txt = "\n".join(out) + "\n"
    open("SSD_RESULTS.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
