"""Consolidated three-way comparison: SSD vs DP-FedCT vs PATE.

One table per (dataset, alpha) combination, seven epsilon points each,
mean client accuracy averaged over whatever seeds exist.  Everything is
read straight from the logs -- no hand-copied numbers.

The per-row K is the same one used by export_ssd_table.py (see notes 36b /
33 / 39a / 40a+40f).  Baseline runs have no K.
"""

import glob
import os
import re
import statistics as st

MEAN = re.compile(r"Final mean (?:client )?accuracy: ([\d.]+)")
SEEDS = [42, 43, 44]
EPS = ["0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1.0"]

MNIST_K = {"0.01": 5, "0.025": 20, "0.05": 20, "0.1": 50,
           "0.25": 100, "0.5": 500, "1.0": 1000}
CIFAR_K = {e: 10 for e in EPS}
CIFAR_A1_K = {e: 20 for e in EPS}
MNIST_A1_K = {"0.01": 5, "0.025": 20, "0.05": 100, "0.1": 100,
              "0.25": 2000, "0.5": 500, "1.0": 500}


def tag(e):
    return e.replace(".", "_")


def read(path):
    if not os.path.isfile(path):
        return None
    m = MEAN.search(open(path, errors="ignore").read())
    return float(m.group(1)) if m else None


def first(patterns):
    """First existing log among the candidates (glob allowed)."""
    for p in patterns:
        for hit in sorted(glob.glob(p)):
            v = read(hit)
            if v is not None:
                return v
    return None


# --- per-combination log layout -------------------------------------------
# CIFAR alpha=0.1 runs are scattered over several sweep directories; the
# alpha=1.0 and MNIST alpha=1.0 runs each live in one directory.

def cifar_ssd(e, K, s):
    return [f"logs/dp_v2/{d}/ours_K{K}_eps{tag(e)}_s{s}.log"
            for d in ("ultralow", "k_eps_grid", "followups", "k_at_low_eps")]


def cifar_fedct(e, _K, s):
    return [f"logs/dp_v2/ultralow/fedct_eps{tag(e)}_s{s}.log",
            f"logs/dp_v2/eps_sweep_seeds/fedct_s{s}_eps{tag(e)}.log"] + \
           ([f"logs/dp_v2/eps_sweep_main/fedct_eps{tag(e)}.log"] if s == 42 else [])


def cifar_pate(e, _K, s):
    return [f"logs/dp_v2/pate/pate_cifar_eps{tag(e)}_s{s}.log"]


def mnist_ssd(e, K, s):
    return [f"logs/dp_v2/mnist/ours_K{K}_eps{tag(e)}_s{s}.log"]


def mnist_fedct(e, _K, s):
    return [f"logs/dp_v2/mnist/fedct_eps{tag(e)}_s{s}.log"]


def mnist_pate(e, _K, s):
    return [f"logs/dp_v2/pate/pate_mnist_eps{tag(e)}_s{s}.log"]


def a1_dir(d):
    def ssd(e, K, s):
        return [f"logs/dp_v2/{d}/ours_K{K}_eps{tag(e)}_s{s}.log"]

    def fedct(e, _K, s):
        return [f"logs/dp_v2/{d}/fedct_eps{tag(e)}_s{s}.log"]

    def pate(e, _K, s):
        return [f"logs/dp_v2/{d}/pate_eps{tag(e)}_s{s}.log"]
    return ssd, fedct, pate


COMBOS = [
    ("CIFAR-10, α=0.1（强 non-IID）", CIFAR_K,
     (cifar_ssd, cifar_fedct, cifar_pate), "§28 / §34 / §37"),
    ("MNIST, α=0.1（强 non-IID）", MNIST_K,
     (mnist_ssd, mnist_fedct, mnist_pate), "§36 / §37j"),
    ("CIFAR-10, α=1.0（温和 non-IID）", CIFAR_A1_K,
     a1_dir("alpha1"), "§39"),
    ("MNIST, α=1.0（温和 non-IID）", MNIST_A1_K,
     a1_dir("alpha1_mnist"), "§40 / §40e / §40f"),
]


def series(e, K, pathfn):
    vals = []
    for s in SEEDS:
        v = first(pathfn(e, K, s))
        if v is not None:
            vals.append(v)
    return vals


def cell(vals):
    if not vals:
        return "—", 0
    if len(vals) == 1:
        return f"{vals[0]:.4f}", 1
    return f"{st.mean(vals):.4f} ± {st.stdev(vals):.4f}", len(vals)


def delta(a, b):
    """Paired difference in points, over the seeds both methods have."""
    n = min(len(a), len(b))
    if n == 0:
        return "—"
    d = [(x - y) * 100 for x, y in zip(a[:n], b[:n])]
    return f"{st.mean(d):+.2f}"


def main():
    out = ["# SSD vs DP-FedCT vs PATE —— 合并对比表", "",
           "由 `python export_comparison.py` 从日志直接生成，无手抄数字。",
           "指标为**全体客户端平均准确率**（`acc_avg`），"
           "多 seed 时写作 mean ± std（seed 42/43/44）。",
           "`Δ` 列为配对差值（百分点），只在两方都有的 seed 上求。",
           "100 客户端，3 轮 × 20 epoch，cohort 10000（模拟），δ=1e-5。", "",
           "> **PATE 是下界**：数据相关会计（Prop.8 + smooth sensitivity）未实现。",
           "> 在 α=0.1 实测无收益（§37h），在 α=1.0 有收益（§39d）⇒ "
           "α=1.0 两表的 PATE 真值会更高。", ""]
    for name, kmap, (fs, ff, fp), ref in COMBOS:
        out += [f"## {name}", "", f"记录见 notes {ref}。", "",
                "| ε | K | SSD | DP-FedCT | PATE | Δ vs FedCT | Δ vs PATE | seeds |",
                "|---|---|---|---|---|---|---|---|"]
        for e in EPS:
            K = kmap[e]
            s, f, p = series(e, K, fs), series(e, K, ff), series(e, K, fp)
            cs, ns = cell(s)
            cf, nf = cell(f)
            cp, np_ = cell(p)
            out.append(f"| {e} | {K} | **{cs}** | {cf} | {cp} | "
                       f"{delta(s, f)} | {delta(s, p)} | {ns}/{nf}/{np_} |")
        out.append("")
    out += ["---", "",
            "`seeds` 列为 SSD/DP-FedCT/PATE 各自的 run 数。1 表示单 seed 筛查口径，",
            "其 std 无法计算（不代表方差为零）。逐 seed 明细与配对 t 检验见 "
            "`SDD_PHASE1_NOTES.md`。", "",
            "客户端维度的 min/avg/max 分解见 `SSD_RESULTS.md`（仅 SSD）。", "",
            "## 附：未进主表的两个自跑基线（FedDPA / PPFed）", "",
            "均为 CIFAR-10 α=0.1、100 客户端、seed 42。判定依据见 notes §42，"
            "这里只留结论，免得再被当成待补的缺口。", "",
            "| 基线 | 实际跑到的 ε | 最好 acc_avg | 不进表的原因 |",
            "|---|---|---|---|",
            "| FedDPA | 0.5 ~ 25（三轮调参） | **0.155**（ε=25） | "
            "全部 ε 停在随机水平，ε=25 仍低于 SSD/DP-FedCT 在 ε=0.01 的值；"
            "复现未跑起来，放进去是稻草人 |",
            "| PPFed | ≈6 ~ 4000（按 σ 扫，RDP 折算） | **0.256**（σ=0.05，ε≈4000） | "
            "最私密点 ε≈6 与主表 ε≤1 无交集；且无噪声天花板 25.6% 已低于 "
            "DP-FedCT ε=0.01；脚本无隐私会计 |", "",
            "要让任一者进表，须先在无噪声下把天花板调到 ≥40%，再按 ε 反解噪声（§42a/§42b）。"]
    txt = "\n".join(out) + "\n"
    open("SSD_COMPARISON.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
