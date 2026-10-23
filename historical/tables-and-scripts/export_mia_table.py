"""SSD_MIA.md: aggregate-level membership inference, four combinations x seven eps.

Reads the JSON files mia_aggregate.py writes under logs/dp_v2/mia_agg/ -- no
hand-copied numbers.  One row per (combination, eps); the K of a row is the one
the main tables use (export_comparison.py), so the JSON for that K is the source.
"""

import glob
import json
import os

from export_comparison import CIFAR_A1_K, CIFAR_K, EPS, MNIST_A1_K, MNIST_K

DIR = "logs/dp_v2/mia_agg"
COMBOS = [
    ("CIFAR-10, α=0.1（强 non-IID）", "cifar10", 0.1, CIFAR_K),
    ("MNIST, α=0.1（强 non-IID）", "mnist", 0.1, MNIST_K),
    ("CIFAR-10, α=1.0（温和 non-IID）", "cifar10", 1.0, CIFAR_A1_K),
    ("MNIST, α=1.0（温和 non-IID）", "mnist", 1.0, MNIST_A1_K),
]


def load(dataset, alpha, K):
    hits = glob.glob(f"{DIR}/mia_agg_{dataset}_a{alpha}_K{K}_s*.json")
    return json.load(open(hits[0])) if hits else None


def main():
    out = ["# SSD 聚合级成员推断（MIA）—— 强对手下的最优攻击", "",
           "由 `python export_mia_table.py` 从 `logs/dp_v2/mia_agg/*.json`"
           "（`mia_aggregate.py` 输出）生成，无手抄数字。方法与解读见 notes §43。", "",
           "**攻击对象**：服务器每轮看到的噪声聚合 S = Σ_i Y_i（威胁模型 §2），T=3 轮。",
           "**对手**：与 adjacency 定义一致的强对手——知道其他所有客户端的报告，判别队列里是客户端 i 还是 j。",
           "对高斯机制这是精确的 μ-GDP 检验，μ_ij = ‖Z_i − Z_j‖₂·√T/σ，"
           "TPR(FPR) = Φ(Φ⁻¹(FPR) + μ)。报告用主线的 one-hot 投票（round-1 检查点，seed 42）。",
           "**与队列大小无关**：σ 加在和上，n=100 与 n=10000 的强对手问题相同。", "",
           "**列定义**：`d_max/Δ₂` = 最坏客户端对的实际 L2 距离 / 灵敏度上界 √(2K)；"
           "`μ_bound` 按上界（= √(2ρ_total)，只依赖 ε）；`max` 列是最坏客户端对、`mean` 列是全部 4950 对的均值；"
           "`ε_GDP` 是把 μ 精确换算回 (ε, δ=1e-5) 的值（bound / 最坏对）。随机猜测：AUC 0.5、TPR@1%FPR 0.010、TPR@0.1%FPR 0.001。", ""]
    for name, ds, a, kmap in COMBOS:
        out += [f"## {name}", "",
                "| ε | K | d_max/Δ₂ | d_mean/Δ₂ | μ_bound | μ_max | AUC_max | TPR@1%FPR max / mean | TPR@0.1%FPR max | ε_GDP bound / 最坏对 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for e in EPS:
            K = kmap[e]
            j = load(ds, a, K)
            row = next((r for r in (j["rows"] if j else []) if f"{r['epsilon']:g}" == f"{float(e):g}"), None)
            if row is None:
                out.append(f"| {e} | {K} | — | — | — | — | — | — | — | — |")
                continue
            d = j["distance"]
            out.append(
                f"| {e} | {K} | {d['max'] / d['bound']:.3f} | {d['mean'] / d['bound']:.3f} | "
                f"{row['mu_bound']:.4f} | {row['mu_max']:.4f} | {row['auc_max']:.4f} | "
                f"{row['tpr@0.01_max']:.4f} / {row['tpr@0.01_mean']:.4f} | {row['tpr@0.001_max']:.4f} | "
                f"{row['eps_gdp_bound']:.3f} / {row['eps_gdp_max']:.3f} |")
        out.append("")
    # discretisation check, one line per json
    out += ["---", "", "**闭式 vs 离散高斯 + 16-bit 量化蒙特卡洛**（各文件在其最大 ε 处，TPR@1%FPR，20 对 × 20000 次）：", ""]
    for f in sorted(glob.glob(f"{DIR}/mia_agg_*.json")):
        j = json.load(open(f))
        mc = j["mc_check"]
        out.append(f"- {j['dataset']} α={j['alpha']} K={j['K']} @ ε={mc['epsilon']:g}: "
                   f"模拟 {mc['tpr_mc_mean']:.4f} vs 闭式 {mc['tpr_closed_form_mean']:.4f}")
    out += ["", "旧的 `mia_attack.py`（攻击客户端本地最终模型）不在威胁模型内，其结果不再引用（notes §14 / §43）。"]
    txt = "\n".join(out) + "\n"
    with open("SSD_MIA.md", "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
