# SDD — 8 页正文·review.md 编辑修订版

本版在 `sdd-www2027-paper-3fig-compiled-8pages-second-review.zip` 上，落实当前 `review.md` 中已认可及调整后认可的意见。正文 8 页，结论完整位于第 8 页，参考文献从第 9 页开始，全文 12 页。原始包与此前交付件均保留。

论文的核心记忆点是：**把语义粒度作为隐私设计变量，让紧凑发布支持小预算知识迁移。**

## 本轮修改

统一 Table 3 与 Appendix C 的效用评价术语，保留 100 个训练客户端和队列残余噪声标定的准确区分；精简 PATE-DI 会计协议说明；§5.3 主动突出 ε=0.1 下 SDD 42.3% 对 DP-FedAvg 10.3% 的准确率，把高预算细节留在完整图表；Table 4 脚注明确写 4950 个客户端对。

两张生成表的说明同步写入生成器或 metadata，重建后数值行一致。全部实验数值、数字宏、三张图、参考文献与既有实验材料保持原样。没有新增训练。

逐条取舍见 `REVIEW_MD_DECISIONS.txt`，输入意见快照见 `REVIEW_MD_INPUT.md`。没有删除正确的贡献列表、跨栏续接和坐标轴刻度，也没有引入未经检验的预算比或因果归因。

## 已落实的主要改进

- 解释 `1 ≤ K ≤ |P|` 的发布粒度谱系：单样本簇在接口层面恢复逐样本投票发布，实验覆盖 K=5–2000。
- 主动呈现低预算优势、共同效用目标的 5–10 倍预算差异，以及强异质性 MNIST、ε=1 相对 PATE-DI 的 20.28 个百分点增益。
- 统一评估实现名 PATE-DI；数据无关 Rényi-DP 会计协议集中定义于 §5.1，原始 PATE 文献仍使用 PATE 名称。
- Table 4 从原始审计 JSON 补全两个设置的全部七档预算，共 14 行；数值由生成器导出。
- 将三 seed cohort 点与小 cohort 测点分别用于验证操作规模、定位初始上升沿。Appendix E 聚焦发布稳定性，完整重复条件和日志留在 artifact。
- 提前强调 ε=0.1 时 10% 串谋容忍度对应的 5.4% 噪声增量与三 seed 相同准确率；结论补上基于 vote margin 的队列规划工具。

28 个主比较设置的准确率、标准差和显著性标记保持一致。Fig.1 使用此前统一的 PATE-DI 图例；三张图和此前实验验证材料保留。

## 主要结果与评价口径

强异质性 CIFAR-10、ε=0.1：对比 DP-FedCT／PATE-DI，SDD 提升 14.2／14.9 个百分点。强异质性 MNIST、ε=0.05：提升 45.1／42.7 个百分点。四项所选比较均有三 seed 配对 p<0.01 支撑。

共同目标预算差异来自七档已测预算：MNIST 达到 90% 平均准确率时，SDD 对 DP-FedCT 的预算比为 1:5（α=0.1）或 1:10（α=1.0）。主曲线用 100 个实际训练客户端估计共识，按部署 cohort n=10⁴ 标定残余噪声；主结果为 t=0。逐格配对检验未作多重比较调整。

审计评估 seed-42 round-1 实际报告、4950 个替换客户端对与三次独立固定报告发布。已测 ε≤0.25 内，最大高斯检验 TPR 为 1.15%，FPR 为 1%，随机基线为 1%。这是有明确对象与范围的检验结果。定理中的离散 Gaussian 机制与数值实现中取整的连续 Gaussian 采样在论文中各自说明。

## 文件与重建

- `paper/main.pdf`：编译后的论文；`paper/main.tex`：LaTeX 入口。
- `paper/figures/`：三张矢量图；`figure-review/plot_candidates.py`：从冻结结果重建图形。
- `tables-and-scripts/`：表格生成器与说明元数据；`export_budget_targets.py` 可在本包内重建共同目标预算表。
- `audit-inputs/`：本次完整七档审计表用到的六份原始 JSON 快照。
- `validation-experiments/`：此前完成的 6 组训练验证、20,000 次固定报告噪声重复、日志、输入数组与分析；本次编辑没有新增训练。
- `results/`、`provenance/`：原包结果与技术出处，保留原样。
- `REVISION_NOTES.txt`、`REVIEW_MD_DECISIONS.txt`：本轮修改与意见取舍；`FIRST_REVIEW_NOTES.txt`、`SECOND_REVIEW_NOTES.txt`、`REVISION_NOTES_HISTORY.txt`：此前修订历史。
- `validation.json`：页数、数值、数据出处和排版核验记录。

```bash
cd paper
latexmk -pdf main.tex
# 或
tectonic main.tex
```

图形生成器可在本包内运行。完整表格导出依赖原实验仓库的 `logs/dp_v2/`、`dp_utils.py` 和相关模块；在原实验仓库运行本包生成器，并将 `OUT` 指向本包 `paper/`。审计导出明确选择 seed 42、全部七档预算；原日志路径不存在时会读取本包 `audit-inputs/`。生成的表格和数字宏不手改数值。

此前六组验证复用各 seed 的 round-1 检查点，完成其余两轮，整体 T=3、每轮 20 epochs、100 个训练客户端。Fig.2 的 1k、3k、10k 为三 seed 均值及标准差；100、300 为 seed-42 测点。ε=0.25／0.5 的重复固定训练人口及首轮模型，属于条件验证，不并入三人口 seed 的主结果。完整协议见 `validation-experiments/VALIDATION_REPORT.txt`；训练数据与模型权重存于原实验仓库。
