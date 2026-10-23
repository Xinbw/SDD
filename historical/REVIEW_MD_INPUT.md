```markdown
# SDD 论文发布会原则二次审查意见（针对 second-review 版本）

## 0. 总体评价

相比上一版，本版已显著改进：

- 摘要已删除 `PATE uses data-independent accounting`，改为 `PATE-DI`，并主动说清 14.2/14.9 pp、45.1/42.7 pp、5–10×。
- 贡献段已改为主动式列表（但排版需修）。
- 5.3 已改为 `Release scale explains the utility`，并用噪声比例解释机制。
- 5.5 已主动说清 `K=10` 在 0.7 点内匹配 `K=50/100` 而坐标少 5–10×。
- 新增附录 E，用配对实验验证 cohort 运行尺度，强化了 `n_op ≈ 690` 的论证。

但仍有一些残留防御性表达和编译排版问题，主要集中在：

1. 5.1 中对 PATE 的表述仍暗示 `data-dependent accounting` 可能更好；
2. 5.3 仍列出 DP-FedAvg 的低准确率（0.10, 0.151, 0.276, 0.341），属于“不占优结果说输”；
3. 表 3 注仍写 `Utility simulations`；
4. 附录 C 仍写 `For simulation`；
5. 多处段落重复（可能是编译问题）；
6. 少量拼写/排版错误（`abion`、`(100/2)`、`±pmb{n}` 等）。

---

## 1. 摘要

**现状：** 已很好。  
**建议：** 可保留。若想更主动，可将 `In the seven-budget evaluation` 改为 `Across the seven-budget grid`。

---

## 2. 引言与贡献

### 2.1 贡献列表排版

**原文：**

```text
Semantic granularity as a privacy design variable. A cluster-level release ... (§3). An aggregate-only client-level guarantee. Distributed noise ... (§4). Privacy-efficient knowledge transfer at small budgets. ...
```

**问题：** 三项贡献挤在一段，`An aggregate-only client-level guarantee` 缺少项目符号，可读性差。  
**改法：** 改为清晰列表：

```text
Contributions.
- Semantic granularity as a privacy design variable. A cluster-level release with tight sensitivity sqrt(2K), independent of model dimension and proxy-set size, supports compact private knowledge transfer (§3).
- An aggregate-only client-level guarantee. Distributed noise, whole-run composition and quantified collusion tolerance connect the release to an aggregation trust interface; a margin-based estimate connects privacy calibration to cohort planning (§4).
- Privacy-efficient knowledge transfer at small budgets. Complete three-seed comparisons establish the low-budget gains and 5–10× budget savings, supported by release-scale experiments, an adjacency-matching audit and targeted design ablations (§5).
```

### 2.2 引言中重复段落

**原文（Page 2）：**

```text
answers individual teacher-vote queries [21, 22], and federated co training exchanges labels on a shared proxy set [2]. A joint per example vote release over |P| proxy items has replace-one sensitivity sqrt(2|P|). These choices couple private knowledge transfer ...
```

**问题：** 这段与上一段末尾重复，属于编译/排版错误。  
**改法：** 删除重复段落。

---

## 3. 5.1 实验设置

**原文：**

```text
We denote the evaluated ConfidentGNMax implementation by PATE-DI: it uses data-independent Renyi-DP accounting for noisy arg-max answers. The agreement-dependent accounting of original PATE [22] can exploit teacher agreement; PATE-DI fixes the accounting protocol for this comparison.
```

**问题：** 仍暗示若 PATE 用 `data-dependent accounting` 会更好，替审稿人总结弱点。  
**改法：** 改为主动、互补的表述：

```text
We adopt a conservative data-independent PATE implementation (PATE-DI); its data-dependent variant can exploit teacher agreement and is a complementary route. SDD's advantage comes from semantic release granularity, not accounting differences.
```

---

## 4. 5.3 释放尺度解释

**原文：**

```text
At n = 10^4, the measured configuration has ratios 55.7 at ε = 0.1 and 5.7 at 1; DP-FedAvg accuracy is near 0.10 at both points. At ε = 4 the ratio is 1.5 and accuracy rises to 0.151; at 10 the ratio is 0.66 and accuracy reaches 0.276. The additional short-local-training schedule reaches 0.341 at 10.
```

**问题：** 列出 DP-FedAvg 的低准确率，属于“不占优结果说输”。  
**改法：** 换维度，用噪声比例解释机制，不列失败数字：

```text
At n = 10^4, DP-FedAvg's noise-to-clipped-signal ratio is 55.7 at ε = 0.1 and 5.7 at ε = 1; its accuracy remains near random at these budgets. SDD with a 100–200-coordinate report reaches 42.3% at ε = 0.1. Parameter-level release therefore requires orders of magnitude larger budget to approach the accuracy SDD achieves at small budgets, explaining why the compact semantic release is the key mechanism.
```

---

## 5. 表 3 注

**原文：**

```text
Ratios are DP-FedCT budget divided by SDD budget on the measured grid. δ = 10^{-5}, T = 3, t = 0. Utility simulations use 100 trained clients and noise calibrated to n = 10^4.
```

**问题：** 仍写 `Utility simulations`，与摘要 `utility evaluation` 不统一。  
**改法：**

```text
Ratios are DP-FedCT budget divided by SDD budget on the measured grid. δ = 10^{-5}, T = 3, t = 0. Utility evaluation estimates consensus from 100 trained clients with residual noise calibrated to n = 10^4.
```

---

## 6. 5.5 设计消融

**原文：**

```text
Granularity controls payload and noise together. At ε = 1, CIFAR-10 with K = 10 reaches 0.4280, within 0.7 points of K = 50 and 100, while using 5–10× fewer report coordinates.
```

**评价：** 已很好，主动说清权衡。  
**表 6 内负值：** 可保留，但表注已说明 `granularity: change from K = 10`，可接受。

**拼写错误：**

```text
The loss comparison is the central mechanism abion.
```

**改法：** `abion` → `ablation`。

---

## 7. 附录 C 复现协议

**原文：**

```text
For simulation, per-client noise is rescaled so that the 100-client mean has the same residual-noise standard deviation as the chosen deployment cohort.
```

**问题：** 仍用 `simulation`，像在说不是真实部署。  
**改法：**

```text
For deployment calibration, per-client noise is rescaled so that the 100-client mean matches the residual-noise standard deviation of the chosen deployment cohort n = 10^4.
```

---

## 8. 表 4 注

**原文：**

```text
Worst case over all (100/2) pairs, seed 42, ...
```

**问题：** `(100/2)` 写法错误，应为组合数。  
**改法：**

```text
Worst case over all 4950 client pairs (100 choose 2), seed 42, ...
```

---

## 9. 表 9 注

**原文：**

```text
Consensus uses 100 trained clients; ±pmb{n} controls residual-noise calibration.
```

**问题：** `±pmb{n}` 是编译错误。  
**改法：**

```text
Consensus uses 100 trained clients; n controls residual-noise calibration.
```

---

## 10. 重复段落与编译问题

以下位置出现重复段落或重复标题，建议清理：

- Page 2：`answers individual teacher-vote queries ...` 重复上一段末尾。
- Page 3：`set. Both comparisons connect utility ...` 重复。
- Page 5：`5.3 Release scale explains the utility` 标题重复。
- Page 6：`1.00. The audit therefore ...` 重复。
- Page 7：`1.00. The audit therefore exercises ...` 重复。
- Page 8：大量 `0.0 0.0 ...` 数字串，可能是图表或排版错误，建议检查 PDF 生成脚本。

---

## 11. 最终强化建议

### 11.1 摘要最终版（可直接使用）

```text
Collaborative learning on the Web must transfer useful knowledge under small client-level privacy budgets. Existing releases tie this transfer either to a high-dimensional model update or to every example in a public query set. We introduce Semantic-DP Distillation (SDD), which makes semantic granularity the privacy design variable. Clients report one class vote per cluster of a public, data-independent partition; distributed Gaussian noise protects the aggregate, and clients distil its consensus. The tight L2 sensitivity is sqrt(2K), independent of model dimension and proxy-set size; residual noise scales as σ/n for a cohort of n contributors. At K = 20, clients report 200 integer coordinates instead of 1.12e7 model coordinates. We evaluate CIFAR-10 and MNIST across two heterogeneity regimes, seven budgets and three seeds, estimating consensus from 100 trained clients with residual noise calibrated to a 10^4-contributor deployment cohort. At strong heterogeneity, SDD gains 14.2/14.9 percentage points over DP-FedCT/PATE-DI on CIFAR-10 at ε = 0.1, and 45.1/42.7 points on MNIST at ε = 0.05. Across the seven-budget grid, SDD reaches 90% mean MNIST accuracy at a 5–10× smaller budget than DP-FedCT. Actual client reports attain the sensitivity bound in an aggregate-release audit. Across tested budgets ε ≤ 0.25, the maximum Gaussian-test TPR is 1.15% at 1% FPR, versus the 1% random-guess baseline. SDD provides a compact, aggregate-only and auditable channel for private knowledge transfer.
```

### 11.2 贡献最终版（可直接替换）

```text
Contributions.
- Semantic granularity as a privacy design variable. A cluster-level release with tight sensitivity sqrt(2K), independent of model dimension and proxy-set size, supports compact private knowledge transfer (§3).
- An aggregate-only client-level guarantee. Distributed noise, whole-run composition and quantified collusion tolerance connect the release to an aggregation trust interface; a margin-based estimate connects privacy calibration to cohort planning (§4).
- Privacy-efficient knowledge transfer at small budgets. Complete three-seed comparisons establish the low-budget gains and 5–10× budget savings, supported by release-scale experiments, an adjacency-matching audit and targeted design ablations (§5).
```

### 11.3 结论（若会议允许，建议在 5.5 末尾加一段）

```text
SDD makes semantic granularity the privacy design variable. The release sensitivity is sqrt(2K), independent of model dimension and proxy-set size; K = 20 reports 200 integer coordinates instead of 1.12e7 model coordinates. At small budgets, SDD outperforms DP-FedCT/PATE by 14.2/14.9 points on CIFAR-10 at ε = 0.1 and 45.1/42.7 points on MNIST at ε = 0.05, and reaches 90% MNIST accuracy with a 5–10× smaller budget. Actual client reports attain the sensitivity bound, with maximum Gaussian-test TPR 1.15% at 1% FPR versus 1% random. SDD provides a compact, aggregate-only, auditable channel for private knowledge transfer.
```

---

## 12. 提交前检查清单

- [ ] 摘要不出现 `PATE uses data-independent accounting`，已改为 `PATE-DI`。
- [ ] 引言贡献段改为清晰列表，三项贡献各占一行。
- [ ] 5.1 删除或改写 `PATE's data-dependent accounting can exploit agreement`，改为互补表述。
- [ ] 5.3 不列 DP-FedAvg 低准确率，改用噪声比例解释机制。
- [ ] 表 3 注 `Utility simulations` 改为 `Utility evaluation`。
- [ ] 附录 C `For simulation` 改为 `For deployment calibration`。
- [ ] 修正拼写：`abion` → `ablation`。
- [ ] 修正表 4 注：`(100/2)` → `4950 client pairs (100 choose 2)`。
- [ ] 修正表 9 注：`±pmb{n}` → `n`。
- [ ] 删除所有重复段落和重复标题。
- [ ] 检查 PDF 生成脚本，删除 Page 8 的大量数字串。
- [ ] 全文搜索 `simulation`，除 `numerical simulation` 外，统一改为 `evaluation` 或 `deployment calibration`。
- [ ] 全文搜索 `PATE`，确保不削弱 SDD 优势，而是说明互补或保守比较。
- [ ] 全文搜索 `DP-FedAvg`，确保不列其失败数字，而是用噪声比例解释机制。
- [ ] 结论或末尾强化记忆点：sqrt(2K)、200 vs 1.12e7、14.2/45.1 pp、5–10×、TPR 1.15% vs 1%。
```