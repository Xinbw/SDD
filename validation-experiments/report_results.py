"""Write a concise, source-derived validation report with an explicit run status."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    r = json.loads((HERE / "analysis_results.json").read_text())
    complete = r["all_new_training_jobs_complete"]
    done = sum(x["complete"] for x in r["new_end_to_end_jobs"])
    lines = ["Fig.1 / Fig.2 异常点复核", "",
             f"完整训练补跑：{done}/6 完成。" + ("验证完成。" if complete else "其余仍在运行，训练结论待完整配对结果。"),
             "各组复用对应 seed 的首轮模型检查点，补跑其余两轮；整体 T=3，每轮 20 epochs，100 个训练客户端。",
             "", "1. Fig.1(a)：原图两个预算点的实际结果", ""]
    for eps in ["0.25", "0.5"]:
        x = r["existing_fig1a_plateau"][eps]
        lines.append(f"ε={eps}：{100*x['mean']:.2f}% ± {100*x['std']:.2f} pp；seeds 42/43/44：{x['values']}。")
    lines += ["两点逐 seed 结果相同，属于平台；原始日志的各轮轨迹也相同。", "",
              "2. Fig.2(a)：1k 与 3k 的完整训练配对验证", ""]
    for cohort in ["1000", "3000"]:
        x = r["cohort_validation"][cohort]
        value = f"{100*x['mean']:.2f}%"
        if x["std"] is not None:
            value += f" ± {100*x['std']:.2f} pp"
        lines.append(f"n={cohort}：{value}；已完成 seeds={x['seeds']}，逐 seed={x['values']}。")
    if "cohort_3000_minus_1000" in r:
        x = r["cohort_3000_minus_1000"]
        lines += [f"3k−1k 的配对差值：{x['mean']:+.2f} ± {x['std']:.2f} pp；95% CI=[{x['mean_ci95'][0]:.2f}, {x['mean_ci95'][1]:.2f}]；未调整配对 p={x['unadjusted_p']:.4f}。",
                  "这些值报告实际测得的差异及不确定性，不据非显著 p 值推断等效。"]
    old_a = r["cohort_validation"]["1000"]["logs"][0]["round_pseudo_label_accuracy"]
    old_b = r["cohort_validation"]["3000"]["logs"][0]["round_pseudo_label_accuracy"]
    lines += [f"原 seed 42 各轮伪标签准确率：1k={old_a}，3k={old_b}。",
              "3k 的首轮伪标签更准确，后续轮次的共识轨迹发生变化；发布层噪声与最终训练准确率需分别核验。"]
    lines += ["", "3. 固定首轮报告的独立数值噪声重复实验", "",
              "20,000 次配对噪声抽样，使用生产代码的 float64 高斯取整、逐客户端噪声份额及 16-bit 网格。",
              "这里测量的是固定报告的伪标签准确率，不是 20,000 次完整训练。"]
    mc = r["fixed_report_replay"]
    for name in ["cohort1000", "cohort3000", "eps0_25", "eps0_5"]:
        x = mc["conditions"][name]
        lines.append(f"{name}：平均伪标签准确率 {100*x['pseudo_label_accuracy']['mean']:.2f}%；簇标签翻转率 {100*x['cluster_flip_fraction']['mean']:.2f}%。")
    gain = mc["paired_comparisons"]["cohort3000 minus cohort1000"]["pseudo_label_gain_points"]
    lines += [f"3k−1k：伪标签平均提升 {gain['mean']:.2f} pp；Monte-Carlo 95% CI [{gain['mean_ci95'][0]:.2f}, {gain['mean_ci95'][1]:.2f}]。",
              "噪声减小改善该固定报告的平均稳定性；最终准确率还受硬目标选择和后续训练轨迹影响。"]
    if "margin_summary" in mc:
        m = mc["margin_summary"]
        lines += [f"曲线使用 vote 编码：其标准中位 margin={m['vote_median']:.4f}，对应 sigma/m≈{m['vote_operating_cohort_rounded']:.0f}。",
                  "原 940 参考线来自 mean-posterior 报告的下中位数；新的参考线按 vote 报告与标准中位数计算。"]
    lines += ["",
              "4. ε=0.25 / 0.5 的新条件重复", "",
              "固定 seed 42 的客户端划分与首轮模型，后续训练/噪声 RNG 设置为 10042，两个预算共用此设置。",
              "这是条件重复，单独陈述，不并入三个独立客户端种子的主均值。"]
    for eps, record in r["conditional_epsilon_repeat"].items():
        if record["complete"]:
            lines.append(f"ε={eps}：最终准确率 {100*record['accuracy']:.2f}%；各轮伪标签准确率={record['round_pseudo_label_accuracy']}。")
    if complete:
        lines += [f"新配对中 ε=0.5−0.25：{r['conditional_epsilon_gain_points']:+.2f} pp。",
                  f"各轮完整簇标签一致：{r['conditional_cluster_labels_match_every_round']}。",
                  "各轮后处理概率向量的最大坐标差：" +
                  ", ".join(f"{x:.6f}" for x in r["conditional_max_postprocessed_probability_change_by_round"]) +
                  "；概率输出发生变化，硬目标保持一致。"]
    lines += ["", "5. 其他局部下降的配置核查", ""]
    for x in r["existing_adjacent_decreases"]:
        if x["method"] == "SDD":
            lines.append(f"{x['setting']}，ε {x['left_epsilon']}→{x['right_epsilon']}：{x['paired']['mean']:+.2f} pp；未调整配对 p={x['paired']['unadjusted_p']:.4f}；SDD 的 K={x['K_used_by_SDD']}。")
    lines += ["MNIST α=1.0 的 0.25→0.5 同时改变 K（2000→500），不应将差异单独归因于 ε。",
              "隐私预算和参与规模控制发布噪声；它们不构成对有限种子下最终训练准确率逐点单调的约束。", "",
              "复现与出处：plan.json、原始训练日志、各轮指标、analysis_results.json、fixed_report_replay.json。",
              "本次新日志单独保存，原论文冻结数据与此前交付件保留。"]
    (HERE / "VALIDATION_REPORT.txt").write_text("\n".join(lines) + "\n")
    print(f"Validation report updated: {done}/6 full training jobs complete.")


if __name__ == "__main__":
    main()
