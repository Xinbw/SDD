"""Integrate completed experiments into a separate paper, compile and validate it."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "paper-experiment-validation/sdd-www2027"
PREVIOUS = ROOT / "sdd-www2027-paper-3fig-compiled-8pages-reviewed.zip"
STEM = "sdd-www2027-paper-3fig-compiled-8pages-validated"


def replace_once(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path.name, old[:60], text.count(old))
    path.write_text(text.replace(old, new, 1))


def main():
    r = json.loads((HERE / "analysis_results.json").read_text())
    assert r["all_new_training_jobs_complete"], "Do not integrate incomplete training runs."
    assert r["cohort_validation"]["1000"]["seeds"] == [42, 43, 44]
    assert r["cohort_validation"]["3000"]["seeds"] == [42, 43, 44]
    plan = json.loads((HERE / "plan.json").read_text())
    for name, expected in plan["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert not DEST.exists(), "A separate revision already exists; inspect it before rerunning."
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PREVIOUS) as z:
        z.extractall(DEST.parent)
    snapshots = DEST / "validation-experiments"
    snapshots.mkdir()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".py", ".log", ".txt", ".pdf", ".png", ".npz"}:
            shutil.copyfile(path, snapshots / path.name)
    inputs = snapshots / "inputs"
    inputs.mkdir()
    for path in [ROOT / "logs/dp_v2/round1_clean_reports_K10.npy",
                 ROOT / "data/cifar10/cluster_result/10_C_2000.npy",
                 ROOT / "data/cifar10/proxy_cifar10_src_idx_2000.npy",
                 ROOT / "data/cifar10/proxy_cifar10_orig_labels_2000.npy"]:
        shutil.copyfile(path, inputs / path.name)
    for seed in [42, 43, 44]:
        for eps in ["0_25", "0_5"]:
            path = ROOT / f"logs/dp_v2/k_eps_grid/ours_K10_eps{eps}_s{seed}.log"
            shutil.copyfile(path, snapshots / path.name)
    for cohort in [100, 300, 1000, 3000]:
        path = ROOT / f"logs/dp_v2/ablation_current/cohort{cohort}_eps1_0.log"
        shutil.copyfile(path, snapshots / f"original_{path.name}")
    # Include the exact original inputs for every seed in the revised curve.
    for row in r["cohort_validation"].values():
        for record in row["logs"]:
            path = ROOT / record["log"]
            if record["log"].startswith("logs/"):
                target = snapshots / "original-logs" / record["log"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    for path in (HERE / "work").rglob("*"):
        if path.is_file() and path.suffix in {".txt", ".npy"}:
            target = snapshots / "per-client-metrics" / path.relative_to(HERE / "work")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    paper = DEST / "paper"
    margin = r["fixed_report_replay"]["margin_summary"]
    floor = int(margin["vote_operating_cohort_rounded"])
    numbers = (paper / "numbers.tex").read_text()
    numbers = re.sub(r"(\\newcommand\{\\MarginKTen\}\{)[^}]+", lambda m: m[1]+f"{margin['vote_median']:.4f}", numbers)
    numbers = re.sub(r"(\\newcommand\{\\CohortFloor\}\{)[^}]+", lambda m: m[1]+str(floor), numbers)
    (paper / "numbers.tex").write_text(numbers)
    exporter = DEST / "tables-and-scripts/export_tex_tables.py"
    text = exporter.read_text()
    begin, end = text.index("def _margin_k10():"), text.index("\ndef macros():")
    text = text[:begin] + '''def _margin_k10():
    """Median margin of the one-hot vote aggregate; standard even-K median."""
    snapshot = Path(__file__).resolve().parents[1] / "validation-experiments/fixed_report_replay.json"
    return json.loads(snapshot.read_text())["margin_summary"]["vote_median"]

''' + text[end:]
    exporter.write_text(text)
    plotter = DEST / "figure-review/plot_candidates.py"
    replace_once(plotter,
        "    acc = np.array([.267, .361, .420, .411, .428])",
        '''    import csv, json
    validation = OUT.parent / "validation-experiments"
    with (validation / "cohort_curve.csv").open() as stream:
        cohort_rows = list(csv.DictReader(stream))
    acc = np.array([float(row["mean_accuracy"]) for row in cohort_rows])
    std = np.array([float(row["std_across_seeds"]) if row["std_across_seeds"] else np.nan for row in cohort_rows])
    cohort_counts = np.array([int(row["n_seeds"]) for row in cohort_rows])
    operating_n = json.loads((validation / "fixed_report_replay.json").read_text())["margin_summary"]["vote_operating_cohort_rounded"]''')
    replace_once(plotter,
        '    ax.axvline(940, color="#555555", ls="--", lw=1, label=r"margin estimate: $n\\approx940$")',
        '''    multi_seed = cohort_counts > 1
    ax.errorbar(n[multi_seed], acc[multi_seed], yerr=std[multi_seed], fmt="none",
                color=COLORS["SDD"], capsize=3, lw=.9)
    ax.axvline(operating_n, color="#555555", ls="--", lw=1,
               label=f"vote-margin estimate: n≈{operating_n:.0f}")''')
    replace_once(plotter,
        '    ax.legend(frameon=False, loc="lower right", fontsize=7); ax.grid(axis="y", color="#dddddd", lw=.6)',
        '    ax.legend(frameon=True, facecolor="white", edgecolor="white", framealpha=.97,\n'
        '              loc="lower right", fontsize=7); ax.grid(axis="y", color="#dddddd", lw=.6)')
    eval_path = paper / "sec-eval.tex"
    replace_once(eval_path, r"n_{\mathrm{op}}\approx940", r"n_{\mathrm{op}}\approx\CohortFloor{}")
    replace_once(eval_path,
        "The cohort sweep uses seed $42$ and a\n  fixed $100$-client estimate of consensus.",
        "Cohort points at $n\\ge1000$ use three seeds with standard-deviation bars;\n  $n=100,300$ use seed $42$. Consensus uses $100$ trained clients.")
    curve = r["cohort_validation"]
    means = [curve[str(n)]["mean"] for n in [100, 300, 1000, 3000, 10000]]
    replacement = (
        "reaches accuracies " + ", ".join(f"${m:.3f}$" for m in means[:-1]) + f" and ${means[-1]:.3f}$ at\n"
        "$n=100$, $300$, $1000$, $3000$ and $10000$.\n"
        "The points at $n\\ge1000$ use three matched seeds; the two smaller cohorts\n"
        "use seed $42$. Table~\\ref{tab:cohort-validation} gives exact values.\n"
        "The utility rise near $10^3$ matches the vote-margin operating scale.\n"
        "This gives a practical planning rule: measure\n")
    replace_once(eval_path,
        "reaches accuracies $0.267$, $0.361$, $0.420$, $0.411$ and $0.428$ at\n"
        "$n=100$, $300$, $1000$, $3000$ and $10000$.\n"
        "The steep improvement up to $10^3$ and the subsequent band around $0.42$\n"
        "match the predicted transition. This gives a practical planning rule: measure\n", replacement)
    replace_once(eval_path,
        "The corresponding single-seed utility points are $0.420$ and $0.428$.\n",
        f"The corresponding three-seed utility means are ${means[2]:.3f}$ and ${means[4]:.3f}$.\n")
    replace_once(eval_path,
        "$0.038$ noise per coordinate, comparable to the measured median margin\n",
        "$0.038$ noise per coordinate, below the median vote margin\n")
    appendix = paper / "sec-appendix.tex"
    replace_once(appendix, r"n_{\mathrm{op}}\approx940", r"n_{\mathrm{op}}\approx\CohortFloor{}")
    # Name the release encoding for the margin used by the planning equation.
    threat = paper / "sec-threat-model.tex"
    replace_once(threat, "Using the median round-1 margin", "Using the median round-1 vote margin")
    tables = [r"% generated by integrate_paper.py from analysis_results.json",
              r"\begin{table}[ht]",
              r"  \caption{Paired validation of the cohort operating scale. Three seeds characterize the $1k$, $3k$ and $10k$ points; smaller cohorts locate the initial rise.}",
              r"  \label{tab:cohort-validation}", r"  \small",
              r"  \begin{tabular}{@{}rrrr@{}}", r"    \toprule",
              r"    cohort $n$ & residual $\sigma/n$ & accuracy & seeds \\", r"    \midrule"]
    sigma = margin["sigma"]
    for n in [100, 300, 1000, 3000, 10000]:
        row = curve[str(n)]
        value = f"{row['mean']:.4f}"
        if row["std"] is not None:
            value += rf"\,$\pm$\,{row['std']:.4f}"
        tables.append(f"    {n} & {sigma/n:.4f} & {value} & {row['n']} " + r"\\")
    tables += [r"    \bottomrule", r"  \end{tabular}",
               r"  \vspace{2pt}",
               r"  \parbox{\linewidth}{\footnotesize CIFAR-10, $\alpha=0.1$, $K=10$, $T=3$, $\varepsilon=1$, $t=0$. Consensus uses 100 trained clients; $n$ controls residual-noise calibration. Three-seed rows use 42/43/44 and report mean and standard deviation; other rows use seed 42.}",
               r"\end{table}", ""]
    (paper / "tab-cohort-validation.tex").write_text("\n".join(tables))
    conditional = r["conditional_epsilon_repeat"]
    appendix_text = rf'''
\section{{Validation of Nearby Release Operating Points}}
\label{{app:operating-validation}}

\input{{tab-cohort-validation}}

The matched cohort experiments hold the partition, round-1 checkpoint and
learning schedule fixed within each seed. The vote aggregate has median
top-two margin $\MarginKTen{{}}$ on the seed-$42$ reports, using the mean of
the two middle values for even $K$. With $\sigma=\SigmaKTen{{}}$, this gives
$n_{{\mathrm{{op}}}}\approx\CohortFloor{{}}$ in Eq.~\eqref{{eq:scale}}.
Across the three matched seeds, mean client accuracy rises from
$0.4229$ at $n=1000$ to $0.4416$ at $3000$, a $1.87$-point gain.

A separate $20000$-repeat numerical-noise experiment uses those fixed reports
and the implementation's rounded continuous-Gaussian shares on the 16-bit grid.
Changing calibration from $n=1000$ to $3000$ raises mean pseudo-label accuracy
from $0.5992$ to $0.6715$ and reduces the fraction of clusters whose labels
change from $0.2961$ to $0.1206$. These measurements characterize the release
before subsequent client learning.

The original $\varepsilon=0.25$ and $0.5$ CIFAR-10 points each score
$0.4416\pm0.0329$ over seeds $42/43/44$. A further conditional repeat holds
the seed-$42$ population and round-1 models fixed, with post-checkpoint RNG
seed $10042$ shared across the two budgets. It scores
${conditional['0.25']['accuracy']:.4f}$ at $0.25$ and
${conditional['0.5']['accuracy']:.4f}$ at $0.5$. The post-processed probabilities
differ between budgets, while the complete cluster-label vectors agree
in ${sum(r['conditional_cluster_labels_match_every_round'])}$ of the three rounds.
This repeat separately probes
noise and subsequent training randomness; the main curves retain their
three independent population seeds. Hard targets map continuous noisy
aggregates to discrete cluster decisions, so nearby release scales can retain
the same targets or follow different learning trajectories.
'''
    with appendix.open("a") as stream:
        stream.write(appendix_text)
    # Render only the changed mechanism figure; other figures remain frozen.
    subprocess.run([sys.executable, "-c",
                    "import plot_candidates as p; p.style(); p.mechanism()"],
                   cwd=DEST / "figure-review", check=True)
    compile_log = paper / "build-output.txt"
    with compile_log.open("w") as stream:
        subprocess.run(["/home/xbw/.local/bin/tectonic", "--only-cached", "--keep-logs",
                        "--keep-intermediates", "main.tex"], cwd=paper, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    log = (paper / "main.log").read_text()
    assert not re.search(r"Overfull \\hbox|Overfull \\vbox|(?:Citation|Reference) .+ undefined|! LaTeX Error", log)
    doc = fitz.open(paper / "main.pdf")
    refs = [i+1 for i, p in enumerate(doc) if "REFERENCES" in p.get_text()]
    assert refs[0] == 9 and len(doc) <= 12, (refs, len(doc))
    assert "knowledge clients need to share." in " ".join(doc[7].get_text().split())
    preview = DEST / "validation-preview"
    preview.mkdir()
    for i in [5, 7, len(doc)-1]:
        doc[i].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(preview / f"page-{i+1}.png")
    previous_validation = json.loads((DEST / "validation.json").read_text())
    previous_validation.update(pdf_pages=len(doc), main_body_pages=8, references_begin_page=9,
                               experiment_validation_complete=True,
                               new_full_training_runs=6, fixed_report_noise_repeats=20000,
                               corrected_vote_margin=margin,
                               cohort_3000_minus_1000=r["cohort_3000_minus_1000"],
                               figure_files_unchanged_from_8page_version={
                                   "fig-main-curves.pdf": True, "fig-ablations.pdf": True,
                                   "fig-mechanism.pdf": False})
    previous_validation.pop("number_macros_unchanged", None)
    previous_validation["number_macros_changed"] = ["MarginKTen", "CohortFloor"]
    previous_validation["visual_review"] = "Mechanism figure and final page previews generated; final visual inspection pending."
    (DEST / "validation.json").write_text(json.dumps(previous_validation, indent=2) + "\n")
    readme = DEST / "README.md"
    text = readme.read_text().replace("PDF 全文 11 页", f"PDF 全文 {len(doc)} 页")
    text = text.replace("8 页正文·审阅意见落实版", "8 页正文·异常点实验验证版")
    text = text.replace("数值来自原包的冻结实验，叙事、图示重点和表格说明已更新。",
                        "主比较数值来自原包的冻结实验；cohort 曲线补充配对训练，实验验证见下文。")
    text += "\n## 异常点实验验证\n\n新增 6 组训练验证及 20,000 次固定报告噪声重复实验。训练验证复用对应 seed 的首轮模型检查点，补跑其余两轮；整体 T=3、每轮 20 epochs、100 个训练客户端。Fig.2 在 1k、3k、10k 使用三 seed 均值和标准差；100、300 保留单 seed 测点。vote 编码的 margin 与规划参考线由固定实际报告重新计算。完整分析、日志与脚本见 `validation-experiments/`，准确口径见 `VALIDATION_REPORT.txt`。\n"
    text += "\n归档包含新训练日志、各轮每客户端指标、原 cohort 曲线的逐 seed 日志，以及固定报告噪声实验的输入数组和重复样本。完整训练脚本在原实验仓库中运行，需要原始数据、模型代码及 round-1 checkpoints；本论文归档未包含训练数据与模型权重。\n"
    readme.write_text(text)
    with (DEST / "REVISION_NOTES.txt").open("a") as stream:
        stream.write("\n八、异常点配对实验验证\n新增 6 组训练验证，固定 100 客户端、K=10、T=3、每轮 20 epoch，复用对应 seed 的首轮检查点并完成其余两轮。\n1k/3k 各补 seed 43/44；ε=0.25/0.5 另作固定首轮模型的后续 RNG 条件重复。\nFig.2 使用源数据生成的三 seed 均值与标准差，并新增精确 cohort 验证表。\n20,000 次固定报告噪声重复测量发布层伪标签，不并入训练次数。\n参考线采用 vote 报告的标准中位 margin，替换平均概率报告的旧参考口径。\nFig.1、Fig.3、主比较冻结结果及所有旧交付件保留。\n")
    shutil.copyfile(paper / "main.pdf", ROOT / (STEM + ".pdf"))
    archive = ROOT / (STEM + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(DEST.rglob("*")):
            if not path.is_file() or any(x in path.parts for x in ["validation-preview", "__pycache__"]):
                continue
            if path.name == "build-output.txt" or (path.parent == paper and path.suffix in {".aux", ".log", ".out", ".blg", ".bbl"}):
                continue
            z.write(path, "sdd-www2027/" + str(path.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read("sdd-www2027/paper/main.pdf") == (ROOT / (STEM + ".pdf")).read_bytes()
    print(json.dumps({"body_pages": 8, "total_pages": len(doc), "pdf": str(ROOT/(STEM+".pdf")),
                      "source_archive": str(archive), "visual_inspection": "pending"}, indent=2))


if __name__ == "__main__":
    main()
