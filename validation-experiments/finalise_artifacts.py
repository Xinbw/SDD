"""Check source evidence, record visual inspection, and package the final revision."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import zipfile

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "paper-experiment-validation/sdd-www2027"
PAPER = DEST / "paper"
STEM = "sdd-www2027-paper-3fig-compiled-8pages-validated"


def main():
    r = json.loads((HERE / "analysis_results.json").read_text())
    state = json.loads((HERE / "status.json").read_text())
    plan = json.loads((HERE / "plan.json").read_text())
    assert r["all_new_training_jobs_complete"]
    assert len(state["jobs"]) == 6 and state["active_job"] is None
    assert state["finished_utc"] and not state.get("error")
    for name, digest in plan["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for job in state["jobs"]:
        text = (HERE / f"{job['id']}.log").read_text()
        assert "[Eval] Dropped 1033/5000 global test images" in text
        assert float(re.search(r"Final mean client accuracy: ([\d.]+)", text)[1]) == job["accuracy"]
        summary = (HERE / "work" / job["id"] /
                   "mid_data/cifar10/acc_of_test/0.1/experiment_summary.txt").read_text()
        assert "Evaluated clients: 100/100" in summary
        rounds = ast.literal_eval(re.search(r"Recorded round accuracies: (\[.+\])", summary)[1])
        assert len(rounds) == 3 and abs(rounds[-1] - job["accuracy"]) < .000051
    for row in r["cohort_validation"].values():
        values = []
        for record in row["logs"]:
            p = ROOT / record["log"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"]
            values.append(float(re.search(r"Final mean client accuracy: ([\d.]+)", p.read_text())[1]))
        assert row["values"] == values and row["mean"] == statistics.mean(values)
        if len(values) > 1:
            assert row["std"] == statistics.stdev(values)
    with (HERE / "cohort_curve.csv").open() as stream:
        for row in csv.DictReader(stream):
            expected = r["cohort_validation"][row["cohort"]]
            assert float(row["mean_accuracy"]) == expected["mean"]
            assert int(row["n_seeds"]) == expected["n"]
    assert r["conditional_cluster_labels_match_every_round"] == [True, True, True]
    assert all(x > 0 for x in r["conditional_max_postprocessed_probability_change_by_round"])
    mc = r["fixed_report_replay"]
    for name, digest in mc["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert mc["trials"] == 20000 and mc["margin_summary"]["vote_operating_cohort_rounded"] == 690
    with np.load(HERE / "fixed_report_replay_samples.npz") as samples:
        assert all(len(samples[key]) == 20000 for key in samples.files)
    frozen = 0
    with zipfile.ZipFile(ROOT / "sdd-www2027-paper-3fig-compiled-8pages-reviewed.zip") as previous:
        names = [x for x in previous.namelist() if not x.endswith("/")]
        protected = {"paper/main.tex", "paper/refs.bib", "paper/sec-intro.tex",
                     "paper/sec-method.tex", "paper/sec-related.tex", "paper/sec-conclusion.tex",
                     "paper/tab-main-a01.tex", "paper/tab-main-a10.tex", "paper/tab-fedavg.tex",
                     "paper/tab-ablation.tex", "paper/tab-mia.tex", "paper/tab-budget-targets.tex",
                     "paper/figures/fig-main-curves.pdf", "paper/figures/fig-ablations.pdf"}
        for name in names:
            relative = name.removeprefix("sdd-www2027/")
            if relative in protected or relative.startswith(("results/", "provenance/")):
                assert previous.read(name) == (DEST / relative).read_bytes(), relative
                frozen += 1
        regex = r"\\newcommand\{\\(\w+)\}\{([^}]+)\}"
        old_macros = dict(re.findall(regex, previous.read("sdd-www2027/paper/numbers.tex").decode()))
        new_macros = dict(re.findall(regex, (PAPER / "numbers.tex").read_text()))
        assert old_macros.keys() == new_macros.keys()
        assert {k for k in old_macros if old_macros[k] != new_macros[k]} == {"MarginKTen", "CohortFloor"}
    log = (PAPER / "main.log").read_text()
    assert not re.search(r"Overfull \\hbox|Overfull \\vbox|(?:Citation|Reference) .+ undefined|! LaTeX Error", log)
    doc = fitz.open(PAPER / "main.pdf")
    assert len(doc) == 12
    assert [i + 1 for i, page in enumerate(doc) if "REFERENCES" in page.get_text()] == [9]
    assert "knowledge clients need to share." in " ".join(doc[7].get_text().split())
    appendix = doc[11].get_text()
    assert "0.4229" in appendix and "0.4416" in appendix and "0.4290" in appendix
    assert "1.87-point gain" in " ".join(appendix.split())
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    # Refresh derived evidence after the additional probability-vector check.
    snapshots = DEST / "validation-experiments"
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".py", ".log", ".txt", ".pdf", ".png", ".npz"}:
            shutil.copyfile(path, snapshots / path.name)
    validation = json.loads((DEST / "validation.json").read_text())
    validation.update(
        visual_review="Passed: revised Fig.2, main pages 6/7/8 and appendix page 12 visually inspected; labels, error bars, captions and conclusion fit.",
        source_hashes_verified=plan["source_hashes"],
        original_protected_files_verified=frozen,
        proxy_test_exclusion_verified_for_new_runs=True,
        new_training_protocol="Six warm-start validation runs: reuse the corresponding seed's round-1 models, complete rounds 2/3 with 20 epochs each; overall T=3 and 100 trained clients.",
        conditional_cluster_labels_match_every_round=r["conditional_cluster_labels_match_every_round"],
        conditional_probability_differences=r["conditional_max_postprocessed_probability_change_by_round"],
        pdf_pages=12, main_body_pages=8, references_begin_page=9)
    (DEST / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    readme = DEST / "README.md"
    text = readme.read_text()
    text += "\n最终核验：6 组训练验证均完成；新增各组测试集已排除 1033 张代理图像。Fig.2 与第 8／12 页已完成视觉检查，正文 8 页、参考文献始于第 9 页、全文 12 页。两个 ε 条件重复的后处理概率不同，三轮硬标签相同，准确率均为 42.90%。\n"
    readme.write_text(text)
    shutil.copyfile(PAPER / "main.pdf", ROOT / (STEM + ".pdf"))
    archive = ROOT / (STEM + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(DEST.rglob("*")):
            if not path.is_file() or any(x in path.parts for x in ["validation-preview", "__pycache__"]):
                continue
            if path.name == "build-output.txt" or (path.parent == PAPER and path.suffix in {".aux", ".log", ".out", ".blg", ".bbl"}):
                continue
            z.write(path, "sdd-www2027/" + str(path.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read("sdd-www2027/paper/main.pdf") == (ROOT / (STEM + ".pdf")).read_bytes()
        assert json.loads(z.read("sdd-www2027/validation.json"))["main_body_pages"] == 8
        assert json.loads(z.read("sdd-www2027/validation-experiments/analysis_results.json"))["all_new_training_jobs_complete"]
    print(json.dumps({"complete": True, "training_runs": 6, "noise_repeats": 20000,
                      "main_pages": 8, "total_pages": 12, "frozen_files_verified": frozen,
                      "pdf_bytes": (ROOT / (STEM + ".pdf")).stat().st_size,
                      "zip_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
