"""Analyse the fixed paired plan; retain population and conditional repeats separately."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scipy import stats
import export_comparison as source


def parse_log(path):
    text = path.read_text(errors="replace")
    final = re.search(r"^Final mean client accuracy: ([\d.]+)$", text, re.M)
    return {
        "log": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "complete": final is not None,
        "accuracy": float(final[1]) if final else None,
        "round_accuracy": [float(x) for x in re.findall(r"\[Round \d+\] valid_acc=([\d.]+)", text)],
        "round_pseudo_label_accuracy": [float(x) for x in re.findall(r"\[Proxy labels\] label_accuracy=([\d.]+)", text)],
    }


def population_statistics(values):
    mean = statistics.mean(values)
    result = {"n": len(values), "mean": mean, "values": values,
              "std": statistics.stdev(values) if len(values) > 1 else None}
    if len(values) > 1:
        half = stats.t.ppf(.975, len(values) - 1) * result["std"] / len(values) ** .5
        result["mean_ci95"] = [float(mean - half), float(mean + half)]
    return result


def paired_statistics(left, right):
    differences = [100 * (y - x) for x, y in zip(left, right)]
    result = population_statistics(differences)
    result["units"] = "percentage points, right minus left"
    # Identical paired observations do not define a t-statistic.
    result["unadjusted_p"] = float(stats.ttest_rel(right, left).pvalue) if any(differences) else None
    return result


def first_log(patterns):
    import glob
    for pattern in patterns:
        for match in sorted(glob.glob(str(ROOT / pattern))):
            if source.read(match) is not None:
                return Path(match)
    raise FileNotFoundError(patterns)


def main():
    result = {"existing_adjacent_decreases": [], "existing_fig1a_plateau": {},
              "cohort_validation": {}, "conditional_epsilon_repeat": {},
              "statistical_scope": "Paired seed tests are descriptive and unadjusted; a non-significant test is not an equivalence result."}
    for name, kmap, functions, _ in source.COMBOS:
        for method, function in zip(["SDD", "DP-FedCT", "PATE"], functions):
            for left, right in [("0.025", "0.05"), ("0.25", "0.5"), ("0.5", "1.0")]:
                a = source.series(left, kmap[left], function)
                b = source.series(right, kmap[right], function)
                if len(a) == len(b) == 3 and statistics.mean(b) < statistics.mean(a):
                    result["existing_adjacent_decreases"].append({
                        "setting": name, "method": method, "left_epsilon": float(left),
                        "right_epsilon": float(right),
                        "K_used_by_SDD": [kmap[left], kmap[right]],
                        "left": population_statistics(a), "right": population_statistics(b),
                        "paired": paired_statistics(a, b),
                    })
    plateau = result["existing_fig1a_plateau"]
    for eps in ["0.25", "0.5"]:
        records = [parse_log(first_log(source.cifar_ssd(eps, 10, seed))) for seed in [42, 43, 44]]
        plateau[eps] = dict(population_statistics([r["accuracy"] for r in records]), logs=records)
    plateau["paired"] = paired_statistics(plateau["0.25"]["values"], plateau["0.5"]["values"])
    result["new_end_to_end_jobs"] = []
    for seed in [43, 44]:
        for cohort in [1000, 3000]:
            path = HERE / f"cohort{cohort}_s{seed}.log"
            if path.exists():
                result["new_end_to_end_jobs"].append(dict(parse_log(path), population_seed=seed, cohort=cohort))
    for eps in [.25, .5]:
        name = f"eps{str(eps).replace('.', '_')}_s42_rng10042"
        path = HERE / (name + ".log")
        if path.exists():
            record = dict(parse_log(path), population_seed=42, post_checkpoint_rng_seed=10042, epsilon=eps)
            transcript = HERE / (name + ".transcript.json")
            if transcript.exists():
                record["transcript"] = json.loads(transcript.read_text())
            result["conditional_epsilon_repeat"][str(eps)] = record
            result["new_end_to_end_jobs"].append(record)
    for cohort in [100, 300, 1000, 3000, 10000]:
        old_path = (ROOT / f"logs/dp_v2/ablation_current/cohort{cohort}_eps1_0.log"
                    if cohort != 10000 else first_log(source.cifar_ssd("1.0", 10, 42)))
        records = [dict(parse_log(old_path), population_seed=42)]
        for seed in [43, 44]:
            path = (first_log(source.cifar_ssd("1.0", 10, seed)) if cohort == 10000
                    else HERE / f"cohort{cohort}_s{seed}.log")
            if path.exists():
                record = dict(parse_log(path), population_seed=seed)
                if record["complete"]:
                    records.append(record)
        result["cohort_validation"][str(cohort)] = dict(
            population_statistics([r["accuracy"] for r in records]),
            seeds=[r["population_seed"] for r in records], logs=records)
    a, b = result["cohort_validation"]["1000"], result["cohort_validation"]["3000"]
    if a["seeds"] == b["seeds"] == [42, 43, 44]:
        result["cohort_3000_minus_1000"] = paired_statistics(a["values"], b["values"])
    result["all_new_training_jobs_complete"] = (len(result["new_end_to_end_jobs"]) == 6
                                               and all(r["complete"] for r in result["new_end_to_end_jobs"]))
    if result["all_new_training_jobs_complete"]:
        a, b = result["conditional_epsilon_repeat"]["0.25"], result["conditional_epsilon_repeat"]["0.5"]
        result["conditional_epsilon_gain_points"] = 100 * (b["accuracy"] - a["accuracy"])
        result["conditional_cluster_labels_match_every_round"] = [
            x["cluster_labels"] == y["cluster_labels"] for x, y in zip(a["transcript"], b["transcript"])]
        result["conditional_max_postprocessed_probability_change_by_round"] = [
            max(abs(u - v) for xr, yr in zip(x["probabilities"], y["probabilities"])
                for u, v in zip(xr, yr))
            for x, y in zip(a["transcript"], b["transcript"])]
    result["fixed_report_replay"] = json.loads((HERE / "fixed_report_replay.json").read_text())
    (HERE / "analysis_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (HERE / "cohort_curve.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cohort", "mean_accuracy", "std_across_seeds", "n_seeds", "seed_values"])
        for cohort, record in result["cohort_validation"].items():
            writer.writerow([cohort, record["mean"], record["std"], record["n"], json.dumps(record["values"])])
    print(json.dumps({"all_new_training_jobs_complete": result["all_new_training_jobs_complete"],
                      "cohort_validation": {n: {k: v for k, v in r.items() if k != "logs"}
                                            for n, r in result["cohort_validation"].items()},
                      "cohort_3000_minus_1000": result.get("cohort_3000_minus_1000")}, indent=2))


if __name__ == "__main__":
    main()
