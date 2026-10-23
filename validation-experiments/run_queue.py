"""Persist a fixed paired experiment plan and run it without replacing old logs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INITIAL_PID = 682876
COMMON = [
    "--dataset", "cifar10", "--clients", "100", "--alpha", "0.1",
    "--q", "1", "--rounds", "3", "--epochs", "20", "--batch-size", "256",
    "--samples-per-client", "6000", "--arch-dist", "0", "0", "1",
    "--lr", "0.1", "--val-ratio", "0.0", "--proxy_size", "2000",
    "--proxy-src-idx", str(ROOT / "data/cifar10/proxy_cifar10_src_idx_2000.npy"),
    "--proxy-split", "test", "--clusters", "10", "--cluster-cache",
    str(ROOT / "data/cifar10/cluster_result/10_C_2000.npy"),
    "--no-final-ckpt", "--report-encoding", "vote", "--confidence-threshold", "0.0",
]
JOBS = [
    {"id": f"cohort{n}_s{s}", "population_seed": s, "cohort": n,
     "epsilon": 1.0, "post_checkpoint_rng_seed": None}
    for s in [43, 44] for n in [1000, 3000]
] + [
    {"id": f"eps{str(e).replace('.', '_')}_s42_rng10042", "population_seed": 42,
     "cohort": 10000, "epsilon": e, "post_checkpoint_rng_seed": 10042}
    for e in [.25, .5]
]


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def completed(path):
    return path.exists() and re.search(r"^Final mean client accuracy: ([\d.]+)$",
                                      path.read_text(errors="replace"), re.M)


def checkpoint(seed):
    name = "run1" if seed == 42 else f"run1_seed{seed}"
    return ROOT / f"ckpt/cifar/0.1/{name}"


def write_state(state):
    target = HERE / "status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(target)


def main():
    plan = {
        "created_utc": timestamp(), "scope": "paired validation of Fig.1(a) and Fig.2(a)",
        "independent_population_seeds": [42, 43, 44],
        "conditional_repeat_rng_seed": 10042,
        "jobs": JOBS,
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ["train_sdd.py", "sdd_core.py", "data.py", "dp_utils.py"]},
        "notes": ["All runs retain 100 clients, K=10, T=3, 20 epochs, original clean test split.",
                  "Existing seed-42 cohort runs and three-seed epsilon results remain separate source evidence.",
                  "The additional epsilon pair varies post-checkpoint RNG, not the client population."],
    }
    path = HERE / "plan.json"
    if not path.exists():
        path.write_text(json.dumps(plan, indent=2) + "\n")
    state = {"started_utc": timestamp(), "jobs": []}
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    for job in JOBS:
        log = HERE / (job["id"] + ".log")
        if job["id"] == "cohort1000_s43" and not completed(log):
            while Path(f"/proc/{INITIAL_PID}").exists():
                state["active_job"] = job["id"]
                state["active_pid"] = INITIAL_PID
                write_state(state)
                time.sleep(10)
            if not completed(log):
                raise RuntimeError("Initial training ended without final metrics; inspect its log.")
        if completed(log):
            record = dict(job, state="complete", accuracy=float(completed(log)[1]))
            state["jobs"].append(record)
            write_state(state)
            continue
        work = HERE / "work" / job["id"]
        work.mkdir(parents=True, exist_ok=True)
        if not (work / "data").exists():
            (work / "data").symlink_to(ROOT / "data", target_is_directory=True)
        command = [sys.executable, "-u", str(HERE / "run_training.py"),
                   "--validation-transcript", str(HERE / (job["id"] + ".transcript.json")),
                   *COMMON, "--seed", str(job["population_seed"]),
                   "--round1-ckpt", str(checkpoint(job["population_seed"])),
                   "--dp-cohort-size", str(job["cohort"]),
                   "--epsilon-total", str(job["epsilon"])]
        if job["post_checkpoint_rng_seed"] is not None:
            command += ["--validation-rng-seed", str(job["post_checkpoint_rng_seed"])]
        started = timestamp()
        print(f"{started} START {job['id']}", flush=True)
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT)
            state.update(active_job=job["id"], active_pid=process.pid,
                         active_started_utc=started, active_command=command)
            write_state(state)
            code = process.wait()
        if code or not completed(log):
            state["error"] = dict(job=job["id"], returncode=code, log=str(log))
            write_state(state)
            raise RuntimeError(f"Experiment failed: {job['id']}, exit={code}")
        record = dict(job, state="complete", accuracy=float(completed(log)[1]),
                      started_utc=started, finished_utc=timestamp(), command=command)
        state["jobs"].append(record)
        write_state(state)
        print(f"{timestamp()} DONE {job['id']} accuracy={record['accuracy']}", flush=True)
    state.update(active_job=None, active_pid=None, finished_utc=timestamp())
    write_state(state)
    print("ALL_PAIRED_VALIDATIONS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
