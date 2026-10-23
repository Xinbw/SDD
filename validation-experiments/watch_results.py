"""Maintain reports during the queue and compile the separate paper after completion."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(script):
    subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT, check=True)


def main():
    previous_count = -1
    last_heartbeat = 0
    while True:
        state = json.loads((HERE / "status.json").read_text())
        if state.get("error"):
            raise RuntimeError(state["error"])
        count = len(state["jobs"])
        if count != previous_count:
            run("analyse_results.py")
            run("plot_validation.py")
            run("report_results.py")
            previous_count = count
        active = state.get("active_job")
        if active is None and state.get("finished_utc"):
            run("integrate_paper.py")
            print("VALIDATION_ARTIFACTS_READY_FOR_VISUAL_REVIEW", flush=True)
            break
        if time.monotonic() - last_heartbeat >= 50:
            log = HERE / f"{active}.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            progress = re.findall(r"\[Validation progress\].+", text)
            rounds = re.findall(r"\[Round \d+\] valid_acc=.+", text)
            detail = progress[-1] if progress else (rounds[-1] if rounds else "initialising")
            print(json.dumps({"full_training_completed": f"{count}/6", "active_job": active,
                              "latest_progress": detail}, ensure_ascii=False), flush=True)
            last_heartbeat = time.monotonic()
        time.sleep(5)


if __name__ == "__main__":
    main()
