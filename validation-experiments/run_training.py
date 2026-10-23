"""Run an unchanged training configuration with optional conditional RNG repeats.

--validation-rng-seed resets training/noise RNGs after loading round-1 weights.
The population seed, partition and warm checkpoint remain fixed. Such a run is
a conditional repeat, not an additional independent client-population seed.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import train_sdd
from sdd_core import Client, Server


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--validation-rng-seed", type=int)
parser.add_argument("--validation-transcript", type=Path, required=True)
validation, training_args = parser.parse_known_args()
original_load = train_sdd.load_round1_checkpoint


def load_and_seed(*args, **kwargs):
    result = original_load(*args, **kwargs)
    if validation.validation_rng_seed is not None:
        seed = validation.validation_rng_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"[Validation] Conditional post-checkpoint RNG seed: {seed}")
    return result


original_aggregate = Server.aggregate_reports
transcript = []


def record_aggregate(self, *args, **kwargs):
    result = original_aggregate(self, *args, **kwargs)
    if self.pseudo_labels is not None:
        transcript.append({
            "round": len(transcript) + 1,
            "cluster_labels": self.pseudo_labels.argmax(dim=1).tolist(),
            "probabilities": self.pseudo_labels.tolist(),
        })
        validation.validation_transcript.write_text(
            json.dumps(transcript, indent=2) + "\n")
    return result


train_sdd.load_round1_checkpoint = load_and_seed
Server.aggregate_reports = record_aggregate
original_train = Client.new_train
training_completions = 0
training_started = time.monotonic()


def train_with_progress(self, *args, **kwargs):
    global training_completions
    result = original_train(self, *args, **kwargs)
    training_completions += 1
    if training_completions % 10 == 0:
        elapsed = time.monotonic() - training_started
        print(f"[Validation progress] trained {training_completions}/200 client-rounds; "
              f"elapsed={elapsed:.1f}s", flush=True)
    return result


Client.new_train = train_with_progress
sys.argv = [str(ROOT / "train_sdd.py"), *training_args]
if __name__ == "__main__":
    train_sdd.main()
