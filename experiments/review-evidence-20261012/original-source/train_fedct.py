"""
FedCT CIFAR-10 experiment aligned with SSD/FedPKDA settings.

Algorithm:
  Each communication round:
    1. Selected clients train for E epochs on local data +
       pseudo-labeled proxy data (from round ≥ 2).
    2. ALL clients predict hard labels on the proxy set.
    3. Optional: Randomized Response DP on per-client predictions.
    4. Majority vote → new pseudo labels for next round.

Alignment with SSD/FedPKDA:
  - Dirichlet(alpha) partition via SSD's data.py
  - ResNet-18 (strong arch) via SSD's models.py
  - Test set: last 5000 CIFAR-10 test samples MINUS every proxy image
    (the proxy is distilled on every round; leaving it in the test set would
    score memorisation — same exclusion train_sdd.py applies)
  - Same proxy IMAGES: indices are applied to the TEST split, exactly as
    prepare_clustered_proxy does. (An earlier version applied the same index
    file to the train split, which silently selected 2000 different images.)
  - SGD(lr=0.1, momentum=0.9, wd=5e-4)
  - Same round-1 models: --warm-start-load accepts train_sdd.py's own
    --round1-ckpt directory (round 1 is plain local training in both methods),
    so the two pipelines are identical until the first pseudo-label release.

Differences from SSD:
  - SSD: soft-label KL-distillation on clustered proxy
  - FedCT: hard-label majority vote on proxy

DP modes (--dp-mode)
  none     no privacy; plain majority vote.
  rr       FedCT's native Randomized Response, per-label epsilon. NOT comparable
           to SSD's epsilon-total: under client-level adjacency the joint release
           is proxy_size labels x rounds, so the honest client-level epsilon is
           eps * proxy_size * rounds (printed at startup).
  central  the fair port of FedCT into SSD's trust model: each client one-hot
           encodes its hard labels (N x C), adds its 1/n share of discrete
           Gaussian noise, and the server sees only the SUM (argmax per row ==
           majority vote when noise-free). Same accountant, same epsilon-total /
           delta / cohort semantics as train_sdd.py. The only remaining
           difference from SSD is WHAT is released: per-sample one-hots with
           L2 sensitivity sqrt(2N) vs cluster averages with sqrt(2K) — i.e. this
           baseline isolates the value of the semantic-cluster release.
"""

import argparse
import datetime
import os
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets

from data import MemDataset, create_client_data, get_dataset_info, load_datasets
from dp_utils import (
    ACCOUNTING_MODES,
    DPConfig,
    add_distributed_discrete_gaussian,
    dequantize,
)
from models import build_model


# ── Proxy dataset wrapper ─────────────────────────────────────────────────────

class LabeledSubset(Dataset):
    """Wraps a subset and overrides its labels with pseudo_labels."""

    def __init__(self, base: Dataset, indices: List[int], labels: np.ndarray):
        self.base = base
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        image, _ = self.base[self.indices[i]]
        return image, int(self.labels[i])


# ── DP: Randomized Response ───────────────────────────────────────────────────

def randomized_response(preds: np.ndarray, num_classes: int, epsilon: float) -> np.ndarray:
    """K-class Randomized Response.

    Retain original label with prob e^ε / (e^ε + K−1);
    flip uniformly to another class otherwise.
    """
    p_keep = np.exp(epsilon) / (np.exp(epsilon) + num_classes - 1)
    noisy = preds.copy()
    flip = np.random.random(len(preds)) > p_keep
    for i in np.where(flip)[0]:
        others = [c for c in range(num_classes) if c != preds[i]]
        noisy[i] = np.random.choice(others)
    return noisy


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def predict_hard(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            X = batch[0].to(device)
            parts.append(model(X).argmax(1).cpu().numpy())
    return np.concatenate(parts)


def majority_vote(all_preds: np.ndarray, num_classes: int) -> np.ndarray:
    """all_preds: [C, N] → consensus labels [N]."""
    return np.apply_along_axis(
        lambda col: np.bincount(col.astype(int), minlength=num_classes).argmax(),
        axis=0,
        arr=all_preds,
    )


# ── Round-1 warm start ────────────────────────────────────────────────────────

def load_warm_start(
    path: str,
    models_dict: Dict[int, nn.Module],
    optims_dict: Dict[int, torch.optim.Optimizer],
    device: torch.device,
) -> str:
    """
    Restore round-1 client models, skipping round-1 local training.

    Two formats are accepted:

    "ssd"    a directory (or the round1_models.pt inside it) written by
             sdd_core.save_round1_checkpoint: ``{client_id: state_dict}``,
             weights only.  This is the SHARED starting point with train_sdd.py:
             round 1 is plain local training in both methods, so loading SSD's
             own checkpoint makes the two pipelines byte-identical up to the
             first pseudo-label release and leaves the label mechanism as the
             only difference.  Optimizers stay fresh, exactly as train_sdd.py
             does when it resumes from --round1-ckpt.
    "fedct"  a .pt file written by --warm-start-save: ``{"models", "optims"}``.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "round1_models.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "models" in ckpt:
        for cid, model in models_dict.items():
            model.load_state_dict(ckpt["models"][cid])
            optims_dict[cid].load_state_dict(ckpt["optims"][cid])
        fmt = "fedct"
    else:
        missing = [cid for cid in models_dict if cid not in ckpt]
        if missing:
            raise SystemExit(
                f"Warm-start checkpoint {path} has no weights for clients {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} — partition mismatch."
            )
        for cid, model in models_dict.items():
            sd = ckpt[cid]
            if any(k.startswith("_orig_mod.") for k in sd):
                sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            model.load_state_dict(sd)
        fmt = "ssd (weights only, fresh optimizers)"

    print(f"[Round 1] Warm-start: loaded {len(models_dict)} models from {path} "
          f"[format: {fmt}]")
    return fmt


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FedCT CIFAR-10 (SSD-aligned)")
    p.add_argument("--dataset",            type=str,   default="cifar10")
    p.add_argument("--clients",            type=int,   default=100)
    p.add_argument("--alpha",              type=float, default=1.0)
    p.add_argument("--q",                  type=float, default=1.0,
                   help="Fraction of clients selected per round.")
    p.add_argument("--rounds",             type=int,   default=10)
    p.add_argument("--epochs",             type=int,   default=40)
    p.add_argument("--batch-size",         type=int,   default=256)
    p.add_argument("--samples-per-client", type=int,   default=6000)
    p.add_argument("--lr",                 type=float, default=0.1)
    p.add_argument("--momentum",           type=float, default=0.9)
    p.add_argument("--weight-decay",       type=float, default=5e-4)
    p.add_argument("--grad-clip",          type=float, default=1.0,
                   help="Max gradient norm for clipping (0 = disabled).")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--proxy-size",         type=int,   default=2000)
    p.add_argument("--proxy-src-idx",      type=str,   default=None,
                   help="Path to .npy file with proxy indices in --proxy-split.")
    p.add_argument("--proxy-split",        type=str,   default="test",
                   choices=["test", "train"],
                   help="Split the proxy indices refer to. 'test' matches "
                        "train_sdd.py (same images); 'train' reproduces the old "
                        "FedCT runs, which selected DIFFERENT images.")
    p.add_argument("--keep-proxy-in-eval", action="store_true", default=False,
                   help="Keep proxy images inside the global test set "
                        "(reproduces pre-fix numbers; scores memorisation).")
    p.add_argument("--dp-mode",            type=str,   default="none",
                   choices=["none", "rr", "central"],
                   help="See module docstring. 'central' is the SSD-comparable "
                        "mode; 'rr' is FedCT's native per-label RR.")
    p.add_argument("--epsilon",            type=float, default=None,
                   help="[rr] per-label Randomized Response epsilon.")
    p.add_argument("--epsilon-total",      type=float, default=1.0,
                   help="[central] (eps, delta)-DP budget for the ENTIRE run, "
                        "same semantics as train_sdd.py.")
    p.add_argument("--delta",              type=float, default=1e-5)
    p.add_argument("--dp-cohort-size",     type=int,   default=None,
                   help="[central] cohort n for the privacy/utility calculus; "
                        "same semantics as train_sdd.py.")
    p.add_argument("--dp-quantization-bits", type=int, default=16)
    p.add_argument("--dp-accounting",      type=str,   default="zcdp",
                   choices=sorted(ACCOUNTING_MODES))
    p.add_argument("--val-ratio",          type=float, default=0.0)
    p.add_argument("--root",               type=str,   default="./data")
    p.add_argument("--device",             type=int,   default=0)
    p.add_argument("--out-dir",            type=str,   default="./mid_data/cifar10/acc_of_test")
    p.add_argument("--warm-start-save",    type=str,   default=None,
                   help="Save model/optimizer states after Round-1 local training to this path.")
    p.add_argument("--warm-start-load",    type=str,   default=None,
                   help="Load Round-1 models from this path, skipping Round-1 local "
                        "training. Accepts a --warm-start-save .pt file OR "
                        "train_sdd.py's --round1-ckpt directory (SSD format, "
                        "weights only) for a shared starting point.")
    p.add_argument(
        "--arch-dist",
        type=float,
        nargs=3,
        metavar=("LIGHT", "STANDARD", "STRONG"),
        default=[0, 0, 1],
        help="Distribution over light/standard/strong architectures (default: all strong).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Same fail-fast guard as train_sdd.py (see _warn_or_die_on_cpu there): a silent
    # CPU fallback cost 23h of a queue on 2026-08-06. SDD_ALLOW_CPU=1 opts out.
    if not torch.cuda.is_available() and os.environ.get("SDD_ALLOW_CPU") != "1":
        raise SystemExit(
            "[Device] FATAL: torch.cuda.is_available() is False — refusing to train "
            "on CPU. Check `nvidia-smi` and /dev/nvidia*; a host driver upgrade under "
            "a live container makes cuInit fail in new processes (fix: restart the "
            "container). Set SDD_ALLOW_CPU=1 to override."
        )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    dataset_info = get_dataset_info(args.dataset)
    train_transform = dataset_info["train_transform"]
    test_transform = dataset_info["test_transform"]

    # ── 1. Proxy set (BEFORE client data: the eval split must exclude it) ─────
    # Two views of the source split: augmented (for training) and clean (for
    # prediction). For datasets with deterministic transforms we preload once.
    dataset_cls = dataset_info["class"]
    _preload = dataset_info.get("preload", False)
    _proxy_workers = 0 if _preload else 2
    _from_train = args.proxy_split == "train"
    if _preload:
        # train_transform == test_transform for MNIST-like datasets; one MemDataset suffices
        _raw_proxy = dataset_cls(root=args.root, train=_from_train, download=True,
                                 transform=train_transform)
        proxy_base_train = MemDataset(_raw_proxy)
        proxy_base_test = proxy_base_train
    else:
        proxy_base_train = dataset_cls(root=args.root, train=_from_train, download=True,
                                       transform=train_transform)
        proxy_base_test = dataset_cls(root=args.root, train=_from_train, download=True,
                                      transform=test_transform)

    if args.proxy_src_idx is not None:
        proxy_indices: List[int] = np.load(args.proxy_src_idx).astype(int).tolist()[: args.proxy_size]
    else:
        rng_proxy = np.random.default_rng(args.seed + 9999)
        tmp = list(range(len(proxy_base_test)))
        rng_proxy.shuffle(tmp)
        proxy_indices = tmp[: args.proxy_size]
    print(f"[Proxy] {len(proxy_indices)} images from the '{args.proxy_split}' split"
          + ("" if args.proxy_src_idx is None else f" via {args.proxy_src_idx}"))

    proxy_true_labels = np.array([proxy_base_test.targets[i] for i in proxy_indices])

    # ── 2. Client data (SSD Dirichlet partition) ──────────────────────────────
    # When the proxy comes from the test split, its images are removed from the
    # global evaluation set — clients train on them (pseudo-labeled) every round.
    exclude_eval = (
        proxy_indices
        if (args.proxy_split == "test" and not args.keep_proxy_in_eval)
        else None
    )
    client_data = create_client_data(
        dataset_name=args.dataset,
        num_clients=args.clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        root=args.root,
        seed=args.seed,
        samples_per_client=args.samples_per_client if args.samples_per_client > 0 else None,
        architecture_distribution=args.arch_dist,
        global_test_size=5000,
        exclude_test_indices=exclude_eval,
        val_ratio=args.val_ratio,
    )
    proxy_pred_loader = DataLoader(
        Subset(proxy_base_test, proxy_indices),
        batch_size=256, shuffle=False, num_workers=_proxy_workers, pin_memory=True,
    )

    # ── 3. Shared test loader ─────────────────────────────────────────────────
    test_loader = next(iter(client_data.values())).test_loader

    # ── 4. Initialise models & optimisers ─────────────────────────────────────
    all_client_ids: List[int] = list(client_data.keys())
    models_dict: Dict[int, nn.Module] = {}
    optims_dict: Dict[int, torch.optim.Optimizer] = {}
    criterion = nn.CrossEntropyLoss()

    for cid in all_client_ids:
        m = build_model(client_data[cid].architecture, args.dataset).to(device)
        opt = torch.optim.SGD(
            m.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        )
        models_dict[cid] = m
        optims_dict[cid] = opt

    # ── 4b. DP configuration ─────────────────────────────────────────────────
    dp_config: Optional[DPConfig] = None
    if args.dp_mode == "central":
        # One client's whole dataset changing can change its entire N x C one-hot
        # release; each row moves by at most sqrt(2), so Delta_2 = sqrt(2N).
        # DPConfig computes l2_sensitivity as sqrt(2 * num_clusters), so passing
        # the proxy size as "num_clusters" yields exactly that: the per-sample
        # release is the degenerate clustering where every sample is its own
        # cluster. Same accountant and cohort semantics as train_sdd.py.
        dp_config = DPConfig(
            epsilon=args.epsilon_total,
            delta=args.delta,
            q=args.q,
            num_clusters=args.proxy_size,
            num_rounds=args.rounds,
            cohort_size=args.dp_cohort_size,
            quantization_bits=args.dp_quantization_bits,
            accounting=args.dp_accounting,
        )
        print(dp_config.summary(len(all_client_ids)))
        print(f"[DP-central] L2 sensitivity sqrt(2N)={dp_config.l2_sensitivity:.2f} "
              f"vs SSD's cluster release sqrt(2K)={np.sqrt(2 * 20):.2f} — "
              f"{dp_config.l2_sensitivity / np.sqrt(2 * 20):.1f}x more noise at "
              f"the same epsilon.")
    elif args.dp_mode == "rr":
        if args.epsilon is None:
            raise SystemExit("--dp-mode rr requires --epsilon (per-label RR epsilon).")
        eps_client = args.epsilon * args.proxy_size * args.rounds
        print(f"[DP-RR] per-label eps={args.epsilon}. Under client-level adjacency "
              f"the joint release is {args.proxy_size} labels x {args.rounds} rounds, "
              f"so the honest client-level epsilon is <= {eps_client:g} "
              f"(basic composition). NOT comparable to SSD's epsilon-total.")
    elif args.epsilon is not None:
        raise SystemExit("--epsilon given but --dp-mode is 'none'; pass --dp-mode rr.")

    rng_select = np.random.default_rng(args.seed + 1)
    num_selected = max(1, int(args.q * len(all_client_ids)))

    pseudo_labels: Optional[np.ndarray] = None
    round_accs: List[float] = []

    # ── 5. FedCT training loop ────────────────────────────────────────────────
    for r in range(1, args.rounds + 1):
        # Client selection
        if args.q < 1.0:
            selected = rng_select.choice(all_client_ids, size=num_selected, replace=False).tolist()
        else:
            selected = all_client_ids

        # Local training — optionally skipped in Round 1 when loading a warm-start checkpoint
        loss_sum = 0.0
        if r == 1 and args.warm_start_load is not None:
            load_warm_start(args.warm_start_load, models_dict, optims_dict, device)
            loss_sum = float("nan")
        else:
            for cid in selected:
                cdata = client_data[cid]
                model = models_dict[cid]
                optimizer = optims_dict[cid]

                if pseudo_labels is not None:
                    # Combine local (augmented) + pseudo-labeled proxy (augmented)
                    proxy_labeled = LabeledSubset(proxy_base_train, proxy_indices, pseudo_labels)
                    loader = DataLoader(
                        ConcatDataset([cdata.train_loader.dataset, proxy_labeled]),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=_proxy_workers, pin_memory=True,
                    )
                else:
                    loader = cdata.train_loader

                epoch_loss = 0.0
                for _ in range(args.epochs):
                    epoch_loss = train_epoch(model, loader, optimizer, criterion, device,
                                             grad_clip=args.grad_clip)
                loss_sum += epoch_loss

        if r == 1 and args.warm_start_load is not None:
            print(f"[Round {r}] train_loss=n/a  (local training skipped: warm start)")
        elif np.isnan(loss_sum):
            print(f"[Round {r}] train_loss=nan  (WARNING: gradient explosion detected)")
        else:
            print(f"[Round {r}] train_loss={loss_sum / len(selected):.4f}")

        # Save warm-start checkpoint after Round-1 local training (before voting)
        if r == 1 and args.warm_start_save is not None:
            ckpt = {
                "models": {cid: models_dict[cid].state_dict() for cid in all_client_ids},
                "optims": {cid: optims_dict[cid].state_dict() for cid in all_client_ids},
            }
            torch.save(ckpt, args.warm_start_save)
            print(f"[Round {r}] Warm-start checkpoint saved → {args.warm_start_save}")

        # Communication: collect hard label predictions from ALL clients
        all_preds = np.stack(
            [predict_hard(models_dict[cid], proxy_pred_loader, device) for cid in all_client_ids],
            axis=0,
        )  # shape: [num_clients, proxy_size]

        # Privatised consensus
        if args.dp_mode == "central":
            # Each client one-hot encodes its predictions and adds its 1/n noise
            # share; the server sees only the sum (per-row argmax of the clean
            # sum IS the majority vote, so the mechanisms agree at sigma=0).
            n_part = len(all_client_ids)
            client_std = dp_config.client_noise_std(n_part)
            if r == 1:
                print(f"[DP] {n_part} reporting clients | per-client share "
                      f"std={client_std:.6f} | noise std on averaged histogram="
                      f"{dp_config.noise_std_on_average(n_part):.6f}")
            eye = np.eye(num_classes, dtype=np.float32)
            total = None
            for i in range(n_part):
                onehot = torch.from_numpy(eye[all_preds[i]])       # (N, C)
                contrib = add_distributed_discrete_gaussian(
                    onehot, client_std, dp_config.quantization_scale
                )
                total = contrib if total is None else total + contrib
            avg = dequantize(total, dp_config.quantization_scale,
                             dtype=torch.float32) / n_part
            pseudo_labels = avg.argmax(dim=1).numpy()
        else:
            if args.dp_mode == "rr":
                all_preds = np.stack([
                    randomized_response(all_preds[i], num_classes, args.epsilon)
                    for i in range(len(all_client_ids))
                ], axis=0)
            # Majority vote consensus
            pseudo_labels = majority_vote(all_preds, num_classes)
        label_acc = (pseudo_labels == proxy_true_labels).mean()
        print(f"[Round {r}] proxy_label_accuracy={label_acc:.4f}  "
              f"coverage={args.proxy_size}/{args.proxy_size}")

        # Evaluate all clients on shared test set
        accs = [evaluate(models_dict[cid], test_loader, device) for cid in all_client_ids]
        mean_acc = mean(accs)
        std_acc = pstdev(accs)
        round_accs.append(mean_acc)
        print(f"[Round {r}] valid_acc={mean_acc:.4f}  std={std_acc:.4f}  "
              f"best={max(accs):.4f}  worst={min(accs):.4f}")

    # ── 6. Save results ───────────────────────────────────────────────────────
    out_dir = os.path.join(args.out_dir, str(args.alpha))
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "experiment_summary_fedct.txt")

    if args.dp_mode == "central":
        cohort = args.dp_cohort_size if args.dp_cohort_size is not None else len(all_client_ids)
        eps_str = (f"central eps_total={args.epsilon_total} delta={args.delta} "
                   f"cohort={cohort} acct={args.dp_accounting}")
    elif args.dp_mode == "rr":
        eps_str = f"rr per-label eps={args.epsilon}"
    else:
        eps_str = "none"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_accs = [evaluate(models_dict[cid], test_loader, device) for cid in all_client_ids]
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[FedCT] {ts}\n")
        f.write(f"  alpha={args.alpha}  epsilon={eps_str}\n")
        f.write(f"  arch_dist={args.arch_dist}\n")
        f.write(f"  rounds={args.rounds}  epochs={args.epochs}  q={args.q}\n")
        f.write(f"  lr={args.lr}  batch={args.batch_size}\n")
        f.write(f"  clients={args.clients}  samples_per_client={args.samples_per_client}\n")
        f.write(f"  proxy_size={args.proxy_size}  seed={args.seed}\n")
        f.write(f"  mean_accuracy={mean(final_accs):.6f}\n")
        f.write(f"  std_accuracy={pstdev(final_accs):.6f}\n")
        f.write(f"  best_client={max(final_accs):.6f}  worst_client={min(final_accs):.6f}\n")
        f.write(f"  round_accuracies={[round(a, 6) for a in round_accs]}\n")

    print(f"\n==== FedCT Experiment Summary ====")
    print(f"Final mean accuracy: {mean(final_accs):.4f}  std: {pstdev(final_accs):.4f}")
    print(f"Round accuracies: {round_accs}")
    print(f"[Summary] Written to {summary_path}")


if __name__ == "__main__":
    main()
