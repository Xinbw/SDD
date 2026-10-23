"""
PATE baseline (Confident-GNMax), aligned run-for-run with train_sdd.py.

Same scaffolding as train_fedct.py on purpose -- identical data partition, same
round-1 warm start, same proxy set, same evaluation -- so the aggregation
mechanism is the only thing that differs.

Why not plain GNMax: under RDP it is algebraically identical to
train_fedct.py --dp-mode central (see pate_accounting.py), so it would just
reproduce that curve.  The new information is the Confident screen.

Accounting is RDP (pate_accounting.py), which is PATE's own analysis.  Note
that the SSD and DP-FedCT numbers we compare against were produced with zCDP,
which is 1.2-1.45x looser in sigma here -- so this setup FAVOURS PATE.  Stated
so that a low-budget PATE loss reads as conservative rather than rigged.
"""
_ORIG_DOC = """
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
from pate_accounting import calibrate as pate_calibrate, summary as pate_summary
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
    p.add_argument("--pate-answer-budget", type=int,   default=1000,
                   help="每轮回答的查询数上限 M，必须事先固定（数据无关），默认 N/2。")
    p.add_argument("--pate-threshold",     type=float, default=0.25,
                   help="Confident 筛选阈值，按票数占比计。默认 0.25 取自 "
                        "diagnose_pate_consensus.py 实测的干净投票分布（100 个 alpha=0.1 "
                        "教师：plurality 份额均值 0.2253、中位 0.2200、最大 0.3400），"
                        "落在中位与最大值之间。注意 0.5 这类常见取值在本场景干净投票永远够不到，"
                        "筛过的全是被噪声推上去的样本。阈值取自干净投票统计而非测试集精度，"
                        "但真实部署中获知它要花预算 —— 偏袒 PATE，论文需注明。")
    p.add_argument("--pate-no-screen",     action="store_true",
                   help="退化成 plain GNMax：不筛选、回答全部查询。用于验证它与 "
                        "train_fedct.py --dp-mode central 在同 eps 下数值一致。")
    p.add_argument("--pate-sigma-ratio",   type=float, default=0.5,
                   help="sigma1/sigma2。未调参，见 pate_accounting.calibrate 的说明。")
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

    # ── 4b. PATE (Confident-GNMax) configuration ─────────────────────────────
    n_queries = args.proxy_size
    M = n_queries if args.pate_no_screen else min(args.pate_answer_budget, n_queries)
    cohort = args.dp_cohort_size if args.dp_cohort_size is not None else len(all_client_ids)
    if args.pate_no_screen:
        # sigma1 -> inf makes the screening term vanish, leaving plain GNMax.
        from pate_accounting import epsilon_of
        import math as _m
        _lo, _hi = 1e-3, 1.0
        while epsilon_of(1e12, _hi, n_queries, M, args.rounds, args.delta, cohort) > args.epsilon_total:
            _hi *= 2.0
        for _ in range(200):
            _mid = _m.sqrt(_lo * _hi)
            if epsilon_of(1e12, _mid, n_queries, M, args.rounds, args.delta, cohort) > args.epsilon_total:
                _lo = _mid
            else:
                _hi = _mid
        sigma1, sigma2 = float("inf"), _hi
    else:
        sigma1, sigma2 = pate_calibrate(
            epsilon=args.epsilon_total, delta=args.delta, n_queries=n_queries,
            n_answered=M, rounds=args.rounds, cohort=cohort,
            sigma_ratio=args.pate_sigma_ratio,
        )
    print(pate_summary(args.epsilon_total, args.delta, sigma1, sigma2,
                       n_queries, M, args.rounds, cohort))
    # Noise seen by the argmax is on the COHORT-AVERAGED histogram, matching how
    # DPConfig.noise_std_on_average reports it for SSD and DP-FedCT.
    screen_std = (float("inf") if sigma1 == float("inf") else sigma1 / cohort)
    answer_std = sigma2 / cohort
    print(f"[PATE] noise on averaged histogram: screen={screen_std:.6f} "
          f"answer={answer_std:.6f}  (threshold={args.pate_threshold} of the vote share)")
    rng_pate = np.random.default_rng(args.seed + 4242)

    rng_select = np.random.default_rng(args.seed + 1)
    num_selected = max(1, int(args.q * len(all_client_ids)))

    pseudo_labels: Optional[np.ndarray] = None
    train_idx = np.array([], dtype=np.int64)
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

                if pseudo_labels is not None and len(train_idx):
                    # Only the ANSWERED queries carry a label; unanswered ones were
                    # never paid for and must not be used. This is the utility side
                    # of Confident-GNMax: fewer labels, but cleaner ones.
                    sel_indices = [proxy_indices[j] for j in train_idx]
                    proxy_labeled = LabeledSubset(proxy_base_train, sel_indices,
                                                  pseudo_labels[train_idx])
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

        # Confident-GNMax
        # Vote shares in [0,1]: each of the n reporting clients contributes 1/n.
        # Working in shares (not raw counts) is what makes the noise scale
        # comparable to SSD's noise-on-the-average.
        n_part = len(all_client_ids)
        votes = np.zeros((n_queries, num_classes), dtype=np.float64)
        rows = np.arange(n_queries)
        for i in range(n_part):
            votes[rows, all_preds[i]] += 1.0
        votes /= n_part

        # (a) screening: is there a strong enough plurality to be worth paying for?
        if args.pate_no_screen:
            passed = np.arange(n_queries)
        else:
            noisy_max = votes.max(axis=1) + rng_pate.normal(0.0, screen_std, size=n_queries)
            passed = np.flatnonzero(noisy_max > args.pate_threshold)
        # (b) pre-committed answer budget: first M in index order. The rule must
        # not depend on the data, or the count itself leaks (pate_accounting.py).
        answered = passed[:M]

        # (c) answering: GNMax on the survivors only
        pseudo_labels = np.full(n_queries, -1, dtype=np.int64)
        if answered.size:
            noisy = votes[answered] + rng_pate.normal(
                0.0, answer_std, size=(answered.size, num_classes))
            pseudo_labels[answered] = noisy.argmax(axis=1)
        train_idx = answered
        print(f"[Round {r}] screened_pass={passed.size}/{n_queries}  "
              f"answered={answered.size} (budget {M})")

        # Accuracy is measured ON THE ANSWERED SUBSET only; comparing it against
        # SSD's figure (which covers all 2000) needs the coverage column too.
        if answered.size:
            label_acc = (pseudo_labels[answered] == proxy_true_labels[answered]).mean()
        else:
            label_acc = float("nan")
        print(f"[Round {r}] proxy_label_accuracy={label_acc:.4f}  "
              f"coverage={answered.size}/{n_queries}")

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
    summary_path = os.path.join(out_dir, "experiment_summary_pate.txt")

    eps_str = (f"PATE Confident-GNMax eps_total={args.epsilon_total} delta={args.delta} "
               f"cohort={cohort} M={M} thr={args.pate_threshold} acct=rdp")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_accs = [evaluate(models_dict[cid], test_loader, device) for cid in all_client_ids]
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[PATE] {ts}\n")
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

    print(f"\n==== PATE Experiment Summary ====")
    print(f"Final mean accuracy: {mean(final_accs):.4f}  std: {pstdev(final_accs):.4f}")
    print(f"Round accuracies: {round_accs}")
    print(f"[Summary] Written to {summary_path}")


if __name__ == "__main__":
    main()
