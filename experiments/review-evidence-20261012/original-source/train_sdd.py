import argparse
import datetime
import os
import random
import sys
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from cluster import prepare_clustered_proxy
from data import ClientData, create_client_data, get_num_classes
from dp_utils import ACCOUNTING_MODES, DPConfig, log_dp_settings
from proxy_hetero import get_supported_proxy_datasets, prepare_heterogeneous_proxy
from sdd_core import (
    Client,
    ClientContext,
    SDDSystem,
    Server,
    load_round1_checkpoint,
    round1_checkpoint_filename,
    save_final_checkpoint,
    save_round1_checkpoint,
)


def _proxy_indices_in_eval_split(args) -> Optional[np.ndarray]:
    """Proxy indices that sit in the same split the global test set is cut from.

    Returned so create_client_data can drop them from the evaluation set: the
    clients distil on every proxy image each round, so any of them left in the
    test set turns reported accuracy into a memorisation score. Returns None when
    no overlap is possible — a proxy from another dataset, from the train split,
    or the default first-`proxy_size` selection, which lies below the held-out
    region entirely.
    """
    if getattr(args, "keep_proxy_in_eval", False):
        return None
    hetero = args.proxy_dataset is not None and args.proxy_dataset != args.dataset
    if hetero or args.proxy_split != "test" or not args.proxy_src_idx:
        return None
    if not os.path.exists(args.proxy_src_idx):
        return None
    return np.load(args.proxy_src_idx).reshape(-1)


def build_clients(
    dataset_name: str,
    client_data: Dict[int, ClientData],
    lr: float,
    optimizer: str,
    weight_decay: float,
    devices: List[torch.device],
) -> Dict[int, Client]:
    clients: Dict[int, Client] = {}
    if not devices:
        devices = [torch.device("cpu")]
    for idx, (cid, cdata) in enumerate(client_data.items()):
        device = devices[idx % len(devices)]
        ctx = ClientContext(
            client_id=cid,
            dataset_name=dataset_name,
            train_loader=cdata.train_loader,
            val_loader=cdata.val_loader,
            test_loader=cdata.test_loader,
            architecture=cdata.architecture,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            device=device,
        )
        clients[cid] = Client(ctx)
    return clients


def _resolve_devices(device_ids: Optional[List[int]]) -> List[torch.device]:
    if torch.cuda.is_available():
        total = torch.cuda.device_count()
        if device_ids:
            filtered = [idx for idx in device_ids if 0 <= idx < total]
            if not filtered:
                filtered = list(range(total))
        else:
            filtered = list(range(total))
        return [torch.device(f"cuda:{idx}") for idx in filtered]
    _warn_or_die_on_cpu()
    return [torch.device("cpu")]


def _warn_or_die_on_cpu() -> None:
    """
    Falling back to CPU is ~20x slower and otherwise silent: on 2026-08-06 a queue
    ran 23h on CPU before anyone noticed (host NVIDIA driver was upgraded under a
    live container, so cuInit failed in every new process while nvidia-smi still
    looked healthy). Long queues must fail fast instead.

    Set SDD_ALLOW_CPU=1 to opt out (CPU-only diagnostics, clustering, tests).
    """
    if os.environ.get("SDD_ALLOW_CPU") == "1":
        print("[Device] WARNING: no CUDA device; running on CPU (SDD_ALLOW_CPU=1).")
        return
    detail = ""
    try:  # surface the real driver error rather than just "not available"
        torch.cuda.init()
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        detail = f"\n  torch.cuda.init() -> {type(exc).__name__}: {exc}"
    raise SystemExit(
        "[Device] FATAL: torch.cuda.is_available() is False — refusing to train on "
        "CPU.\n  Check `nvidia-smi` and /dev/nvidia*; a host driver upgrade during a "
        "live container makes cuInit fail in new processes while nvidia-smi still "
        "works (fix: restart the container).\n  Re-run with SDD_ALLOW_CPU=1 if CPU "
        "is genuinely intended." + detail
    )


def compute_cluster_majority_labels(
    cluster_ids: torch.Tensor,
    true_labels: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """
    Utility for computing majority label per cluster. Retained for compatibility,
    though current flow leverages raw sample labels directly.
    """
    majority = torch.full((num_clusters,), -1, dtype=torch.long)
    for cid in range(num_clusters):
        mask = cluster_ids == cid
        if torch.any(mask):
            labels = true_labels[mask]
            values, counts = labels.unique(return_counts=True)
            majority[cid] = values[counts.argmax()]
    return majority


def _parse_client_sample_counts(spec: Optional[str]) -> Optional[Dict[int, int]]:
    """Parses CLI string like '0:200,1:150' into a dict."""
    if spec is None:
        return None
    mapping: Dict[int, int] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid format '{entry}'. Expected entries like client_id:count."
            )
        cid_str, count_str = entry.split(":", 1)
        mapping[int(cid_str)] = int(count_str)
    return mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic-DP Distillation (SDD)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "fashion-mnist", "cifar10"])    # 选择使用的数据集
    parser.add_argument("--clients", type=int, default=100)    # 客户端总数
    parser.add_argument("--alpha", type=float, default=0.1)     # Dirichlet 非 IID 划分的 α 参数
    parser.add_argument("--proxy_size", type=int, default=2000)    # 代理数据集大小（即将使用多少公共样本）
    parser.add_argument(
        "--proxy-dataset",
        type=str,
        default=None,
        help=(
            "异源代理数据集名称（留空则使用与 --dataset 相同的同源代理）。"
            f"支持: {get_supported_proxy_datasets()}。"
            "示例：cifar10 客户端 + cifar100 代理 → --proxy-dataset cifar100。"
        ),
    )
    parser.add_argument("--clusters", type=int, default=10)    # 聚类数量 K
    parser.add_argument("--q", type=float, default=1)    # 客户端采样率 q
    parser.add_argument("--rounds", type=int, default=2)    # 通信轮数 T
    parser.add_argument("--epochs", type=int, default=80)    # 每轮客户端本地训练的 epoch 数
    parser.add_argument("--late-epochs", type=int, default=None)    # 从 --late-epochs-from-round 起改用此 epoch 数
    parser.add_argument("--late-epochs-from-round", type=int, default=999)    # 切换 epoch 数的起始轮次
    parser.add_argument("--batch-size", type=int, default=256)    # 客户端本地训练 batch size；256 时 AMP 生效且 GPU 利用率最优
    parser.add_argument("--proxy-batch-size", type=int, default=256)    # 代理数据在客户端蒸馏时的 batch size
    parser.add_argument(
        "--epsilon-total",
        dest="epsilon_total",
        type=float,
        default=1.0,
        help=(
            "(ε, δ)-DP budget for the ENTIRE training run (all --rounds together), "
            "under the multi-message-shuffle aggregate view. Noise is the "
            "distributed discrete Gaussian with L2 sensitivity sqrt(2K); "
            "composition is zCDP across rounds."
        ),
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5,
        help="δ of the (ε, δ)-DP guarantee. Must be > 0 (Gaussian-type mechanism).",
    )
    parser.add_argument(
        "--dp-accounting",
        dest="dp_accounting",
        choices=sorted(ACCOUNTING_MODES),
        default="zcdp",
        help=(
            "How (ε, δ) is turned into noise. 'zcdp' is the closed-form "
            "composition every existing result was produced with. 'rdp' analyses "
            "the SAME mechanism with Rényi composition at the optimal order — "
            "about 20%% less noise at the same ε, no new assumption — and is the "
            "only mode that credits --q < 1 as Poisson subsampling amplification "
            "(which DOES assume secrecy of the sample)."
        ),
    )
    parser.add_argument(
        "--keep-proxy-in-eval",
        action="store_true",
        default=False,
        help=(
            "Keep proxy images inside the global test set. Off by default: the "
            "shipped index file overlaps the held-out region by 1033/2000, and "
            "clients distil on every proxy image, so leaving them in measures "
            "memorisation. Set this only to reproduce numbers from before the fix."
        ),
    )
    parser.add_argument(
        "--dp-cohort-size",
        type=int,
        default=None,
        help=(
            "Cohort size n used in the privacy/utility calculus. Defaults to the "
            "number of clients actually reporting each round. Set it larger to "
            "SIMULATE the utility of a bigger deployment: the injected noise then "
            "matches what an n-client cohort leaves after averaging, and the "
            "reported ε holds for that deployment. The aggregate is still estimated "
            "from the clients this run actually has — report it as a simulation."
        ),
    )
    parser.add_argument(
        "--dp-quantization-bits",
        type=int,
        default=16,
        help=(
            "Fixed-point resolution for the shuffle shares (grid 1/2^bits). "
            "Raise it if the discrete-Gaussian sigma in integer units is too small."
        ),
    )
    parser.add_argument(
        "--dp-collusion-t",
        type=int,
        default=0,
        help=(
            "Number of colluding clients the guarantee tolerates. Colluders "
            "subtract their own noise shares, so shares are sized for the n - t "
            "that survive; the cost is sqrt(n/(n-t)) more residual noise (10%% "
            "collusion = 5%% noise). Default 0 reproduces every existing result."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.1)    # 客户端学习率
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])    # 客户端优化器类型
    parser.add_argument("--distill-coeff", type=float, default=1.0)    # 蒸馏损失系数
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="服务端对聚合后软标签的置信度过滤阈值（max softmax）。"
             "低于该值的簇不用于知识蒸馏（0.0 表示不过滤）。",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="客户端本地数据中用于验证集的比例（0.0 表示全部用于训练）。")
    parser.add_argument("--seed", type=int, default=42)    # 随机种子
    parser.add_argument("--root", type=str, default="./data")    # torchvision 数据缓存根目录
    parser.add_argument("--eval-every", type=int, default=1)    # 多少轮评估一次客户端
    parser.add_argument("--eval-sample", type=int, default=100)    # 每次评估抽取多少客户端
    parser.add_argument(
        "--concurrent-clients",
        type=int,
        default=1,
        help="允许同时训练模型的客户端数量（每张卡内部并行度）。",
    )
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="*",
        default=None,
        help="指定可用 GPU ID（默认使用所有可见 GPU；若无 GPU 则退回 CPU）。",
    )
    parser.add_argument(
        "--cluster-cache",
        type=str,
        required=True,
        help="Required .npy file containing precomputed proxy cluster IDs.",
    )
    parser.add_argument(
        "--proxy-src-idx",
        type=str,
        default=None,
        help="路径到预选代理样本索引 numpy 文件（.npy）。索引顺序原样保留，必须与"
             "--cluster-cache 中的簇编号逐行对应。未提供时使用当前 split 前 "
             "proxy_size 个样本。",
    )
    parser.add_argument(
        "--proxy-labels",
        type=str,
        default=None,
        help="路径到代理样本真实标签 numpy 文件（.npy，目标数据集类别体系）。"
             "提供后异源代理场景下可计算代理标签准确率。",
    )
    parser.add_argument(
        "--proxy-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="代理数据来源划分：test（默认，测试集前半段）或 train（训练集，配合预生成索引文件使用）。",
    )
    # parser.add_argument(
    #     "--round-metrics-dir",
    #     type=str,
    #     default="./mid_data/cifar10/acc_of_test",
    #     help="保存每轮客户端全局测试准确率与聚合统计的目录。",
    # )
    parser.add_argument(
        "--arch-dist",
        type=float,
        nargs=3,
        metavar=("LIGHT", "STANDARD", "STRONG"),
        help="Distribution over light/standard/strong architectures.",
        default=[0, 0, 1],
    )    # 控制 light / standard / strong 三种架构的概率分布
    parser.add_argument(
        "--samples-per-client",
        type=int,
        default=6000,
        help="若设置为正整数，则每个客户端通过有放回采样获取固定数量的样本；为空时沿用 Dirichlet 划分。",
    )
    parser.add_argument(
        "--client-sample-counts",
        type=str,
        default=None,
        help='指定部分客户端的样本数，格式类似 "0:200,1:150"；其余客户端可与 samples-per-client 联合使用。',
    )
    parser.add_argument(
        "--round1-ckpt",
        type=str,
        default='./ckpt/cifar/0.1/run1',
        help=(
            "第一轮检查点目录。非分布式使用 round1_models.pt；分布式下每个 rank 使用"
            "独立的 round1_models_rank{rank}.pt。若当前 rank 对应的文件存在，则跳过"
            "第一轮训练直接从第二轮开始；否则完成第一轮后保存。"
            "代理数据聚类缓存请继续使用 --cluster-cache。"
        ),
    )
    parser.add_argument(
        "--no-final-ckpt",
        action="store_true",
        default=False,
        help=(
            "跑完不保存 final 检查点，只要指标。用于 seed 重复实验：final 目录名"
            "只由 eps/cohort/thr/distill 决定、**不含 seed**，换 seed 重跑会覆盖掉"
            "已有 seed 的检查点；同时每个 run 约 4.2G，多 seed 队列会撑爆磁盘。"
            "注意这样就无法事后用检查点重打分（评测口径现已干净，通常不需要）。"
        ),
    )
    parser.add_argument(
        "--soft-labels",
        action="store_true",
        default=False,
        help=(
            "在同源代理场景下也使用软标签 KL 蒸馏，而非 argmax→硬标签 CE 混合。"
            "异源代理（hetero_proxy）场景下始终使用软标签，此参数无需额外指定。"
        ),
    )
    parser.add_argument(
        "--distill-temp",
        type=float,
        default=0.5,
        help=(
            "软标签蒸馏温度 T（仅 --soft-labels 生效）。"
            "T < 1 对 DP 加噪后的平坦分布进行锐化，恢复类别信号；"
            "T = 1 不做处理；T > 1 进一步平滑。默认 0.5。"
        ),
    )
    parser.add_argument(
        "--proxy-ratio",
        type=float,
        default=1.0,
        help=(
            "软标签蒸馏时 proxy 样本数相对于客户端本地样本数的比例（仅 --soft-labels 生效）。"
            "默认 1.0（proxy 与本地等量）；设为 2.0 则最多使用 2 倍本地数量的 proxy 样本。"
        ),
    )
    parser.add_argument(
        "--report-encoding",
        choices=["soft", "vote", "sharpen"],
        default="soft",
        help=(
            "客户端上报 (K, C) 报告前的编码方式。每行都仍是概率分布，"
            "L2 sensitivity √(2K) 与噪声标定完全不变，区别只在聚合后 margin 的大小。"
            "soft=原始 softmax 均值（默认）；vote=每簇 top-1 one-hot 投票；"
            "sharpen=按 --report-sharpen-tau 做幂归一化锐化。"
            "round-1 实测干净簇级准确率：soft 0.6355 / vote 0.6910 / sharpen(0.25) 0.6910。"
        ),
    )
    parser.add_argument(
        "--report-sharpen-tau",
        type=float,
        default=0.25,
        help="--report-encoding sharpen 时的温度，越小越锐（0 < tau <= 1）。",
    )
    parser.add_argument(
        "--use-true-proxy-labels",
        action="store_true",
        default=False,
        help=(
            "异源代理场景下，用已知的映射真实标签（--proxy-labels 提供）作为 hard CE 训练目标，"
            "而非 DP 聚合伪标签。服务端 keep_indices 仍用于过滤可信聚类，preserve 预测-聚合-再训练流程。"
            "映射标签有效（≥0）时生效，否则自动回退到 DP 伪标签。"
        ),
    )
    parser.add_argument(
        "--eval-at-epoch",
        type=int,
        default=None,
        help=(
            "在每轮本地训练的第 N 个 epoch 结束后输出验证准确率（均值）。"
            "默认不启用（None）。示例：--eval-at-epoch 20 在第 20 epoch 后打印 val_acc@ep20。"
        ),
    )
    return parser.parse_args()


def _make_participant_reducer(is_distributed: bool):
    """Sum the per-rank participant counts so the DP noise share is calibrated
    to the GLOBAL cohort, not to one rank's slice of the clients."""
    if not is_distributed:
        return None

    def _reduce(local_count: int) -> int:
        t = torch.tensor([local_count], dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return max(1, int(t.item()))

    return _reduce


def _reject_legacy_privacy_flags(argv: List[str]) -> None:
    """--epsilon / --epsilon-local meant a per-round LOCAL budget; the flag now
    denotes the whole run's central budget. Fail loudly rather than silently
    reinterpreting an old command line as a very different privacy level."""
    for flag in ("--epsilon", "--epsilon-local", "--eps-per-label"):
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            raise SystemExit(
                f"{flag} has been removed. Its meaning changed: the mechanism is now "
                f"a distributed discrete Gaussian whose budget covers the ENTIRE run "
                f"under the aggregate view, not one local per-round report. "
                f"Use --epsilon-total (and see --dp-cohort-size)."
            )


def main():
    _reject_legacy_privacy_flags(sys.argv[1:])
    args = parse_args()
    if not os.path.isfile(args.cluster_cache):
        raise FileNotFoundError(
            f"--cluster-cache '{args.cluster_cache}' does not exist. Online "
            f"clustering is disabled; produce the cluster ids offline first."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    devices = _resolve_devices(args.device_ids)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = devices[local_rank % len(devices)]
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    is_distributed = world_size > 1
    if is_distributed:
        # NCCL requires one distinct GPU per rank; fall back to gloo when more
        # ranks than GPUs are launched (otherwise NCCL aborts with "invalid usage").
        n_gpu = torch.cuda.device_count()
        backend = "nccl" if (n_gpu >= world_size and torch.cuda.is_available()) else "gloo"
        if backend != "nccl" and rank == 0:
            print(f"[Dist] world_size={world_size} but {n_gpu} GPU(s) visible; "
                  f"using the '{backend}' backend.")
        dist.init_process_group(
            backend=backend, init_method="env://", rank=rank, world_size=world_size
        )

    round_metrics_dir = os.path.join("./mid_data", args.dataset.lower(), "acc_of_test", str(args.alpha))
    if rank == 0:
        os.makedirs(round_metrics_dir, exist_ok=True)
    client_sample_counts = _parse_client_sample_counts(args.client_sample_counts)

    max_world = max(world_size, 1)
    local_client_ids = [cid for cid in range(args.clients) if cid % max_world == rank]
    concurrent_clients = max(1, args.concurrent_clients if args.concurrent_clients is not None else 1)
    # concurrent_clients > 1 uses ThreadPoolExecutor; forking DataLoader workers
    # from a non-main thread deadlocks. Force num_workers=0 to avoid this.
    dl_workers = 0 if concurrent_clients > 1 else None
    client_data = create_client_data(
        dataset_name=args.dataset,
        num_clients=args.clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        root=args.root,
        seed=args.seed,
        architecture_distribution=args.arch_dist,
        samples_per_client=args.samples_per_client,
        client_sample_counts=client_sample_counts,
        target_client_ids=local_client_ids,
        val_ratio=args.val_ratio,
        exclude_test_indices=_proxy_indices_in_eval_split(args),
        train_dataloader_workers=dl_workers,
        device=device,
    )
    clients = build_clients(args.dataset, client_data, args.lr, args.optimizer, args.weight_decay, [device])

    # Must match the global_test_size used in create_client_data (default 5000).
    _EVAL_TEST_SIZE = 5000
    proxy_dataset_name = args.proxy_dataset  # None → homogeneous (same as --dataset)
    print(f"[Cluster cache] Loading precomputed proxy clusters from {args.cluster_cache}")
    if proxy_dataset_name is not None and proxy_dataset_name != args.dataset:
        proxy_bundle = prepare_heterogeneous_proxy(
            proxy_dataset_name=proxy_dataset_name,
            target_dataset_name=args.dataset,
            proxy_size=args.proxy_size,
            num_clusters=args.clusters,
            root=args.root,
            device=device,
            seed=args.seed,
            cluster_cache_path=args.cluster_cache,
            src_idx_path=args.proxy_src_idx,
            true_labels_path=args.proxy_labels,
        )
    else:
        proxy_bundle = prepare_clustered_proxy(
            dataset_name=args.dataset,
            proxy_size=args.proxy_size,
            num_clusters=args.clusters,
            root=args.root,
            device=device,
            seed=args.seed,
            cluster_cache_path=args.cluster_cache,
            global_test_size=_EVAL_TEST_SIZE if args.proxy_split == "test" else 0,
            src_idx_path=args.proxy_src_idx,
            true_labels_path=args.proxy_labels,
            proxy_split=args.proxy_split,
        )
    dp_config = DPConfig(
        epsilon=args.epsilon_total,
        delta=args.delta,
        q=args.q,
        num_clusters=args.clusters,
        num_rounds=args.rounds,
        cohort_size=args.dp_cohort_size,
        quantization_bits=args.dp_quantization_bits,
        accounting=args.dp_accounting,
        collusion_t=args.dp_collusion_t,
    )
    expected_participants = max(1, round(args.q * args.clients))
    log_dp_settings(dp_config, expected_participants)

    _hetero = proxy_dataset_name is not None and proxy_dataset_name != args.dataset
    system = SDDSystem(
        clients=clients,
        proxy_dataset=proxy_bundle.dataset,
        proxy_labels=proxy_bundle.true_labels,
        dataset_name=args.dataset,
        dp_config=dp_config,
        subsampling_rate=args.q,
        local_epochs=args.epochs,
        late_epochs=args.late_epochs,
        late_epochs_from_round=args.late_epochs_from_round,
        proxy_batch_size=args.proxy_batch_size,
        concurrent_clients=concurrent_clients,
        round_metrics_dir=round_metrics_dir,
        hetero_proxy=_hetero,
        distill_coeff=args.distill_coeff,
        use_soft_labels=args.soft_labels,
        distill_temp=args.distill_temp,
        proxy_ratio=args.proxy_ratio,
        eval_at_epoch=args.eval_at_epoch,
        use_true_proxy_labels=args.use_true_proxy_labels,
        participant_count_reducer=_make_participant_reducer(is_distributed),
        report_encoding=args.report_encoding,
        report_sharpen_tau=args.report_sharpen_tau,
    )
    if args.report_encoding != "soft":
        print(f"[Report] Client reports re-encoded as '{args.report_encoding}'"
              f"{f' (tau={args.report_sharpen_tau})' if args.report_encoding == 'sharpen' else ''};"
              f" rows stay on the simplex so sensitivity sqrt(2K) is unchanged.")
    if _hetero:
        print(f"[HeteroProxy] Training mode: supervised CE + soft-label KL distillation "
              f"(distill_coeff={args.distill_coeff}, temp={args.distill_temp}, "
              f"proxy_ratio={args.proxy_ratio})")
    elif args.soft_labels:
        print(f"[SoftLabels] Homogeneous proxy with soft-label KL distillation enabled "
              f"(distill_coeff={args.distill_coeff}, temp={args.distill_temp}, "
              f"proxy_ratio={args.proxy_ratio})")

    server = Server(
        num_clusters=args.clusters,
        num_classes=get_num_classes(args.dataset),
        confidence_threshold=args.confidence_threshold,
        quantization_scale=dp_config.quantization_scale,
    )
    history = {"round_accuracy": []}
    num_classes = get_num_classes(args.dataset)

    # Under distributed training each rank owns a disjoint set of clients and
    # writes its own checkpoint file, so ranks never race on the same path.
    ckpt_rank = rank if is_distributed else None

    # Round-1 checkpoint: load existing or arrange to save after round 1.
    # Only model weights are saved — pseudo_labels are NOT saved because they
    # depend on the privacy budget.  On load we skip training but re-run reporting
    # with the current budget, so pseudo_labels stay consistent with it.
    round1_ckpt_dir = args.round1_ckpt
    skip_round1_train = False
    if round1_ckpt_dir is not None:
        ckpt_models_path = os.path.join(
            round1_ckpt_dir, round1_checkpoint_filename(ckpt_rank)
        )
        if os.path.exists(ckpt_models_path):
            load_round1_checkpoint(clients, round1_ckpt_dir, device, rank=ckpt_rank)
            skip_round1_train = True
            print(f"[Checkpoint] Round-1 weights loaded from {ckpt_models_path}; "
                  f"will skip training in round 1 and re-run reporting with the "
                  f"current privacy budget.")
        else:
            print(f"[Checkpoint] No existing checkpoint at {ckpt_models_path}; "
                  f"will save after round 1.")

    if is_distributed:
        for rnd in range(1, args.rounds + 1):
            reports, _ = system._run_single_round(
                server=server,
                num_classes=num_classes,
                round_idx=rnd,
                skip_training=(skip_round1_train and rnd == 1),
            )
            gathered_reports = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_reports, reports)
            flat_reports = [r for sub in gathered_reports for r in sub]
            if rank == 0:
                server.aggregate_reports(flat_reports)
            pseudo_labels_payload = [server.pseudo_labels.cpu() if server.pseudo_labels is not None else None]
            dist.broadcast_object_list(pseudo_labels_payload, src=0)
            server.pseudo_labels = pseudo_labels_payload[0]
            keep_indices_payload = [server.keep_indices.cpu() if server.keep_indices is not None else None]
            dist.broadcast_object_list(keep_indices_payload, src=0)
            server.keep_indices = keep_indices_payload[0]
            # Save round-1 checkpoint (only model weights, no pseudo_labels).
            # Each rank writes its own file; barrier so no rank races ahead while
            # another is still writing into the shared directory.
            if rnd == 1 and round1_ckpt_dir is not None and not skip_round1_train:
                save_round1_checkpoint(system.clients, round1_ckpt_dir, rank=ckpt_rank)
                dist.barrier()
            # All K clusters are reported every round, so no pruning is needed.
            if rank == 0:
                proxy_stats_b = system._report_proxy_accuracy(server.pseudo_labels)
            else:
                proxy_stats_b = (0.0, 0, 0)
            local_records = [(cid, client.evaluate()) for cid, client in system.clients.items()]
            gathered_records = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_records, local_records)
            if rank == 0:
                all_records = [item for sub in gathered_records for item in sub]
                all_records.sort(key=lambda x: x[0])
                path = os.path.join(round_metrics_dir, f"round_{rnd:04d}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("client_id,accuracy\n")
                    for cid, acc in all_records:
                        f.write(f"{cid},{acc:.6f}\n")
                    agg_acc, agg_correct, agg_total = proxy_stats_b
                    f.write(f"\nkept_clusters={dp_config.num_clusters}\n")
                    f.write(f"aggregate_accuracy={agg_acc:.6f}\n")
                    f.write(f"aggregate_correct={agg_correct}\n")
                    f.write(f"aggregate_total={agg_total}\n")
                if all_records:
                    avg_acc = sum(acc for _, acc in all_records) / len(all_records)
                else:
                    avg_acc = 0.0
                history["round_accuracy"].append(avg_acc)
        dist.barrier()
    else:
        history = system.run(
            server=server,
            num_rounds=args.rounds,
            num_classes=num_classes,
            eval_every=args.eval_every,
            eval_clients=min(args.eval_sample, len(clients)),
            skip_round1_train=skip_round1_train,
            round1_ckpt_dir=round1_ckpt_dir,
            ckpt_rank=ckpt_rank,
        )

    # ---- Final evaluation ---------------------------------------------------
    # Every rank evaluates the clients it owns; results are gathered so that the
    # summary covers ALL clients, not just rank 0's local shard.
    # Save final weights; the path encodes every knob that changes the result,
    # so sweeps over epsilon / cohort / threshold do not overwrite each other.
    eps_tag = f"epstotal_{args.epsilon_total}".replace(".", "_")
    cohort_tag = (
        f"cohort{args.dp_cohort_size}" if args.dp_cohort_size is not None
        else "cohortactual"
    )
    thr_tag = f"thr{args.confidence_threshold}".replace(".", "_")
    # distill_coeff also changes the result, so it must be part of the path or a
    # no-distillation baseline would silently overwrite the matching main run.
    # Omitted at the default 1.0 to keep existing directory names valid.
    distill_tag = (
        "" if args.distill_coeff == 1.0
        else f"_distill{args.distill_coeff}".replace(".", "_")
    )
    if args.proxy_ratio != 1.0:
        distill_tag += f"_pratio{args.proxy_ratio}".replace(".", "_")
    # Soft-label KL vs hard-label CE changes the result but was not in the path,
    # so a hard-label run would have silently overwritten the matching soft-label
    # main run (same trap as distill_coeff above). Tagged only when soft labels
    # are OFF, so every existing --soft-labels directory name stays valid.
    if not args.soft_labels:
        distill_tag += "_hardlabels"
    if args.report_encoding != "soft":
        distill_tag += f"_enc{args.report_encoding}"
        if args.report_encoding == "sharpen":
            distill_tag += f"{args.report_sharpen_tau}".replace(".", "_")
    # The accountant changes sigma, and q now changes both the cohort and (under
    # rdp) the budget, so both must be in the path. Omitted at their frozen
    # defaults so every existing directory name stays valid.
    if args.dp_accounting != "zcdp":
        distill_tag += f"_acct{args.dp_accounting}"
    # The evaluation set itself changed; a run that keeps the old contaminated one
    # must not land on the same path as its cleaned counterpart.
    if args.keep_proxy_in_eval:
        distill_tag += "_evalcontam"
    if args.q != 1.0:
        distill_tag += f"_q{args.q}".replace(".", "_")
    final_ckpt_dir = os.path.join(
        os.path.dirname(round1_ckpt_dir) if round1_ckpt_dir else "./ckpt",
        f"proxy{args.proxy_size}_K{args.clusters}",
        f"final_{eps_tag}_{cohort_tag}_{thr_tag}{distill_tag}",
    )
    if args.no_final_ckpt:
        print(f"[Checkpoint] --no-final-ckpt: skipped saving final weights "
              f"(would have been {final_ckpt_dir})")
    else:
        save_final_checkpoint(clients, final_ckpt_dir, rank=ckpt_rank)

    if is_distributed:
        local_results = [(client.id, client.evaluate()) for client in clients.values()]
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)
        dist.barrier()
        eval_results = sorted(
            [item for sub in gathered for item in sub], key=lambda x: x[0]
        )
    else:
        eval_pool = list(clients.values())
        if len(eval_pool) > args.eval_sample:
            eval_pool = random.sample(eval_pool, args.eval_sample)
        eval_results = [(client.id, client.evaluate()) for client in eval_pool]

    if (not is_distributed) or rank == 0:
        final_accs = [acc for _, acc in eval_results]
        mean_acc = mean(final_accs) if final_accs else 0.0
        std_acc = pstdev(final_accs) if len(final_accs) > 1 else 0.0
        p10_acc = float(np.percentile(final_accs, 10)) if final_accs else 0.0
        max_cid, max_acc = max(eval_results, key=lambda x: x[1]) if eval_results else (-1, 0.0)
        min_cid, min_acc = min(eval_results, key=lambda x: x[1]) if eval_results else (-1, 0.0)

        print("==== Experiment Summary ====")
        print(f"Final mean client accuracy: {mean_acc:.4f}")
        print(f"Final std of client accuracy: {std_acc:.4f}")
        print(f"Final P10  client accuracy: {p10_acc:.4f}")
        print(f"Best  client: id={max_cid}, accuracy={max_acc:.4f}")
        print(f"Worst client: id={min_cid}, accuracy={min_acc:.4f}")
        if history["round_accuracy"]:
            print(f"Recorded round accuracies: {history['round_accuracy']}")

        summary_path = os.path.join(round_metrics_dir, "experiment_summary.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        proxy_src = args.proxy_dataset if args.proxy_dataset else args.dataset
        mode = "heterogeneous (CE + KL distill)" if _hetero else "homogeneous (CE + hard-label mix)"
        entry_lines = [
            f"==== Experiment Summary [{timestamp}] ====",
            f"dataset={args.dataset}",
            f"proxy_dataset={proxy_src}  # {mode}",
            f"distill_coeff={args.distill_coeff}",
            f"clients={args.clients}",
            f"alpha={args.alpha}",
            f"proxy_size={args.proxy_size}",
            f"clusters={args.clusters}",
            f"q={args.q}",
            f"rounds={args.rounds}",
            f"epochs={args.epochs}",
            f"epsilon_total={args.epsilon_total}",
            f"delta={args.delta}",
            "mechanism=distributed discrete Gaussian (multi-message shuffle, "
            "aggregate view)",
            f"l2_sensitivity=sqrt(2K)={dp_config.l2_sensitivity:.6g}",
            f"rho_zcdp_total={dp_config.rho_total:.6g}",
            f"sigma_on_aggregate={dp_config.sigma_aggregate:.6g}",
            f"dp_cohort_size={dp_config.effective_cohort(expected_participants)}"
            + ("  # SIMULATED (larger than this run's participants)"
               if dp_config.is_simulated_cohort(expected_participants) else ""),
            f"noise_std_on_aggregate="
            f"{dp_config.noise_std_on_average(expected_participants):.6g}",
            f"quantization_bits={args.dp_quantization_bits}",
            f"lr={args.lr}",
            f"optimizer={args.optimizer}",
            f"confidence_threshold={args.confidence_threshold}",
            f"seed={args.seed}",
            "",
            f"Final mean client accuracy: {mean_acc:.4f}",
            f"Final std of client accuracy: {std_acc:.4f}",
            f"Final P10  client accuracy: {p10_acc:.4f}",
            f"Best  client accuracy: {max_acc:.4f}  (client_id={max_cid})",
            f"Worst client accuracy: {min_acc:.4f}  (client_id={min_cid})",
            f"Evaluated clients: {len(eval_results)}/{args.clients}",
            (f"Final weights saved to: {final_ckpt_dir}"
             if not args.no_final_ckpt else "Final weights: not saved (--no-final-ckpt)"),
        ]
        if history["round_accuracy"]:
            entry_lines.append(f"Recorded round accuracies: {history['round_accuracy']}")
        entry_lines.append("=" * 60)
        entry_lines.append("")
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(entry_lines) + "\n")
        print(f"[Summary] Experiment summary appended to {summary_path}")
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
