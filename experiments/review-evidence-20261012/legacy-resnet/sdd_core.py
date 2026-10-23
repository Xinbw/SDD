# sdd_core.py
import contextlib
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from cluster import ClusteredProxyDataset
from data import GpuTensorLoader


class _TensorLabelDataset(Dataset):
    """Wraps a dataset whose labels are plain ints, converting them to LongTensors.

    Needed when combining a torchvision dataset (int labels) with a TensorDataset
    (tensor labels) inside ConcatDataset — the default collate_fn cannot mix the two.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        x, y = self._dataset[idx]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        return x, y
from dp_utils import (
    DPConfig,
    add_distributed_discrete_gaussian,
    apply_laplace_noise,
    dequantize,
    project_rows_to_simplex,
)
from models import build_model


def _clean_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Strip the ``_orig_mod.`` prefix that torch.compile adds to state_dict keys."""
    sd = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    return sd


def round1_checkpoint_filename(rank: Optional[int] = None) -> str:
    """Round-1 checkpoint file name; rank-scoped under distributed training."""
    return "round1_models.pt" if rank is None else f"round1_models_rank{rank}.pt"


def final_checkpoint_filename(rank: Optional[int] = None) -> str:
    """Final checkpoint file name; rank-scoped under distributed training."""
    return "final_models.pt" if rank is None else f"final_models_rank{rank}.pt"


def save_round1_checkpoint(
    clients: Dict[int, "Client"],
    ckpt_dir: str,
    rank: Optional[int] = None,
) -> None:
    """
    Save client model state_dicts after round-1 local training.

    Only model weights are saved — pseudo_labels are NOT saved because they
    depend on the privacy budget.  When the checkpoint is loaded, round-1
    reporting is re-run with the current budget so that pseudo_labels are always
    consistent with the requested privacy level.

    Under distributed training each rank writes its OWN file
    (``round1_models_rank{rank}.pt``) containing only the clients it holds, so
    ranks never overwrite each other.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dicts = {cid: _clean_state_dict(client.model) for cid, client in clients.items()}
    path = os.path.join(ckpt_dir, round1_checkpoint_filename(rank))
    torch.save(state_dicts, path)
    print(f"[Checkpoint] Round-1 weights for {len(state_dicts)} clients saved to {path}")


def save_final_checkpoint(
    clients: Dict[int, "Client"],
    ckpt_dir: str,
    rank: Optional[int] = None,
) -> None:
    """Save all client model weights after the final training round (rank-scoped)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dicts = {cid: _clean_state_dict(client.model) for cid, client in clients.items()}
    path = os.path.join(ckpt_dir, final_checkpoint_filename(rank))
    torch.save(state_dicts, path)
    print(f"[Checkpoint] Final weights for {len(state_dicts)} clients saved to {path}")


def load_round1_checkpoint(
    clients: Dict[int, "Client"],
    ckpt_dir: str,
    device: torch.device,
    rank: Optional[int] = None,
) -> None:
    """
    Load client model state_dicts from a round-1 checkpoint directory.
    Each rank only loads weights for the clients it owns (matched by client_id).
    After loading, the caller should re-run round-1 reporting (skipping training)
    to obtain pseudo_labels consistent with the current privacy budget.
    """
    models_path = os.path.join(ckpt_dir, round1_checkpoint_filename(rank))
    state_dicts = torch.load(models_path, map_location=device)
    loaded = 0
    for cid, client in clients.items():
        if cid in state_dicts:
            target = client.model._orig_mod if hasattr(client.model, '_orig_mod') else client.model
            sd = state_dicts[cid]
            # Strip _orig_mod. prefix if checkpoint was saved from a torch.compile'd model
            if any(k.startswith("_orig_mod.") for k in sd):
                sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            target.load_state_dict(sd)
            loaded += 1
    print(f"[Checkpoint] Loaded model weights for {loaded}/{len(clients)} clients from {ckpt_dir}")


def _build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float = 5e-4):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {name}")


@dataclass
class ClientContext:
    client_id: int
    dataset_name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    architecture: str
    lr: float
    optimizer: str
    weight_decay: float
    device: torch.device


class Client:
    def __init__(self, ctx: ClientContext):
        self.id = ctx.client_id
        self.dataset_name = ctx.dataset_name
        self.train_loader = ctx.train_loader
        self.val_loader = ctx.val_loader
        self.test_loader = ctx.test_loader
        self.device = ctx.device
        self.model = build_model(ctx.architecture, ctx.dataset_name).to(self.device)
        # torch.compile fuses Conv+BN+ReLU sequences into single Triton kernels,
        # cutting ~60 kernel launches per ResNet-18 forward to a handful. Requires
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True so Triton's auto-tuner
        # (needs ~100 MB for L2-flush buffers) doesn't OOM under heavy model load.
        if self.device.type == 'cuda' and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        self.base_lr = ctx.lr
        self.optimizer = _build_optimizer(self.model, ctx.optimizer, ctx.lr, ctx.weight_decay)
        # AMP: float16 cuts compute time ~2× on CIFAR ResNet-18 (5.88ms→3ms fwd).
        bs = ctx.train_loader.batch_size or 32
        self.use_amp = (self.device.type == 'cuda') and (bs >= 64)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        # Dedicated CUDA stream per client so concurrent clients (ThreadPoolExecutor)
        # can submit GPU work simultaneously rather than serializing on the default stream.
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None

    def new_train(
        self,
        epochs: int,
        proxy_mix_dataset: Optional[TensorDataset] = None,
        distill_coeff: float = 1.0,
        proxy_ratio: float = 1.0,
        eval_at_epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train locally.

        Behaviour is determined by the label dtype in proxy_mix_dataset:

        - No proxy / hard labels (torch.long):
            Supervised CE on local data, with optional hard-label proxy images
            mixed in at a 1:1 ratio (original homogeneous behaviour).

        - Soft labels (torch.float):
            Two-phase training per epoch —
            Phase 1: supervised CE on local private data (unchanged).
            Phase 2: KL-divergence distillation on proxy images using the
                     server-aggregated soft pseudo-labels as targets.
            Loss = CE_local + distill_coeff * KL(log p_model || p_pseudo)
            This avoids forcing an argmax onto low-confidence OOD predictions.
        """
        has_proxy = proxy_mix_dataset is not None and len(proxy_mix_dataset) > 0
        use_soft = has_proxy and proxy_mix_dataset.tensors[1].dtype != torch.long
        # 3-tensor proxy dataset: (image, scaled_soft_label, original_conf_weight)
        # original_conf is computed before temperature scaling in _build_proxy_mix_dataset.
        has_precomputed_conf = use_soft and len(proxy_mix_dataset.tensors) == 3

        self.model.train()
        ce_criterion = nn.CrossEntropyLoss()

        # ---- Build loaders ---------------------------------------------------
        if use_soft:
            # Phase 1 uses the unmodified local loader.
            active_loader = self.train_loader
            # Phase 2 loader: subsample proxy by proxy_ratio relative to local data.
            n_local = len(self.train_loader.dataset)
            n_proxy = len(proxy_mix_dataset)
            n_sample = min(n_proxy, max(1, int(n_local * proxy_ratio)))
            if n_sample == n_proxy:
                # All proxy samples used — skip the randperm+index tensor copy
                # (3 × ~24 MB per client) and let the DataLoader handle shuffling.
                proxy_ds_for_loader = proxy_mix_dataset
            else:
                perm = torch.randperm(n_proxy)[:n_sample]
                if has_precomputed_conf:
                    proxy_tensors = (
                        proxy_mix_dataset.tensors[0][perm],
                        proxy_mix_dataset.tensors[1][perm],
                        proxy_mix_dataset.tensors[2][perm],
                    )
                else:
                    proxy_tensors = (
                        proxy_mix_dataset.tensors[0][perm],
                        proxy_mix_dataset.tensors[1][perm],
                    )
                proxy_ds_for_loader = TensorDataset(*proxy_tensors)
            proxy_loader: Optional[DataLoader] = DataLoader(
                proxy_ds_for_loader,
                batch_size=self.train_loader.batch_size or 32,
                shuffle=True,
                drop_last=False,
                pin_memory=(self.device.type == 'cuda'),
            )
        elif has_proxy:
            # Hard-label: mix proxy images directly into a single combined loader.
            n_local = len(self.train_loader.dataset)
            n_proxy = len(proxy_mix_dataset)
            n_sample = min(n_proxy, n_local)
            perm = torch.randperm(n_proxy)[:n_sample]
            proxy_images = proxy_mix_dataset.tensors[0][perm]
            proxy_labels = proxy_mix_dataset.tensors[1][perm]
            if isinstance(self.train_loader, GpuTensorLoader):
                # Under GPU preloading train_loader.dataset is a size-only stub
                # (data.GpuTensorLoader._Sized) with no __getitem__, so the
                # DataLoader/ConcatDataset route below raises TypeError. Keep the
                # data resident on GPU and concatenate the tensors instead; the
                # loader's augment_fn then covers the proxy images too, matching
                # the hard-label baseline that augments its mixed-in proxy.
                active_loader = GpuTensorLoader(
                    torch.cat([self.train_loader.images,
                               proxy_images.to(self.train_loader.images.device)]),
                    torch.cat([self.train_loader.labels,
                               proxy_labels.to(self.train_loader.labels.device).long()]),
                    self.train_loader.batch_size,
                    augment_fn=self.train_loader.augment_fn,
                )
            else:
                sampled_ds = TensorDataset(proxy_images, proxy_labels)
                combined_ds = ConcatDataset(
                    [_TensorLabelDataset(self.train_loader.dataset), sampled_ds]
                )
                active_loader = DataLoader(
                    combined_ds,
                    batch_size=self.train_loader.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=False,
                )
            proxy_loader = None
        else:
            active_loader = self.train_loader
            proxy_loader = None

        # ---- Training loop (dedicated CUDA stream for GPU/CPU overlap) ---------
        # With concurrent_clients > 1 each client runs in its own CUDA stream so
        # the GPU executes multiple clients' kernels in parallel instead of serially.
        _stream_ctx = (
            torch.cuda.stream(self.stream)
            if self.stream is not None
            else contextlib.nullcontext()
        )
        with _stream_ctx:
            total_ce_loss_gpu = torch.zeros(1, device=self.device)
            total_kl_loss_gpu = torch.zeros(1, device=self.device)
        total_ce_samples = 0
        total_kl_samples = 0

        # Soft-label case: build an infinite cycling iterator so each local batch
        # is paired with a proxy batch in a joint CE+KL update. Cycling the proxy
        # loader means it reshuffles each time it's exhausted, adding variety.
        def _inf_proxy_gen():
            while True:
                yield from proxy_loader

        _proxy_iter = _inf_proxy_gen() if (use_soft and proxy_loader is not None) else None

        mid_val_acc: Optional[float] = None
        for epoch_idx in range(epochs):
            self._set_optimizer_lr(self._scheduled_lr(epoch_idx))

            if _proxy_iter is not None:
                # Joint CE + KL in a single combined forward+backward pass.
                # Concatenating local and proxy images into one batch halves model
                # invocations and — especially after torch.compile fuses kernels —
                # halves the number of backward graph traversals per step.
                with _stream_ctx:
                    for images, labels in active_loader:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        px_batch = next(_proxy_iter)
                        if has_precomputed_conf:
                            px_imgs, px_soft, px_conf = px_batch
                            px_conf = px_conf.to(self.device, non_blocking=True).float()
                        else:
                            px_imgs, px_soft = px_batch
                            px_conf = None
                        px_imgs = px_imgs.to(self.device, non_blocking=True)
                        px_soft = px_soft.float().to(self.device, non_blocking=True)

                        n_local = images.size(0)
                        self.optimizer.zero_grad()

                        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                            combined_out = self.model(torch.cat([images, px_imgs], dim=0))
                            ce_loss = ce_criterion(combined_out[:n_local], labels)
                            kl_logits = combined_out[n_local:].float()
                            log_probs = F.log_softmax(kl_logits, dim=1)
                            conf_w = px_conf if px_conf is not None else px_soft.max(dim=1).values
                            kl_per = F.kl_div(log_probs, px_soft, reduction='none').sum(dim=1)
                            kl_loss = (kl_per * conf_w).sum() / conf_w.sum().clamp(min=1e-8)
                            total_loss = ce_loss + distill_coeff * kl_loss

                        if self.use_amp:
                            self.scaler.scale(total_loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            total_loss.backward()
                            self.optimizer.step()

                        total_ce_loss_gpu.add_(ce_loss.detach() * n_local)
                        total_ce_samples += n_local
                        total_kl_loss_gpu.add_(kl_loss.detach() * px_imgs.size(0))
                        total_kl_samples += px_imgs.size(0)
            else:
                # Phase 1: supervised CE (local data, or local+hard-proxy combined)
                with _stream_ctx:
                    for images, labels in active_loader:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        self.optimizer.zero_grad()
                        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                            logits = self.model(images)
                            loss = ce_criterion(logits, labels)
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                        total_ce_loss_gpu.add_(loss.detach() * images.size(0))
                        total_ce_samples += images.size(0)

            # Mid-epoch validation snapshot (optional, controlled by caller).
            if eval_at_epoch is not None and epoch_idx == eval_at_epoch - 1:
                self.model.eval()
                mid_val_acc = self.evaluate()
                self.model.train()

        # Sync dedicated stream before reading GPU accumulators.
        if self.stream is not None:
            self.stream.synchronize()
        result: Dict[str, float] = {
            "supervised_loss": total_ce_loss_gpu.item() / max(total_ce_samples, 1)
        }
        if use_soft:
            result["distill_loss"] = total_kl_loss_gpu.item() / max(total_kl_samples, 1)
        if mid_val_acc is not None:
            result[f"val_acc_at_ep{eval_at_epoch}"] = mid_val_acc
        return result

    def _scheduled_lr(self, epoch_idx: int) -> float:
        """
        Piecewise learning-rate schedule that emulates the reference scheduler:
        keep the base LR for the first 40 epochs, drop by 10x for epochs 40-79,
        and another 10x afterwards. The base LR itself is still configurable via
        CLI so users can scale the entire schedule.
        """
        if epoch_idx < 40:
            return self.base_lr
        if epoch_idx < 80:
            return self.base_lr * 0.1
        return self.base_lr * 0.01

    def _set_optimizer_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def evaluate_on_loader(self, loader: Optional[DataLoader]) -> float:
        if loader is None:
            return 0.0
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)

    def evaluate_validation(self) -> float:
        return self.evaluate_on_loader(self.val_loader)

    def evaluate(self) -> float:
        return self.evaluate_on_loader(self.test_loader)

    def report_clusters(
        self,
        proxy_loader: DataLoader,
        num_clusters: int,
        num_classes: int,
    ) -> torch.Tensor:
        return self._cluster_sums(proxy_loader, num_clusters, num_classes).cpu()

    def report_clusters_direct(
        self,
        images_gpu: torch.Tensor,
        cluster_ids_gpu: torch.Tensor,
        num_clusters: int,
        num_classes: int,
    ) -> torch.Tensor:
        """Cluster-averaged softmax probabilities, shape (K, C).

        Operates on pre-loaded GPU tensors to skip DataLoader overhead.
        Every one of the K clusters is reported every round (no partial
        cluster selection in this phase); each returned row is a probability
        distribution summing to 1.
        """
        bs = self.train_loader.batch_size or 256
        self.model.eval()
        _stream_ctx = (
            torch.cuda.stream(self.stream)
            if self.stream is not None
            else contextlib.nullcontext()
        )
        with _stream_ctx:
            sums = torch.zeros(num_clusters, num_classes, device=self.device)
            counts = torch.zeros(num_clusters, device=self.device)
            with torch.no_grad():
                for start in range(0, images_gpu.shape[0], bs):
                    imgs = images_gpu[start:start + bs]
                    cids = cluster_ids_gpu[start:start + bs]
                    with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                        logits = self.model(imgs)
                    probs = F.softmax(logits.float(), dim=-1)
                    expanded = cids.unsqueeze(-1).expand(-1, num_classes)
                    sums.scatter_add_(0, expanded, probs)
                    counts.scatter_add_(0, cids, torch.ones_like(cids, dtype=counts.dtype))
        if self.stream is not None:
            self.stream.synchronize()
        if torch.any(counts == 0):
            missing = torch.where(counts == 0)[0].tolist()
            raise RuntimeError(f"Empty proxy clusters: {missing}")
        cluster_probs = (sums / counts.unsqueeze(1)).cpu()
        row_sums = cluster_probs.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            raise RuntimeError(
                "cluster_probs rows are not probability distributions "
                f"(row sums range [{row_sums.min():.6f}, {row_sums.max():.6f}])."
            )
        return cluster_probs

    def report_per_sample_direct(
        self,
        images_gpu: torch.Tensor,
        epsilon_local: float,
        num_classes: int,
    ) -> torch.Tensor:
        """Per-sample reporting: returns privatised softmax probs, shape (N, C).

        No cluster averaging is performed — the client releases one probability
        vector per proxy sample. Because all N rows come from the same privately
        trained model, they form ONE joint release with L1 sensitivity 2N, so
        Laplace noise of scale 2N / epsilon_local is added to every coordinate
        and each row is then projected back onto the probability simplex.

        Used by the ablation entry point where clustering is disabled.
        """
        bs = self.train_loader.batch_size or 256
        self.model.eval()
        _stream_ctx = (
            torch.cuda.stream(self.stream)
            if self.stream is not None
            else contextlib.nullcontext()
        )
        all_probs = []
        with _stream_ctx:
            with torch.no_grad():
                for start in range(0, images_gpu.shape[0], bs):
                    imgs = images_gpu[start:start + bs]
                    with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                        logits = self.model(imgs)
                    all_probs.append(F.softmax(logits.float(), dim=-1).cpu())
        if self.stream is not None:
            self.stream.synchronize()
        per_sample_probs = torch.cat(all_probs, dim=0)  # (N, C)
        l1_sensitivity = 2.0 * per_sample_probs.shape[0]
        noisy = apply_laplace_noise(
            per_sample_probs,
            epsilon_local=epsilon_local,
            l1_sensitivity=l1_sensitivity,
        )
        return project_rows_to_simplex(noisy)

    def _cluster_sums(
        self,
        proxy_loader: DataLoader,
        num_clusters: int,
        num_classes: int,
    ) -> torch.Tensor:
        self.model.eval()
        sums = torch.zeros(num_clusters, num_classes, device=self.device)
        counts = torch.zeros(num_clusters, device=self.device)
        with torch.no_grad():
            for images, cluster_ids in proxy_loader:
                images = images.to(self.device, non_blocking=True)
                cluster_ids = cluster_ids.to(self.device)
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                # Cast to float32 before accumulation to avoid fp16 precision loss
                probs = F.softmax(logits.float(), dim=-1)
                expanded_ids = cluster_ids.unsqueeze(-1).expand(-1, num_classes)
                sums.scatter_add_(0, expanded_ids, probs)
                counts.scatter_add_(0, cluster_ids, torch.ones_like(cluster_ids, dtype=counts.dtype))
        counts = counts.unsqueeze(1).clamp(min=1.0)
        return sums / counts


class Server:
    def __init__(
        self,
        num_clusters: int,
        num_classes: int,
        confidence_threshold: float = 0.0,
        quantization_scale: int = 1,
    ):
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.pseudo_labels: Optional[torch.Tensor] = None
        self.keep_indices: Optional[torch.Tensor] = None
        self.confidence_threshold = confidence_threshold
        self.quantization_scale = quantization_scale

    def aggregate_reports(
        self,
        reports: Sequence[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Aggregate the clients' integer contributions into DP pseudo-labels.

        Under the multi-message shuffle protocol the server can only recover the
        SUM of the contributions — never an individual report — so aggregation is:

            1. sum the integer shares          (this is all the protocol reveals)
            2. de-quantise and divide by n     (mean cluster probability + DP noise)
            3. project each row onto the simplex   (post-processing, free)
            4. filter clusters by confidence ON THE AGGREGATE (post-processing)

        Step 4 replaces the old per-client confidence / per-cluster-median filter:
        those needed individual reports, which no longer exist. Thresholding the
        aggregate is strictly better here anyway — the aggregate is far less noisy
        than any single contribution ever was.
        """
        if not reports:
            self.pseudo_labels = None
            self.keep_indices = None
            self.num_clusters = 0
            return torch.empty(0, dtype=torch.long)

        n = len(reports)
        # (1) The only quantity the protocol exposes.
        total = torch.stack(
            [report["values"].to(torch.int64) for report in reports], dim=0
        ).sum(dim=0)                                         # (K, C) int64

        # (2) de-quantise -> mean probability vector per cluster (+ DP noise)
        avg = dequantize(total, self.quantization_scale, dtype=torch.float32) / n

        # (3) post-processing back onto the simplex, once, on the aggregate
        avg = project_rows_to_simplex(avg)

        # (4) confidence filter on the aggregated soft labels
        K = avg.shape[0]
        if self.confidence_threshold > 0.0:
            max_confs = avg.max(dim=1).values                # (K,)
            keep_indices = torch.where(max_confs >= self.confidence_threshold)[0]
            if keep_indices.numel() == 0:
                keep_indices = max_confs.argmax().unsqueeze(0)
                print(f"[Server] Warning: no cluster passed threshold "
                      f"{self.confidence_threshold:.2f}; keeping top-1.")
            elif keep_indices.numel() < K:
                print(f"[Server] Confidence filter: {keep_indices.numel()}/{K} "
                      f"clusters kept (threshold={self.confidence_threshold:.2f})")
            avg = avg[keep_indices]
        else:
            keep_indices = torch.arange(K, dtype=torch.long)

        self.pseudo_labels = avg
        self.keep_indices = keep_indices
        self.num_clusters = avg.shape[0]
        return keep_indices

    def aggregate_per_sample_reports(
        self,
        reports: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate per-sample privatised probability reports (unclustered path).

        Each report is a (N, C) tensor (one row per proxy sample, no cluster averaging).
        Returns keep_indices: sample indices that passed the confidence filter, or
        arange(N) when using the dynamic-median strategy (all samples retained but
        low-confidence clients filtered per-sample).
        """
        if not reports:
            self.pseudo_labels = None
            self.keep_indices = None
            return torch.empty(0, dtype=torch.long)

        values = torch.stack(reports, dim=0)  # (num_clients, N, C)
        N = values.shape[1]

        if self.confidence_threshold > 0.0:
            max_confs = values.max(dim=2).values          # (num_clients, N)
            conf_mask = max_confs >= self.confidence_threshold  # (num_clients, N) bool
            counts = conf_mask.float().sum(dim=0)         # (N,)
            sums = (values * conf_mask.unsqueeze(2)).sum(dim=0)  # (N, C)
            keep_indices = torch.where(counts > 0)[0]
            if keep_indices.numel() == 0:
                keep_indices = torch.arange(N, dtype=torch.long)
                avg = values.mean(dim=0)
                print(f"[Server] Warning: no sample passed threshold "
                      f"{self.confidence_threshold:.2f}; keeping all {N} samples.")
            else:
                safe_counts = counts[keep_indices].unsqueeze(1).clamp(min=1)
                avg_kept = sums[keep_indices] / safe_counts
                avg = torch.zeros(N, self.num_classes, dtype=torch.float)
                avg[keep_indices] = avg_kept
                n_kept = keep_indices.numel()
                if n_kept < N:
                    print(f"[Server] Per-sample confidence filter: "
                          f"{n_kept}/{N} samples kept "
                          f"(threshold={self.confidence_threshold:.2f})")
        else:
            max_confs = values.max(dim=2).values          # (num_clients, N)
            dyn_thresh, _ = max_confs.median(dim=0)       # (N,)
            conf_mask = max_confs >= dyn_thresh.unsqueeze(0)  # (num_clients, N) bool
            counts = conf_mask.float().sum(dim=0).clamp(min=1)  # (N,)
            sums = (values * conf_mask.unsqueeze(2)).sum(dim=0)  # (N, C)
            avg = sums / counts.unsqueeze(1)
            keep_indices = torch.arange(N, dtype=torch.long)

        self.pseudo_labels = avg
        self.keep_indices = keep_indices
        self.num_clusters = keep_indices.numel()
        return keep_indices

    def broadcast(self) -> Optional[torch.Tensor]:
        return self.pseudo_labels


class SDDSystem:
    """Semantic-DP Distillation orchestration."""

    def __init__(
        self,
        clients: Dict[int, Client],
        proxy_dataset: ClusteredProxyDataset,
        proxy_labels: torch.Tensor,
        dataset_name: str,
        dp_config: DPConfig,
        subsampling_rate: float,
        local_epochs: int,
        proxy_batch_size: int,
        concurrent_clients: int = 1,
        round_metrics_dir: Optional[str] = None,
        hetero_proxy: bool = False,
        distill_coeff: float = 1.0,
        use_soft_labels: bool = False,
        distill_temp: float = 1.0,
        proxy_ratio: float = 1.0,
        eval_at_epoch: Optional[int] = None,
        late_epochs: Optional[int] = None,
        late_epochs_from_round: int = 999,
        use_true_proxy_labels: bool = False,
        participant_count_reducer: Optional[Callable[[int], int]] = None,
        report_encoding: str = "soft",
        report_sharpen_tau: float = 0.25,
    ):
        # Under distributed training each rank only sees its own slice of the
        # participating clients, but the noise share must be calibrated to the
        # GLOBAL cohort. The reducer turns a local count into the global one.
        self.participant_count_reducer = participant_count_reducer
        if report_encoding not in ("soft", "vote", "sharpen"):
            raise ValueError(
                f"report_encoding must be soft|vote|sharpen, got {report_encoding!r}"
            )
        if not (0.0 < report_sharpen_tau <= 1.0):
            raise ValueError(
                f"report_sharpen_tau must be in (0, 1], got {report_sharpen_tau}"
            )
        self.report_encoding = report_encoding
        self.report_sharpen_tau = report_sharpen_tau
        self.clients = clients
        self.prediction_dataset = proxy_dataset
        self.prediction_labels = proxy_labels.clone()
        self.training_dataset = ClusteredProxyDataset(
            proxy_dataset.images.clone(),
            proxy_dataset.cluster_ids.clone(),
        )
        self.training_labels = proxy_labels.clone()
        self.hetero_proxy = hetero_proxy
        # use_true_proxy_labels: when mapped true labels are available (>=0), use them
        # as hard CE targets instead of DP pseudo-labels. Aggregation keep_indices still
        # filter which clusters are included, preserving the predict-aggregate-retrain flow.
        _has_valid_true_labels = (
            proxy_labels.numel() > 0 and proxy_labels.min().item() >= 0
        )
        self.use_true_proxy_labels = use_true_proxy_labels and _has_valid_true_labels
        if use_true_proxy_labels and not _has_valid_true_labels:
            print("[TrueProxyLabels] --use-true-proxy-labels requested but no valid "
                  "true labels found (all -1). Falling back to DP pseudo-labels.")
        if self.use_true_proxy_labels:
            print("[TrueProxyLabels] Enabled: proxy training will use mapped true labels "
                  "(hard CE) selected by server keep_indices.")
        # use_soft_labels: force soft-label KL distillation even for homogeneous proxy
        # Disabled when use_true_proxy_labels is active (hard CE takes precedence).
        self.use_soft_labels = (use_soft_labels or hetero_proxy) and not self.use_true_proxy_labels
        self.distill_coeff = distill_coeff
        self.distill_temp = distill_temp
        self.proxy_ratio = proxy_ratio
        self.eval_at_epoch = eval_at_epoch
        self.dataset_name = dataset_name.lower()
        self.dp_config = dp_config
        self.subsampling_rate = subsampling_rate
        self.local_epochs = local_epochs
        self.late_epochs = late_epochs
        self.late_epochs_from_round = late_epochs_from_round
        self.proxy_batch_size = proxy_batch_size
        self.concurrent_clients = max(1, int(concurrent_clients))
        self.round_metrics_dir = (
            round_metrics_dir
            if round_metrics_dir is not None
            else os.path.join("./mid_data", self.dataset_name, "acc_of_test")
        )
        os.makedirs(self.round_metrics_dir, exist_ok=True)
        self.prediction_loader = DataLoader(
            self.prediction_dataset,
            batch_size=self.proxy_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=True,
        )
        self.prediction_num_clusters = int(
            self.prediction_dataset.cluster_ids.max().item() + 1
        ) if len(self.prediction_dataset) > 0 else 0
        # The DP sensitivity 2K is derived from dp_config.num_clusters, so it must
        # match the number of clusters actually reported each round.
        if (
            self.prediction_num_clusters > 0
            and self.prediction_num_clusters != self.dp_config.num_clusters
        ):
            raise ValueError(
                f"Proxy clustering has {self.prediction_num_clusters} clusters but "
                f"dp_config.num_clusters={self.dp_config.num_clusters}; the L1 "
                f"sensitivity 2K would be wrong."
            )
        # Pre-load proxy images to GPU once; avoids repeated CPU→GPU transfers
        # during report_clusters (same data transferred n_participating times otherwise).
        _device = next(iter(clients.values())).device if clients else torch.device("cpu")
        self._proxy_images_gpu = self.prediction_dataset.images.to(_device, non_blocking=True)
        self._proxy_cluster_ids_gpu = self.prediction_dataset.cluster_ids.to(_device)

    def run(
        self,
        server: Server,
        num_rounds: int,
        num_classes: int,
        eval_every: int,
        eval_clients: int,
        skip_round1_train: bool = False,
        round1_ckpt_dir: Optional[str] = None,
        ckpt_rank: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        history = {"round_accuracy": []}
        for rnd in range(1, num_rounds + 1):
            skip = skip_round1_train and rnd == 1
            epochs_this_round = (
                self.late_epochs
                if self.late_epochs is not None and rnd >= self.late_epochs_from_round
                else self.local_epochs
            )
            reports, loss_records = self._run_single_round(
                server, num_classes, rnd,
                skip_training=skip, epochs=epochs_this_round,
            )
            proxy_stats = (0.0, 0, 0)
            if reports:
                keep_indices = self._aggregate(server, reports)
                proxy_stats = self._report_proxy_accuracy(server.pseudo_labels, server.keep_indices)
            else:
                keep_indices = None
            max_eval = eval_clients if rnd % eval_every == 0 else min(eval_clients // 5, 10)
            client_records = self._log_round_metrics(rnd, keep_indices, proxy_stats, max_eval=max_eval)
            if loss_records:
                avg_sup = sum(m["supervised_loss"] for m in loss_records) / max(len(loss_records), 1)
                print(f"[Round {rnd}] train_loss={avg_sup:.4f}", flush=True)
                if any("distill_loss" in m for m in loss_records):
                    avg_kl = sum(m.get("distill_loss", 0.0) for m in loss_records) / max(len(loss_records), 1)
                    print(f"[Round {rnd}] distill_loss(KL)={avg_kl:.4f}", flush=True)
                if self.eval_at_epoch is not None:
                    key = f"val_acc_at_ep{self.eval_at_epoch}"
                    vals = [m[key] for m in loss_records if key in m]
                    if vals:
                        print(f"[Round {rnd}] val_acc@ep{self.eval_at_epoch}={sum(vals)/len(vals):.4f}", flush=True)
            if rnd == 1 and round1_ckpt_dir is not None and not skip_round1_train:
                save_round1_checkpoint(self.clients, round1_ckpt_dir, rank=ckpt_rank)
            if rnd % eval_every == 0:
                if client_records:
                    values = [acc for _, acc in client_records if not (isinstance(acc, float) and acc != acc)]
                    acc = sum(values) / max(len(values), 1) if values else 0.0
                    p10 = float(np.percentile(values, 10)) if values else 0.0
                else:
                    acc = 0.0
                    p10 = 0.0
                history["round_accuracy"].append(acc)
                print(f"[Round {rnd}] valid_acc={acc:.4f}  p10_acc={p10:.4f}", flush=True)
        return history

    def _run_single_round(
        self,
        server: Server,
        num_classes: int,
        round_idx: int,
        skip_training: bool = False,
        epochs: Optional[int] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, float]]]:
        epochs = epochs if epochs is not None else self.local_epochs
        participating_clients = self._subsample_clients()
        if not participating_clients:
            return [], []
        pseudo_labels = server.broadcast()
        reports: List[Dict[str, torch.Tensor]] = []
        loss_records: List[Dict[str, float]] = []

        # Build proxy mix dataset once per round (shared across all clients).
        # Returns None in round 1 when no pseudo-labels exist yet.
        proxy_mix_dataset: Optional[TensorDataset] = None
        if pseudo_labels is not None and len(self.training_dataset) > 0:
            proxy_mix_dataset = self._build_proxy_mix_dataset(pseudo_labels, server.keep_indices)

        if not skip_training:
            def _train_client(target_client: Client) -> Dict[str, float]:
                metrics = target_client.new_train(
                    epochs, proxy_mix_dataset, self.distill_coeff,
                    self.proxy_ratio, self.eval_at_epoch,
                )
                metrics["client_id"] = target_client.id
                metrics["val_acc"] = target_client.evaluate_validation()
                return metrics

            if self.concurrent_clients > 1:
                with ThreadPoolExecutor(max_workers=self.concurrent_clients) as executor:
                    future_map = {
                        executor.submit(_train_client, client): client.id
                        for client in participating_clients
                    }
                    for future in as_completed(future_map):
                        loss_records.append(future.result())
            else:
                for client in participating_clients:
                    loss_records.append(_train_client(client))
                    print(f"[Local progress] completed client {client.id}", flush=True)

        loss_records.sort(key=lambda rec: rec["client_id"])
        reports = self._collect_reports(participating_clients, num_classes, round_idx)
        return reports, loss_records

    def _collect_reports(
        self,
        participating_clients: List[Client],
        num_classes: int,
        round_idx: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Privatised client contributions for one round.

        Distributed discrete Gaussian: each client quantises its report and adds
        only a 1/n share of the noise. An individual contribution is NOT private
        on its own — the anonymous multi-message shuffler (split-and-mix) ensures
        the server can only recover the SUM, which is where the DP guarantee
        lives. The shares recombine to exactly that sum, so the protocol itself
        changes no number and is not simulated here.

        Overridden by the ablation entry point, which pins the legacy local-DP
        mechanism (full-strength per-client noise on individually visible reports).
        """
        reports: List[Dict[str, torch.Tensor]] = []
        n_participants = len(participating_clients)
        if self.participant_count_reducer is not None:
            n_participants = self.participant_count_reducer(n_participants)
        client_std = self.dp_config.client_noise_std(n_participants)
        if round_idx == 1:
            print(f"[DP] {n_participants} reporting clients | "
                  f"per-client share std={client_std:.6f} | "
                  f"noise std on aggregated pseudo-label="
                  f"{self.dp_config.noise_std_on_average(n_participants):.6f}",
                  flush=True)
        for client in participating_clients:
            cluster_probs = client.report_clusters_direct(
                self._proxy_images_gpu,
                self._proxy_cluster_ids_gpu,
                self.prediction_num_clusters,
                num_classes,
            )  # (K, C)
            cluster_probs = self._encode_report(cluster_probs, num_classes)
            contribution = add_distributed_discrete_gaussian(
                cluster_probs,
                client_noise_std=client_std,
                quantization_scale=self.dp_config.quantization_scale,
            )  # int64 (K, C)
            reports.append({"values": contribution})
        return reports

    def _encode_report(self, cluster_probs: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Re-encode one client's (K, C) report before it is privatised.

        Every encoding leaves each row a probability distribution, so the L2
        sensitivity stays sqrt(2K) and the noise calibration is untouched — the
        encodings differ only in how much of the aggregate's margin survives.

        Measured on the round-1 checkpoints (see diagnose_round1_noise.py
        --encodings): the clean cluster-level pseudo-label accuracy is 0.6355
        for "soft", 0.6910 for both "vote" and "sharpen" at tau=0.25, and the
        median top1-top2 margin of the aggregate goes 0.018 -> 0.040 for "vote".
        Soft averaging lets a handful of very confident but wrong clients pull
        the mean; a vote cannot.
        """
        if self.report_encoding == "soft":
            return cluster_probs
        if self.report_encoding == "vote":
            idx = cluster_probs.argmax(dim=-1)
            return F.one_hot(idx, num_classes).to(cluster_probs.dtype)
        if self.report_encoding != "sharpen":
            raise ValueError(f"unknown report_encoding {self.report_encoding!r}")
        sharp = cluster_probs.clamp(min=0).pow(1.0 / self.report_sharpen_tau)
        return sharp / sharp.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    def _aggregate(
        self,
        server: Server,
        reports,
    ) -> Optional[torch.Tensor]:
        """Dispatch to the correct server aggregation method.

        Subclasses override this to switch between cluster-level and per-sample
        aggregation without duplicating the run() loop logic.
        """
        return server.aggregate_reports(reports)

    def _subsample_clients(self) -> List[Client]:
        selected = [
            client
            for client in self.clients.values()
            if random.random() < self.subsampling_rate
        ]
        if not selected and self.clients:
            selected = [random.choice(list(self.clients.values()))]
        return selected

    def evaluate_clients(self, sample_size: int) -> float:
        client_list = list(self.clients.values())
        if sample_size < len(client_list):
            client_list = random.sample(client_list, sample_size)
        accuracies = [client.evaluate() for client in client_list]
        return sum(accuracies) / max(len(accuracies), 1)

    def _build_proxy_mix_dataset(
        self,
        pseudo_labels: torch.Tensor,
        keep_indices: Optional[torch.Tensor],
    ) -> Optional[TensorDataset]:
        """Build a CPU TensorDataset for proxy-assisted local training.

        Two modes:
        - use_true_proxy_labels=True: uses mapped ground-truth labels (hard CE).
          Aggregation keep_indices still gate which clusters are included, so the
          predict-aggregate step contributes to sample selection.
        - Default: converts DP-averaged soft pseudo-labels to hard/soft labels
          depending on use_soft_labels flag.
        """
        if keep_indices is None or keep_indices.numel() == 0:
            return None
        old_to_new = {int(old): new for new, old in enumerate(keep_indices.tolist())}
        cluster_ids = self.training_dataset.cluster_ids
        images = self.training_dataset.images
        kept_mask = torch.isin(cluster_ids, keep_indices)
        if not kept_mask.any():
            return None
        kept_images = images[kept_mask].cpu()
        kept_old_ids = cluster_ids[kept_mask]
        kept_new_ids = torch.tensor(
            [old_to_new[int(oid)] for oid in kept_old_ids.tolist()],
            dtype=torch.long,
        )

        if self.use_true_proxy_labels:
            # Hard CE on mapped ground-truth labels; DP noise does not corrupt signal.
            # keep_indices ensures we only train on clusters the server deemed reliable.
            true_labels = self.training_labels[kept_mask].long().cpu()
            return TensorDataset(kept_images, true_labels)

        if self.use_soft_labels:
            # Soft-label path: keep the full server-averaged probability distribution.
            # Client.new_train detects float dtype and switches to KL distillation.
            original_soft = pseudo_labels.cpu()[kept_new_ids].float()  # (N, C)
            # Record per-sample confidence BEFORE temperature scaling so that
            # conf_weights in the KL loss reflect the true post-noise confidence
            # rather than the artificially sharpened (T<1) distribution.
            original_conf = original_soft.max(dim=1).values  # (N,)
            scaled_soft = original_soft.clone()
            if self.distill_temp != 1.0 and self.distill_temp > 0:
                # Power-normalise: p_sharp = p^(1/T) / sum(p^(1/T)).
                # T < 1 sharpens (recovers signal lost to DP noise).
                # T > 1 softens further.
                scaled_soft = scaled_soft.pow(1.0 / self.distill_temp)
                scaled_soft = scaled_soft / scaled_soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
            # 3-tensor dataset: (image, scaled_soft_label, original_conf)
            return TensorDataset(kept_images, scaled_soft, original_conf)
        else:
            # Hard-label path: collapse to argmax for supervised CE mixing.
            hard_labels = pseudo_labels.cpu()[kept_new_ids].argmax(dim=1)  # (N,) long
            return TensorDataset(kept_images, hard_labels)

    def _report_proxy_accuracy(
        self,
        pseudo_labels: Optional[torch.Tensor],
        keep_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[float, int, int]:
        if pseudo_labels is None or len(self.training_dataset) == 0:
            print("[Proxy labels] accuracy: 0.0000 (no pseudo-labels available yet)")
            return 0.0, 0, 0
        # Heterogeneous proxy: true_labels are all -1 (unknown taxonomy).
        # Accuracy comparison is meaningless; skip and report N/A.
        if self.training_labels.numel() > 0 and self.training_labels.min().item() < 0:
            print("[Proxy labels] label_accuracy=N/A (heterogeneous proxy — no ground-truth labels)")
            return -1.0, 0, 0
        preds = torch.argmax(pseudo_labels, dim=1)  # (K',)
        cluster_ids = self.training_dataset.cluster_ids
        labels = self.training_labels
        total = labels.numel()
        if total == 0:
            print("[Proxy labels] accuracy after filtering: 0.0000 (no samples kept)")
            return 0.0, 0, 0
        if keep_indices is None:
            keep_indices = torch.arange(pseudo_labels.size(0), dtype=torch.long)
        correct = 0
        covered = 0
        for new_cid, orig_cid in enumerate(keep_indices.tolist()):
            mask = cluster_ids == orig_cid
            if not torch.any(mask):
                continue
            n = mask.sum().item()
            covered += n
            correct += (labels[mask] == preds[new_cid]).sum().item()
        label_acc = correct / covered if covered > 0 else 0.0
        coverage = covered / total if total > 0 else 0.0
        print(
            f"[Proxy labels] label_accuracy={label_acc:.4f} "
            f"coverage={coverage:.4f} ({covered}/{total}) "
            f"correct={correct:d}"
        )
        return label_acc, correct, total

    def _log_round_metrics(
        self,
        round_idx: int,
        keep_indices: Optional[torch.Tensor],
        proxy_stats: Tuple[float, int, int],
        max_eval: int = 0,
    ) -> List[Tuple[int, float]]:
        path = os.path.join(self.round_metrics_dir, f"round_{round_idx:04d}.txt")
        client_records: List[Tuple[int, float]] = []
        all_ids = sorted(self.clients.keys())
        # Limit evaluation to a sample to avoid O(n) test passes every round.
        eval_ids = (
            random.sample(all_ids, min(max_eval, len(all_ids)))
            if max_eval > 0 and max_eval < len(all_ids)
            else all_ids
        )
        eval_set = set(eval_ids)
        for client_id in all_ids:
            if client_id in eval_set:
                acc = self.clients[client_id].evaluate()
            else:
                acc = float("nan")
            client_records.append((client_id, acc))
        num_clusters = keep_indices.numel() if keep_indices is not None else 0
        agg_acc, agg_correct, agg_total = proxy_stats
        valid_accs = [acc for _, acc in client_records if acc == acc]  # 过滤 NaN
        mean_acc = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
        p10_acc = float(np.percentile(valid_accs, 10)) if valid_accs else 0.0
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Round {round_idx} client accuracies:\n")
            f.write("client_id,accuracy\n")
            for cid, acc in client_records:
                f.write(f"{cid},{acc:.6f}\n" if acc == acc else f"{cid},N/A\n")
            f.write("\nClient accuracy summary:\n")
            f.write(f"mean_accuracy={mean_acc:.6f}\n")
            f.write(f"p10_accuracy={p10_acc:.6f}\n")
            f.write(f"evaluated_clients={len(valid_accs)}\n")
            f.write("\nAggregated label summary:\n")
            f.write(f"kept_clusters={num_clusters}\n")
            if agg_acc < 0:
                f.write("aggregate_accuracy=N/A (heterogeneous proxy)\n")
            else:
                f.write(f"aggregate_accuracy={agg_acc:.6f}\n")
            f.write(f"aggregate_correct={agg_correct}\n")
            f.write(f"aggregate_total={agg_total}\n")
        return client_records
