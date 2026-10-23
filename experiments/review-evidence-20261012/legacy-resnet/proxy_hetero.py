"""
proxy_hetero.py — Heterogeneous proxy dataset support.

Provides prepare_heterogeneous_proxy(), a drop-in replacement for
cluster.prepare_clustered_proxy() where the proxy images come from a
*different* public dataset than the clients' private training data.

This simulates the realistic server scenario where the server only has
access to out-of-distribution public data for knowledge aggregation.

Supported proxy→target pairs:
  cifar100      → cifar10        (32×32, 3-ch; different taxonomy, similar domain)
  stl10         → cifar10        (96×96 → resized to 32×32, 3-ch)
  svhn          → cifar10        (32×32, 3-ch; digit images, OOD)
  fashion-mnist → mnist          (28×28, 1-ch)
  mnist         → fashion-mnist  (28×28, 1-ch)

New datasets can be registered in HETERO_PROXY_REGISTRY without changing
the rest of the codebase.

Return value is a ProxyBundle identical in structure to the homogeneous
case. Because the proxy taxonomy differs from the target, true_labels is
set to all -1 (unknown); downstream accuracy metrics are skipped.
"""

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from cluster import (
    KMEANS_N_INIT,
    SIMCLR_BATCH_SIZE,
    SIMCLR_EPOCHS,
    ClusteredProxyDataset,
    ProxyBundle,
    SimCLRDataset,
    SimCLRModel,
    SimCLRTensorAugment,
    _nt_xent_loss,
    _run_kmeans,
    load_precomputed_cluster_ids,
    load_proxy_src_indices,
)
from data import get_dataset_info


# ---------------------------------------------------------------------------
# Proxy dataset registry
# ---------------------------------------------------------------------------
# Each entry: callable(root, transform) → torchvision Dataset,
#             native image channel count, native spatial size (px).
# The transform passed to the loader is ToTensor (no normalization yet).
# ---------------------------------------------------------------------------

def _make_loader_cifar100(root, tf):
    return datasets.CIFAR100(root=root, train=False, download=True, transform=tf)


def _make_loader_cifar100_train(root, tf):
    return datasets.CIFAR100(root=root, train=True, download=True, transform=tf)


def _make_loader_stl10(root, tf):
    # Use test split (8 000 images); 'unlabeled' (100K) also works if downloaded.
    return datasets.STL10(root=root, split="test", download=True, transform=tf)


def _make_loader_svhn(root, tf):
    return datasets.SVHN(root=root, split="test", download=True, transform=tf)


def _make_loader_mnist(root, tf):
    return datasets.MNIST(root=root, train=False, download=True, transform=tf)


def _make_loader_fmnist(root, tf):
    return datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)


HETERO_PROXY_REGISTRY: dict = {
    "cifar100": {
        "loader": _make_loader_cifar100,
        "channels": 3,
        "native_size": 32,
    },
    "cifar100-train": {
        "loader": _make_loader_cifar100_train,
        "channels": 3,
        "native_size": 32,
    },
    "stl10": {
        "loader": _make_loader_stl10,
        "channels": 3,
        "native_size": 96,
    },
    "svhn": {
        "loader": _make_loader_svhn,
        "channels": 3,
        "native_size": 32,
    },
    "mnist": {
        "loader": _make_loader_mnist,
        "channels": 1,
        "native_size": 28,
    },
    "fashion-mnist": {
        "loader": _make_loader_fmnist,
        "channels": 1,
        "native_size": 28,
    },
}

# Normalization statistics for each supported target dataset.
# Used to bring proxy images into the same input space as client models.
_TARGET_NORM_STATS: dict = {
    "mnist":         ((0.1307,),                  (0.3081,)),
    "fashion-mnist": ((0.286,),                   (0.353,)),
    "cifar10":       ((0.4914, 0.4822, 0.4465),   (0.247, 0.243, 0.261)),
}

# Expected spatial resolution for each target dataset.
_TARGET_IMAGE_SIZE: dict = {
    "mnist": 28,
    "fashion-mnist": 28,
    "cifar10": 32,
}


def get_supported_proxy_datasets() -> list:
    """Returns sorted list of registered proxy dataset names."""
    return sorted(HETERO_PROXY_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_hetero_proxy_images(
    proxy_dataset_name: str,
    target_dataset_name: str,
    proxy_size: int,
    root: str,
    seed: int,
    src_indices: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Loads proxy_size images from proxy_dataset_name, resizes to the target
    dataset's spatial resolution, and applies the target dataset's
    normalization so that client models can process the images.

    Args:
        src_indices: Optional pre-computed integer array of sample indices to use.
                     When provided, random selection is skipped and these indices
                     are used directly (overrides proxy_size).

    Returns:
        FloatTensor of shape (N, C, H, W) normalized to target statistics.
    """
    proxy_cfg = HETERO_PROXY_REGISTRY[proxy_dataset_name]
    proxy_channels = proxy_cfg["channels"]
    native_size = proxy_cfg["native_size"]

    target_channels = get_dataset_info(target_dataset_name)["channels"]
    if proxy_channels != target_channels:
        raise ValueError(
            f"Channel mismatch: '{proxy_dataset_name}' has {proxy_channels} channels "
            f"but target '{target_dataset_name}' expects {target_channels}. "
            "Cross-channel proxy pairs are not supported."
        )

    target_size = _TARGET_IMAGE_SIZE.get(target_dataset_name.lower())
    if target_size is None:
        raise ValueError(
            f"No image-size entry for target dataset '{target_dataset_name}'. "
            f"Add it to _TARGET_IMAGE_SIZE."
        )

    # Build a raw transform: resize (if needed) + ToTensor. Normalization is
    # applied manually below so we can use the *target* dataset statistics.
    tf_list = []
    if native_size != target_size:
        tf_list.append(transforms.Resize((target_size, target_size)))
    tf_list.append(transforms.ToTensor())
    raw_transform = transforms.Compose(tf_list)

    proxy_ds = proxy_cfg["loader"](root, raw_transform)
    pool = len(proxy_ds)

    if src_indices is not None:
        indices = src_indices.tolist()
        if max(indices) >= pool:
            raise ValueError(
                f"src_indices contains index {max(indices)} but '{proxy_dataset_name}' "
                f"only has {pool} images."
            )
        print(f"[HeteroProxy] Using {len(indices)} pre-selected indices from src_indices.")
    else:
        if proxy_size > pool:
            raise ValueError(
                f"Requested proxy_size={proxy_size} exceeds available images "
                f"({pool}) in '{proxy_dataset_name}'."
            )
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(pool, generator=rng)[:proxy_size].tolist()

    subset = Subset(proxy_ds, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, drop_last=False)

    raw_images = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        raw_images.append(x.float())
    raw_images = torch.cat(raw_images, dim=0)  # (N, C, H, W) in [0, 1]

    # Apply target normalization (same preprocessing the client models expect)
    if target_dataset_name.lower() not in _TARGET_NORM_STATS:
        raise ValueError(
            f"No normalization stats for target '{target_dataset_name}'. "
            f"Add them to _TARGET_NORM_STATS."
        )
    mean, std = _TARGET_NORM_STATS[target_dataset_name.lower()]
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)
    images = (raw_images - mean_t) / std_t

    return images


# ---------------------------------------------------------------------------
# SimCLR training for heterogeneous proxy
# ---------------------------------------------------------------------------

def _train_simclr_hetero(
    images: torch.Tensor,
    in_channels: int,
    device: torch.device,
    seed: int,
) -> SimCLRModel:
    """
    Trains a SimCLR encoder on the heterogeneous proxy images.
    Functionally identical to cluster._train_simclr_encoder but takes
    in_channels directly instead of deriving it from a dataset name.
    """
    augment = SimCLRTensorAugment(images.size(-1), in_channels)
    contrastive_dataset = SimCLRDataset(images, augment)
    loader = DataLoader(
        contrastive_dataset,
        batch_size=min(SIMCLR_BATCH_SIZE, max(2, len(contrastive_dataset))),
        shuffle=True,
        drop_last=False,
    )
    model = SimCLRModel(in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(seed)
    random.seed(seed)
    model.train()
    for _ in range(SIMCLR_EPOCHS):
        for v1, v2 in loader:
            if v1.size(0) < 2:
                continue
            z1 = model(v1.to(device))
            z2 = model(v2.to(device))
            loss = _nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def compute_hetero_clusters_offline(
    images: torch.Tensor,
    in_channels: int,
    num_clusters: int,
    device: torch.device,
    seed: int,
) -> torch.LongTensor:
    """
    OFFLINE ONLY — SimCLR + k-means over heterogeneous proxy images.

    Never called from the training entry point: `prepare_heterogeneous_proxy`
    requires a pre-computed .npy file.  Use this from a standalone script to
    produce that file, then pass it via --cluster-cache.
    """
    print("[HeteroProxy/offline] Training SimCLR encoder on proxy images...")
    encoder = _train_simclr_hetero(images, in_channels, device, seed)
    with torch.no_grad():
        feats = []
        for (batch,) in DataLoader(TensorDataset(images), batch_size=256, shuffle=False):
            feats.append(encoder.encode(batch.to(device)).cpu())
        embeddings = F.normalize(torch.cat(feats, dim=0), dim=1)

    print(
        f"[HeteroProxy/offline] Running k-means "
        f"(K={num_clusters}, n_init={KMEANS_N_INIT})..."
    )
    cluster_ids = _run_kmeans(embeddings, num_clusters, seed=seed)
    sizes = [(cluster_ids == k).sum().item() for k in range(num_clusters)]
    print(
        f"[HeteroProxy/offline] Cluster sizes — "
        f"min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}"
    )
    return cluster_ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_heterogeneous_proxy(
    proxy_dataset_name: str,
    target_dataset_name: str,
    proxy_size: int,
    num_clusters: int,
    root: str,
    device: torch.device,
    seed: int,
    cluster_cache_path: Optional[str] = None,
    src_idx_path: Optional[str] = None,
    true_labels_path: Optional[str] = None,
) -> ProxyBundle:
    """
    Prepares a heterogeneous proxy dataset for cross-dataset distillation.

    Proxy images come from proxy_dataset_name but are resized and normalized
    to be compatible with models trained on target_dataset_name.

    Online clustering is disabled: cluster_cache_path is mandatory and must
    point at an existing .npy file produced offline.  SimCLR / K-Means are never
    run from this entry point and no new clustering is ever saved.

    By default, true_labels in the returned ProxyBundle is set to all -1
    (unknown taxonomy). If true_labels_path is provided, those labels are used
    instead, enabling proxy label accuracy evaluation.

    Args:
        proxy_dataset_name  : Source of public proxy data (e.g. 'cifar100').
        target_dataset_name : Clients' training dataset (e.g. 'cifar10').
        proxy_size          : Number of proxy images to sample.
        num_clusters        : Number of clusters K.
        root                : Torchvision data cache directory.
        device              : Unused (kept for signature compatibility).
        seed                : Random seed for the fallback deterministic sampling.
        cluster_cache_path  : REQUIRED .npy file with pre-computed cluster IDs.
        src_idx_path        : Optional .npy file with pre-selected sample indices.
                              When provided, sampling is skipped and these exact
                              indices are used, in order, to load images.
        true_labels_path    : Optional .npy file with ground-truth labels aligned
                              to the proxy samples (target-dataset taxonomy).

    Returns:
        ProxyBundle — same type as cluster.prepare_clustered_proxy().
    """
    if proxy_dataset_name not in HETERO_PROXY_REGISTRY:
        raise ValueError(
            f"Unknown proxy dataset '{proxy_dataset_name}'. "
            f"Supported: {get_supported_proxy_datasets()}"
        )
    if proxy_dataset_name == target_dataset_name:
        raise ValueError(
            "proxy_dataset_name and target_dataset_name are the same. "
            "Use cluster.prepare_clustered_proxy() for the homogeneous case."
        )
    if cluster_cache_path is None:
        raise ValueError(
            "--cluster-cache is required; online clustering is disabled."
        )
    if not os.path.isfile(cluster_cache_path):
        raise FileNotFoundError(cluster_cache_path)

    print(
        f"[HeteroProxy] proxy='{proxy_dataset_name}'  target='{target_dataset_name}'  "
        f"size={proxy_size}  K={num_clusters}"
    )

    # Load pre-selected indices if provided (order preserved; must line up with
    # the cluster-id file row-for-row).
    src_indices: Optional[np.ndarray] = None
    if src_idx_path is not None:
        pool_size = len(HETERO_PROXY_REGISTRY[proxy_dataset_name]["loader"](
            root, transforms.ToTensor()
        ))
        src_indices = load_proxy_src_indices(
            src_idx_path,
            proxy_size=proxy_size,
            pool_size=pool_size,
            reserved_start=None,  # proxy dataset is disjoint from the target test set
        )
    else:
        print(
            f"[HeteroProxy] No --proxy-src-idx given: falling back to the "
            f"deterministic seeded permutation (seed={seed}) of "
            f"'{proxy_dataset_name}'.\n"
            f"[HeteroProxy] !! '{cluster_cache_path}' MUST have been computed on "
            f"exactly these samples, in exactly this order (same seed). Passing "
            f"--proxy-src-idx is strongly recommended. !!"
        )

    images = _load_hetero_proxy_images(
        proxy_dataset_name, target_dataset_name, proxy_size, root, seed,
        src_indices=src_indices,
    )

    # ---- Cluster assignment (pre-computed only) ----------------------------
    cluster_ids = load_precomputed_cluster_ids(
        cluster_cache_path,
        expected_size=images.size(0),
        num_clusters=num_clusters,
    )
    sizes = [(cluster_ids == k).sum().item() for k in range(num_clusters)]
    print(
        f"[HeteroProxy] Cluster sizes — "
        f"min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}"
    )

    # ---- Bundle ---------------------------------------------------------------
    if true_labels_path is not None:
        if not os.path.isfile(true_labels_path):
            raise FileNotFoundError(
                f"true_labels_path '{true_labels_path}' does not exist."
            )
        raw = np.asarray(np.load(true_labels_path, allow_pickle=False)).reshape(-1)
        if raw.shape[0] != images.size(0):
            raise ValueError(
                f"true_labels file has {raw.shape[0]} entries but proxy has "
                f"{images.size(0)} images."
            )
        true_labels = torch.as_tensor(raw, dtype=torch.long)
        print(f"[HeteroProxy] Loaded ground-truth labels from '{true_labels_path}' "
              f"(proxy accuracy evaluation enabled).")
    else:
        # true_labels = -1 signals "labels unknown / not applicable".
        # sdd_core._report_proxy_accuracy checks for this and skips the metric.
        true_labels = torch.full((images.size(0),), -1, dtype=torch.long)

    dataset = ClusteredProxyDataset(images, cluster_ids)
    return ProxyBundle(dataset=dataset, true_labels=true_labels, cluster_ids=cluster_ids)


def prepare_heterogeneous_unclustered_proxy(
    proxy_dataset_name: str,
    target_dataset_name: str,
    proxy_size: int,
    root: str,
    seed: int,
    src_idx_path: Optional[str] = None,
    true_labels_path: Optional[str] = None,
) -> ProxyBundle:
    """
    Loads a heterogeneous proxy dataset WITHOUT clustering (for --no-parallel-comp ablation).

    Performs the same image loading and preprocessing as prepare_heterogeneous_proxy
    but skips SimCLR training and K-means. All samples receive dummy cluster_id=0.
    """
    if proxy_dataset_name not in HETERO_PROXY_REGISTRY:
        raise ValueError(
            f"Unknown proxy dataset '{proxy_dataset_name}'. "
            f"Supported: {get_supported_proxy_datasets()}"
        )

    print(
        f"[HeteroProxy] --no-parallel-comp: proxy='{proxy_dataset_name}'  "
        f"target='{target_dataset_name}'  size={proxy_size}  (no clustering)"
    )

    src_indices = None
    if src_idx_path is not None:
        if not os.path.exists(src_idx_path):
            print(f"[HeteroProxy] src_idx_path '{src_idx_path}' not found; falling back to random sampling.")
        else:
            src_indices = np.load(src_idx_path)
            print(f"[HeteroProxy] Loaded {len(src_indices)} pre-selected indices from '{src_idx_path}'.")

    images = _load_hetero_proxy_images(
        proxy_dataset_name, target_dataset_name, proxy_size, root, seed,
        src_indices=src_indices,
    )

    if true_labels_path is not None and os.path.exists(true_labels_path):
        raw = np.load(true_labels_path)
        if raw.shape[0] != images.size(0):
            raise ValueError(
                f"true_labels file has {raw.shape[0]} entries but proxy has "
                f"{images.size(0)} images."
            )
        true_labels = torch.as_tensor(raw, dtype=torch.long)
        print(f"[HeteroProxy] Loaded ground-truth labels from '{true_labels_path}'.")
    else:
        true_labels = torch.full((images.size(0),), -1, dtype=torch.long)
        if true_labels_path is not None:
            print(f"[HeteroProxy] true_labels_path '{true_labels_path}' not found; "
                  f"proxy accuracy will be skipped.")

    # Dummy cluster IDs: all zero — no cluster structure imposed
    cluster_ids = torch.zeros(images.size(0), dtype=torch.long)
    print(f"[HeteroProxy] Skipped clustering, {images.size(0)} samples loaded.")
    dataset = ClusteredProxyDataset(images, cluster_ids)
    return ProxyBundle(dataset=dataset, true_labels=true_labels, cluster_ids=cluster_ids)
