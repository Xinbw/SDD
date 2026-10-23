import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from data import get_dataset_info, load_datasets


# NOTE: the SimCLR / K-Means helpers below are OFFLINE tooling only. The training
# entry points (prepare_clustered_proxy / prepare_heterogeneous_proxy) never call
# them — cluster assignments must come from a pre-computed .npy file.
SIMCLR_BATCH_SIZE = 256
SIMCLR_EPOCHS = 10
SIMCLR_TEMPERATURE = 0.5
KMEANS_N_INIT = 5


class SimCLRTensorAugment:
    """Applies SimCLR-style augmentations directly on torch tensors."""

    def __init__(self, image_size: int, channels: int, min_scale: float = 0.6):
        self.image_size = image_size
        self.channels = channels
        self.min_scale = min_scale

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        scale = random.uniform(self.min_scale, 1.0)
        crop_size = max(1, int(scale * min(h, w)))
        top = random.randint(0, max(0, h - crop_size))
        left = random.randint(0, max(0, w - crop_size))
        cropped = x[:, top : top + crop_size, left : left + crop_size]
        resized = F.interpolate(
            cropped.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        aug = self._random_crop(x)
        if random.random() < 0.5:
            aug = torch.flip(aug, dims=[2])
        if random.random() < 0.3:
            aug = torch.flip(aug, dims=[1])
        if random.random() < 0.7:
            noise_scale = 0.05 if self.channels == 1 else 0.02
            aug = aug + torch.randn_like(aug) * noise_scale
        if self.channels == 3 and random.random() < 0.5:
            color_shift = torch.randn(3, 1, 1, device=aug.device) * 0.05
            aug = aug + color_shift
        return aug.clamp(-2.0, 2.0)


class SimCLRDataset(Dataset):
    """Return two augmented views for each proxy image."""

    def __init__(self, images: torch.Tensor, transform: SimCLRTensorAugment):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        return self.transform(image.clone()), self.transform(image.clone())


class SimCLRModel(nn.Module):
    """Small encoder + projection head."""

    def __init__(self, in_channels: int, feat_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projector(self.encode(x))
        return F.normalize(z, dim=1)


class ClusteredProxyDataset(Dataset):
    """Proxy dataset shared globally - contains images and cluster IDs."""

    def __init__(self, images: torch.Tensor, cluster_ids: torch.LongTensor):
        self.images = images
        self.cluster_ids = cluster_ids

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int):
        return self.images[idx], self.cluster_ids[idx]

    def filter_clusters(self, keep_indices: torch.Tensor) -> torch.Tensor:
        """
        Removes samples whose cluster IDs are not in keep_indices and reindexes remaining clusters.
        """
        keep_indices = keep_indices.to(torch.long).cpu()
        if keep_indices.numel() == 0:
            keep_mask = torch.zeros_like(self.cluster_ids, dtype=torch.bool)
            self.images = self.images[:0]
            self.cluster_ids = self.cluster_ids[:0]
            return keep_mask
        keep_list = keep_indices.tolist()
        keep_mask = torch.isin(self.cluster_ids, keep_indices)
        self.images = self.images[keep_mask]
        kept_ids = self.cluster_ids[keep_mask].clone()
        new_ids = kept_ids.clone()
        for new_idx, old_idx in enumerate(keep_list):
            new_ids[kept_ids == old_idx] = new_idx
        self.cluster_ids = new_ids
        return keep_mask


@dataclass
class ProxyBundle:
    dataset: ClusteredProxyDataset
    true_labels: torch.LongTensor
    cluster_ids: torch.LongTensor


def load_precomputed_cluster_ids(
    path: str,
    expected_size: int,
    num_clusters: int,
) -> torch.LongTensor:
    """
    Load and validate a pre-computed proxy cluster-assignment file.

    The file must contain exactly one cluster id per proxy sample, in the SAME
    order as the proxy samples themselves.  Every cluster 0..K-1 must be
    non-empty; missing/empty clusters are rejected rather than silently
    renumbered, because the DP sensitivity 2K and the (K, C) report shape both
    assume all K clusters exist.

    Args:
        path:          .npy file with shape (N,) or (N, 1).
        expected_size: Number of proxy samples N.
        num_clusters:  K.
    Returns:
        CPU torch.long tensor of shape (N,).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cluster id file not found: {path}")

    raw = np.load(path, allow_pickle=False)
    raw = np.asarray(raw).reshape(-1)

    if raw.shape[0] != expected_size:
        raise ValueError(
            f"Cluster id file '{path}' has {raw.shape[0]} entries but the proxy "
            f"set has {expected_size} samples. The cluster file must be built "
            f"from exactly these proxy samples, in the same order."
        )

    if raw.dtype.kind == "f":
        if not np.isfinite(raw).all():
            raise ValueError(f"Cluster id file '{path}' contains NaN/Inf values.")
        rounded = np.rint(raw)
        if not np.array_equal(raw, rounded):
            raise ValueError(
                f"Cluster id file '{path}' is floating point with non-integral "
                f"values; cluster ids must be integers."
            )
        ids = rounded.astype(np.int64)
    elif raw.dtype.kind in ("i", "u"):
        ids = raw.astype(np.int64)
    else:
        raise TypeError(
            f"Cluster id file '{path}' has unsupported dtype {raw.dtype}; "
            f"expected an integer or integral floating-point array."
        )

    if ids.min() < 0:
        raise ValueError(
            f"Cluster id file '{path}' contains negative cluster id {ids.min()}."
        )
    if ids.max() >= num_clusters:
        raise ValueError(
            f"Cluster id file '{path}' contains cluster id {ids.max()} but "
            f"num_clusters={num_clusters} (valid range 0..{num_clusters - 1})."
        )
    unique = np.unique(ids)
    if unique.shape[0] != num_clusters or unique[0] != 0 or unique[-1] != num_clusters - 1:
        missing = sorted(set(range(num_clusters)) - set(unique.tolist()))
        raise ValueError(
            f"Cluster id file '{path}' does not cover every cluster: expected "
            f"exactly {{0..{num_clusters - 1}}}, missing {missing}. Empty clusters "
            f"are rejected (no automatic renumbering) — regenerate the clustering "
            f"with a smaller K."
        )

    print(
        f"[Proxy clustering] Loaded {ids.shape[0]} cluster ids from '{path}' "
        f"(K={num_clusters}, all clusters non-empty)."
    )
    return torch.from_numpy(ids).to(torch.long)


def load_proxy_src_indices(
    path: str,
    proxy_size: int,
    pool_size: int,
    reserved_start: Optional[int] = None,
) -> np.ndarray:
    """
    Load and validate an explicit proxy sample-index file.

    Index ORDER is preserved verbatim: it defines the row order of the proxy
    set and therefore must match the order of the cluster-id file one-to-one.

    Args:
        path:           .npy file with shape (N,) or (N, 1).
        proxy_size:     Required number of indices.
        pool_size:      Total number of samples in the source split.
        reserved_start: If not None, indices >= this value belong to the held-out
                        global evaluation set and are rejected.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Proxy source-index file not found: {path}")

    raw = np.load(path, allow_pickle=False)
    raw = np.asarray(raw).reshape(-1)

    if raw.dtype.kind == "f":
        if not np.isfinite(raw).all():
            raise ValueError(f"Proxy index file '{path}' contains NaN/Inf values.")
        rounded = np.rint(raw)
        if not np.array_equal(raw, rounded):
            raise ValueError(
                f"Proxy index file '{path}' contains non-integral values."
            )
        indices = rounded.astype(np.int64)
    elif raw.dtype.kind in ("i", "u"):
        indices = raw.astype(np.int64)
    else:
        raise TypeError(
            f"Proxy index file '{path}' has unsupported dtype {raw.dtype}."
        )

    if indices.shape[0] != proxy_size:
        raise ValueError(
            f"Proxy index file '{path}' has {indices.shape[0]} entries but "
            f"proxy_size={proxy_size}."
        )
    if indices.min() < 0:
        raise ValueError(
            f"Proxy index file '{path}' contains a negative index {indices.min()}."
        )
    if indices.max() >= pool_size:
        raise ValueError(
            f"Proxy index file '{path}' contains index {indices.max()} but the "
            f"source split only has {pool_size} samples."
        )
    if np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError(
            f"Proxy index file '{path}' contains duplicate indices; each proxy "
            f"sample must appear exactly once."
        )
    if reserved_start is not None and indices.max() >= reserved_start:
        # Not fatal by project decision: existing index files are kept as-is.
        # Still surfaced loudly — these proxy samples are also in the set used to
        # report client accuracy, so reported accuracy is optimistically biased.
        n_bad = int((indices >= reserved_start).sum())
        print(
            f"[Proxy] WARNING: {n_bad} of {indices.shape[0]} indices in '{path}' fall "
            f"into the held-out global evaluation region [{reserved_start}, "
            f"{pool_size}). Proxy data overlaps the evaluation set; reported "
            f"client accuracy is biased upward by that overlap."
        )

    print(
        f"[Proxy] Loaded {indices.shape[0]} pre-selected proxy indices from "
        f"'{path}' (order preserved; must match the cluster-id file row-for-row)."
    )
    return indices


def _sample_proxy_subset(
    dataset_name: str,
    size: int,
    root: str,
    seed: int,
    use_deterministic_order: bool = False,
    explicit_indices: Optional[Sequence[int]] = None,
    global_test_size: int = 0,
    split: str = "test",
):
    """Samples proxy images from the specified split ("train" or "test").

    `global_test_size` reserves the last N samples of the test set for
    evaluation; only applies when split="test".
    """
    train_set, test_set = load_datasets(dataset_name, root)
    dataset = train_set if split == "train" else test_set
    reserve = global_test_size if split == "test" else 0
    pool = len(dataset) - reserve if reserve > 0 else len(dataset)
    if pool <= 0:
        raise ValueError("global_test_size leaves no samples available for proxy.")
    if explicit_indices is not None:
        indices = list(explicit_indices)
    elif use_deterministic_order:
        if size > pool:
            raise ValueError("Proxy size exceeds available samples.")
        indices = list(range(size))
    else:
        rng = torch.Generator().manual_seed(seed)
        if size > pool:
            raise ValueError("Proxy size exceeds available samples.")
        indices = torch.randperm(pool, generator=rng)[:size].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    images, labels = [], []
    for x, y in loader:
        images.append(x)
        labels.append(y)
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Standard NT-Xent loss used in SimCLR."""
    batch = z1.size(0)
    if batch < 2:
        return torch.tensor(0.0, device=z1.device)
    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(representations, representations.T) / SIMCLR_TEMPERATURE
    mask = torch.eye(2 * batch, device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))
    targets = torch.cat(
        [torch.arange(batch, 2 * batch), torch.arange(0, batch)]
    ).to(similarity.device)
    return F.cross_entropy(similarity, targets)


def _train_simclr_encoder(
    images: torch.Tensor,
    dataset_name: str,
    device: torch.device,
    seed: int,
) -> SimCLRModel:
    info = get_dataset_info(dataset_name)
    augment = SimCLRTensorAugment(images.size(-1), info["channels"])
    contrastive_dataset = SimCLRDataset(images, augment)
    loader = DataLoader(
        contrastive_dataset,
        batch_size=min(SIMCLR_BATCH_SIZE, max(2, len(contrastive_dataset))),
        shuffle=True,
        drop_last=False,
    )
    model = SimCLRModel(info["channels"]).to(device)
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


def _kmeans_plus_plus_init(
    embeddings: torch.Tensor,
    num_clusters: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """k-means++ initialization."""
    n = embeddings.size(0)
    centroids = torch.empty(num_clusters, embeddings.size(1), device=embeddings.device)
    first_idx = torch.randint(0, n, (1,), generator=generator).item()
    centroids[0] = embeddings[first_idx]
    distances = torch.full((n,), float("inf"), device=embeddings.device)
    for cid in range(1, num_clusters):
        diff = embeddings - centroids[cid - 1].unsqueeze(0)
        distances = torch.minimum(distances, torch.sum(diff * diff, dim=1))
        total = torch.sum(distances)
        if total <= 0:
            next_idx = torch.randint(0, n, (1,), generator=generator).item()
        else:
            probs = (distances / total).cpu()
            next_idx = torch.multinomial(probs, 1, generator=generator).item()
        centroids[cid] = embeddings[next_idx]
    return centroids


def _run_kmeans_once(
    embeddings: torch.Tensor,
    num_clusters: int,
    iters: int,
    generator: torch.Generator,
) -> torch.LongTensor:
    centroids = _kmeans_plus_plus_init(embeddings, num_clusters, generator)
    n = embeddings.size(0)
    assignments = torch.zeros(n, dtype=torch.long, device=embeddings.device)
    for _ in range(iters):
        dists = torch.cdist(embeddings, centroids)
        assignments = torch.argmin(dists, dim=1)
        for k in range(num_clusters):
            mask = assignments == k
            if mask.any():
                centroids[k] = embeddings[mask].mean(dim=0)
            else:
                centroids[k] = embeddings[torch.randint(0, n, (1,), generator=generator)]
    inertia = torch.sum((embeddings - centroids[assignments]) ** 2).item()
    return assignments.detach(), inertia


def _extract_pca_features(images: torch.Tensor, n_components: int = 50) -> torch.Tensor:
    """PCA via SVD on flattened images. Returns projected coordinates (N, n_components)."""
    x = images.reshape(images.size(0), -1).float()
    x = x - x.mean(dim=0)
    n_components = min(n_components, x.size(0) - 1, x.size(1))
    _, _, Vh = torch.linalg.svd(x, full_matrices=False)
    return x @ Vh[:n_components].T


class _ConvAutoencoder(nn.Module):
    """Simple conv autoencoder for image feature extraction."""

    def __init__(self, in_channels: int, image_size: int, latent_dim: int = 64):
        super().__init__()
        feat_h = image_size // 4
        flat_dim = 32 * feat_h * feat_h
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.encoder_fc = nn.Linear(flat_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(16, in_channels, 2, stride=2), nn.Sigmoid(),
        )
        self.feat_h = feat_h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_fc(self.encoder_conv(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        h = self.decoder_fc(z).reshape(-1, 32, self.feat_h, self.feat_h)
        return self.decoder_conv(h)


def _train_autoencoder(
    images: torch.Tensor,
    dataset_name: str,
    device: torch.device,
    seed: int,
    latent_dim: int = 64,
    epochs: int = 50,
) -> tuple:
    """Train a conv autoencoder; return (model, images_01) where images_01 ∈ [0,1]."""
    info = get_dataset_info(dataset_name)
    image_size = images.size(-1)
    model = _ConvAutoencoder(info["channels"], image_size, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    img_min, img_max = images.min(), images.max()
    images_01 = ((images - img_min) / (img_max - img_min + 1e-8)).cpu()
    loader = DataLoader(
        TensorDataset(images_01),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    torch.manual_seed(seed)
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = F.mse_loss(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"  [AE] epoch {ep+1}/{epochs}  loss={ep_loss/len(loader):.4f}")
    model.eval()
    return model, images_01


def _run_kmeans(
    embeddings: torch.Tensor,
    num_clusters: int,
    iters: int = 50,
    n_init: int = KMEANS_N_INIT,
    seed: int = 0,
) -> torch.LongTensor:
    generator = torch.Generator().manual_seed(seed)
    best_assignments = None
    best_inertia = float("inf")
    for _ in range(max(1, n_init)):
        assignments, inertia = _run_kmeans_once(embeddings, num_clusters, iters, generator)
        if inertia < best_inertia:
            best_inertia = inertia
            best_assignments = assignments
    return best_assignments.cpu()


def _compute_cluster_accuracy(
    cluster_ids: torch.LongTensor,
    true_labels: torch.LongTensor,
) -> float:
    """Majority label accuracy for cluster assignments."""
    if cluster_ids.numel() == 0:
        return 0.0
    array_ids = cluster_ids.numpy()
    labels = true_labels.numpy()
    total = len(array_ids)
    correct = 0
    for cid in np.unique(array_ids):
        mask = array_ids == cid
        if not np.any(mask):
            continue
        subset = labels[mask]
        majority = np.bincount(subset).argmax()
        correct += np.sum(subset == majority)
    return correct / max(total, 1)


def prepare_clustered_proxy(
    dataset_name: str,
    proxy_size: int,
    num_clusters: int,
    root: str,
    device: torch.device,
    seed: int,
    cluster_cache_path: Optional[str] = None,
    global_test_size: int = 0,
    src_idx_path: Optional[str] = None,
    true_labels_path: Optional[str] = None,
    proxy_split: str = "test",
) -> ProxyBundle:
    """
    Prepares proxy data and cluster assignments from a pre-computed .npy file.

    Online clustering is disabled: `cluster_cache_path` is mandatory and must
    point at an existing file.  SimCLR / K-Means remain in this module for
    offline scripts but are never invoked from the training entry point.

    `global_test_size` reserves the last N test samples for evaluation; proxy
    samples (explicit or deterministic) may never fall inside that region.

    When `src_idx_path` is provided those indices are used verbatim (order
    preserved) — the order must match `cluster_cache_path` row-for-row.
    Otherwise the first `proxy_size` samples of the split are used.
    """
    if cluster_cache_path is None:
        raise ValueError(
            "--cluster-cache is required; online clustering is disabled."
        )
    if not os.path.isfile(cluster_cache_path):
        raise FileNotFoundError(cluster_cache_path)

    # Determine the addressable pool so explicit indices can be range-checked.
    train_set, test_set = load_datasets(dataset_name, root)
    source_set = train_set if proxy_split == "train" else test_set
    pool_size = len(source_set)
    reserved_start = (
        pool_size - global_test_size
        if (proxy_split == "test" and global_test_size > 0)
        else None
    )

    explicit_indices = None
    if src_idx_path is not None:
        explicit_indices = load_proxy_src_indices(
            src_idx_path,
            proxy_size=proxy_size,
            pool_size=pool_size,
            reserved_start=reserved_start,
        ).tolist()
    else:
        print(
            f"[Proxy] No --proxy-src-idx given: using the FIRST {proxy_size} samples "
            f"of the '{proxy_split}' split in dataset order.\n"
            f"[Proxy] !! '{cluster_cache_path}' MUST have been computed on exactly "
            f"these samples, in exactly this order. !!"
        )

    images, labels = _sample_proxy_subset(
        dataset_name,
        proxy_size,
        root,
        seed,
        use_deterministic_order=(explicit_indices is None),
        explicit_indices=explicit_indices,
        global_test_size=global_test_size,
        split=proxy_split,
    )

    # Override labels if a labels file is provided
    if true_labels_path is not None:
        if not os.path.isfile(true_labels_path):
            raise FileNotFoundError(
                f"true_labels_path '{true_labels_path}' does not exist."
            )
        raw = np.load(true_labels_path, allow_pickle=False)
        raw = np.asarray(raw).reshape(-1)
        if raw.shape[0] != images.size(0):
            raise ValueError(
                f"true_labels file has {raw.shape[0]} entries but proxy has "
                f"{images.size(0)} images."
            )
        labels = torch.as_tensor(raw, dtype=torch.long)
        print(f"[Proxy] Loaded labels from '{true_labels_path}'.")

    cluster_ids = load_precomputed_cluster_ids(
        cluster_cache_path,
        expected_size=images.size(0),
        num_clusters=num_clusters,
    )
    acc = _compute_cluster_accuracy(cluster_ids, labels)
    print(
        f"[Proxy clustering] majority-label accuracy of the loaded clustering: "
        f"{acc:.4f}"
    )
    dataset = ClusteredProxyDataset(images, cluster_ids)
    return ProxyBundle(dataset=dataset, true_labels=labels, cluster_ids=cluster_ids)


def prepare_unclustered_proxy(
    dataset_name: str,
    proxy_size: int,
    root: str,
    seed: int,
    global_test_size: int = 0,
    src_idx_path: Optional[str] = None,
    true_labels_path: Optional[str] = None,
    proxy_split: str = "test",
) -> ProxyBundle:
    """
    Prepares proxy data WITHOUT clustering (for --no-parallel-comp ablation).

    Samples proxy images exactly as prepare_clustered_proxy does, but skips
    SimCLR training and K-means. All samples are assigned dummy cluster_id=0
    so the ClusteredProxyDataset contract is satisfied without any cluster
    structure being imposed.
    """
    explicit_indices = None
    if src_idx_path is not None:
        if not os.path.exists(src_idx_path):
            print(f"[Proxy] src_idx_path '{src_idx_path}' not found; falling back to default sampling.")
        else:
            explicit_indices = np.load(src_idx_path).tolist()
            print(f"[Proxy] Loaded {len(explicit_indices)} pre-selected indices from '{src_idx_path}'.")

    images, labels = _sample_proxy_subset(
        dataset_name,
        proxy_size,
        root,
        seed,
        use_deterministic_order=False,
        explicit_indices=explicit_indices,
        global_test_size=global_test_size,
        split=proxy_split,
    )

    if true_labels_path is not None:
        if not os.path.exists(true_labels_path):
            print(f"[Proxy] true_labels_path '{true_labels_path}' not found; using dataset labels.")
        else:
            raw = np.load(true_labels_path)
            if raw.shape[0] != images.size(0):
                raise ValueError(
                    f"true_labels file has {raw.shape[0]} entries but proxy has {images.size(0)} images."
                )
            labels = torch.as_tensor(raw, dtype=torch.long)
            print(f"[Proxy] Loaded labels from '{true_labels_path}'.")

    # Dummy cluster IDs: all zero — no cluster structure imposed
    cluster_ids = torch.zeros(images.size(0), dtype=torch.long)
    print(f"[Proxy] --no-parallel-comp: skipped clustering, {images.size(0)} samples loaded.")
    dataset = ClusteredProxyDataset(images, cluster_ids)
    return ProxyBundle(dataset=dataset, true_labels=labels, cluster_ids=cluster_ids)
