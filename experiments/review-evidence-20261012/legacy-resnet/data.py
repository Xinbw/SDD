import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class MemDataset(Dataset):
    """Pre-loads an entire dataset into CPU RAM as float tensors.

    Eliminates per-sample PIL conversion and transform overhead during training.
    Only suitable for datasets with fully deterministic transforms (no random aug).
    Exposes a `.targets` attribute for compatibility with _get_targets().
    """
    def __init__(self, src: Dataset):
        xs, ys = [], []
        for x, y in src:
            xs.append(x)
            ys.append(int(y))
        self._data = torch.stack(xs)          # (N, C, H, W) float32
        self._labels = torch.tensor(ys, dtype=torch.long)
        self.targets = np.array(ys)           # numpy for _get_targets()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], int(self._labels[idx])

def _cifar10_gpu_aug(imgs: torch.Tensor) -> torch.Tensor:
    """Vectorized random crop (pad=4) + horizontal flip for a batch of CIFAR-10 images on GPU."""
    N, C, H, W = imgs.shape
    pad = 4
    padded = F.pad(imgs, [pad, pad, pad, pad], mode='reflect')  # (N, C, 40, 40)
    tops  = torch.randint(0, 2 * pad + 1, (N,), device=imgs.device)
    lefts = torch.randint(0, 2 * pad + 1, (N,), device=imgs.device)
    row_idx = tops.view(N, 1)  + torch.arange(H, device=imgs.device).view(1, H)   # (N, H)
    col_idx = lefts.view(N, 1) + torch.arange(W, device=imgs.device).view(1, W)   # (N, W)
    n_idx = torch.arange(N, device=imgs.device).view(N, 1, 1, 1).expand(N, C, H, W)
    c_idx = torch.arange(C, device=imgs.device).view(1, C, 1, 1).expand(N, C, H, W)
    r_idx = row_idx.view(N, 1, H, 1).expand(N, C, H, W)
    l_idx = col_idx.view(N, 1, 1, W).expand(N, C, H, W)
    cropped = padded[n_idx, c_idx, r_idx, l_idx]
    flip = torch.rand(N, device=imgs.device) < 0.5
    cropped[flip] = cropped[flip].flip(-1)
    return cropped


class GpuTensorLoader:
    """Holds an entire client's training data on GPU; yields augmented batches each epoch.

    Drop-in replacement for DataLoader when GPU preloading is enabled. Exposes
    .batch_size and .dataset (sized) for compatibility with the training loop.
    """

    class _Sized:
        def __init__(self, n: int) -> None:
            self._n = n
        def __len__(self) -> int:
            return self._n

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        augment_fn=None,
    ) -> None:
        self.images = images        # (N, C, H, W) float32 on GPU
        self.labels = labels        # (N,) long on GPU
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.dataset = self._Sized(len(images))

    def __len__(self) -> int:
        return math.ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        N = len(self.images)
        perm = torch.randperm(N, device=self.images.device)
        for start in range(0, N, self.batch_size):
            idx = perm[start: start + self.batch_size]
            imgs = self.images[idx]
            if self.augment_fn is not None:
                imgs = self.augment_fn(imgs)
            yield imgs, self.labels[idx]


def _make_gpu_loader(
    dataset,
    indices: List[int],
    batch_size: int,
    device: torch.device,
    augment_fn,
) -> GpuTensorLoader:
    """Batch-load a CIFAR-10 client subset into GPU tensors (deterministic transforms only)."""
    idx_arr = np.array(indices, dtype=np.int64)
    raw = dataset.data[idx_arr]                          # (N, 32, 32, 3) uint8 numpy
    tgt = np.array(dataset.targets)[idx_arr]
    imgs = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.0)
    imgs.sub_(torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
    imgs.div_(torch.tensor([0.247,  0.243,  0.261 ]).view(1, 3, 1, 1))
    labels = torch.from_numpy(tgt).long()
    return GpuTensorLoader(
        imgs.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        batch_size,
        augment_fn=augment_fn,
    )


_DATASET_CONFIG = {
    "mnist": {
        "class": datasets.MNIST,
        "channels": 1,
        "num_classes": 10,
        "preload": True,   # deterministic transforms → safe to cache in RAM
        "train_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        "test_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    },
    "fashion-mnist": {
        "class": datasets.FashionMNIST,
        "channels": 1,
        "num_classes": 10,
        "preload": True,   # deterministic transforms → safe to cache in RAM
        "train_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.286,), (0.353,)),
            ]
        ),
        "test_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.286,), (0.353,)),
            ]
        ),
    },
    "cifar10": {
        "class": datasets.CIFAR10,
        "channels": 3,
        "num_classes": 10,
        "gpu_aug": _cifar10_gpu_aug,  # applied per-batch on GPU when preloading is enabled
        "train_transform": transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.247, 0.243, 0.261),
                ),
            ]
        ),
        "test_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.247, 0.243, 0.261),
                ),
            ]
        ),
    },
}


@dataclass
class ClientData:
    """Container for per-client dataloaders and assigned architecture."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    architecture: str


def _get_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise ValueError("Dataset does not expose targets.")
    return np.array(targets)


def get_dataset_info(name: str):
    key = name.lower()
    if key not in _DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset {name}")
    return _DATASET_CONFIG[key]


def load_datasets(name: str, root: str = "./data") -> Tuple[Dataset, Dataset]:
    info = get_dataset_info(name)
    dataset_cls = info["class"]
    train_transform = info.get("train_transform", info.get("transform"))
    test_transform = info.get("test_transform", train_transform)
    train_set = dataset_cls(root=root, train=True, download=True, transform=train_transform)
    test_set = dataset_cls(root=root, train=False, download=True, transform=test_transform)
    if info.get("preload", False):
        print(f"[data] Pre-loading {name} into RAM...", flush=True)
        train_set = MemDataset(train_set)
        test_set = MemDataset(test_set)
        print(f"[data] Pre-load done: train={len(train_set)}  test={len(test_set)}", flush=True)
    return train_set, test_set


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """Returns a list of index lists, one per client."""
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(targets))
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls_ids in class_indices:
        rng.shuffle(cls_ids)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(cls_ids)).astype(int)[:-1]
        shards = np.split(cls_ids, split_points)
        for client_id, shard in enumerate(shards):
            client_indices[client_id].extend(shard.tolist())

    for shard in client_indices:
        rng.shuffle(shard)
    return client_indices


def _generate_label_sequences(
    num_clients: int,
    sample_counts: List[int],
    num_classes: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """Generates per-client label sequences following Dirichlet-distributed label proportions."""
    rng = np.random.default_rng(seed)
    sequences: List[List[int]] = []
    concentration = np.ones(num_classes) * alpha
    for cid in range(num_clients):
        count = sample_counts[cid]
        if count <= 0:
            sequences.append([])
            continue
        probs = rng.dirichlet(concentration)
        labels = rng.choice(num_classes, size=count, p=probs)
        sequences.append(labels.tolist())
    return sequences


def _sample_from_label_sequences(
    label_sequences: List[List[int]],
    targets: np.ndarray,
    seed: int,
) -> List[List[int]]:
    """Maps label sequences to actual dataset indices via sampling with replacement."""
    rng = np.random.default_rng(seed + 1)
    class_to_indices: Dict[int, np.ndarray] = {}
    for label in np.unique(targets):
        class_to_indices[int(label)] = np.where(targets == label)[0]
    client_indices: List[List[int]] = []
    for seq in label_sequences:
        indices: List[int] = []
        for label in seq:
            pool = class_to_indices.get(int(label))
            if pool is None or pool.size == 0:
                raise ValueError(f"No samples available for label {label}.")
            sampled = rng.choice(pool)
            indices.append(int(sampled))
        client_indices.append(indices)
    return client_indices


def _split_train_val(
    indices: List[int],
    targets: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Stratified train/val split by label.

    This keeps the label proportions in train/val approximately consistent
    *within each client*, which is important under non-IID partitions.
    """
    if not indices:
        return [], []
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    idx_arr = np.asarray(indices, dtype=np.int64)
    y = targets[idx_arr].astype(np.int64)

    train_ids: List[int] = []
    val_ids: List[int] = []

    # Split within each label bucket.
    for label in np.unique(y):
        bucket = idx_arr[y == label]
        if bucket.size == 0:
            continue
        rng.shuffle(bucket)

        # Aim for roughly `test_ratio` in val, but never move all samples.
        n = int(bucket.size)
        n_val = int(round(n * test_ratio))
        n_val = max(0, min(n_val, n - 1))  # keep at least 1 in train if possible

        if n_val > 0:
            val_ids.extend(bucket[:n_val].tolist())
            train_ids.extend(bucket[n_val:].tolist())
        else:
            train_ids.extend(bucket.tolist())

    # Ensure both splits are non-empty.
    if len(val_ids) == 0:
        # Move one sample (from the largest class bucket) to val.
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        major_label = int(np.argmax(counts))
        major_bucket = idx_arr[y == major_label]
        rng.shuffle(major_bucket)
        val_ids.append(int(major_bucket[0]))
        # Remove it from train if present; otherwise remove an arbitrary one.
        if val_ids[0] in train_ids:
            train_ids.remove(val_ids[0])
        elif train_ids:
            train_ids.pop()

    if len(train_ids) == 0 and val_ids:
        # Move one back to train.
        train_ids.append(val_ids.pop())

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids



def create_client_data(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    batch_size: int,
    root: str = "./data",
    seed: int = 0,
    samples_per_client: Optional[int] = None,
    client_sample_counts: Optional[Dict[int, int]] = None,
    architecture_distribution: Optional[List[float]] = None,
    target_client_ids: Optional[List[int]] = None,
    global_test_size: int = 5000,
    exclude_test_indices: Optional[Sequence[int]] = None,
    val_ratio: float = 0.2,
    train_dataloader_workers: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[int, ClientData]:
    """
    Prepares dataloaders for the requested clients.

    When `samples_per_client` or `client_sample_counts` is provided, each client first
    draws a label distribution from Dirichlet(alpha) and then samples the requested
    number of examples with replacement from the training set according to that
    distribution. Otherwise clients follow a standard Dirichlet split without
    resampling. The optional `architecture_distribution` lets users bias the
    light/standard/strong model assignment probabilities. Passing `target_client_ids`
    constrains the returned dict to those clients while the underlying Dirichlet
    partitioning stays consistent across the whole population.
    """
    info = get_dataset_info(dataset_name)
    train_set, test_set = load_datasets(dataset_name, root)
    _gpu_aug = info.get("gpu_aug") if device is not None else None
    _can_gpu_preload = _gpu_aug is not None and hasattr(train_set, 'data')
    if _can_gpu_preload:
        print(f"[data] GPU preload enabled for {dataset_name} on {device}; "
              f"random aug will run on GPU per batch.", flush=True)
    targets = _get_targets(train_set)
    num_classes = len(np.unique(targets))
    if samples_per_client is not None and samples_per_client <= 0:
        raise ValueError("samples_per_client must be positive when provided.")
    if target_client_ids is not None:
        target_set = set(target_client_ids)
        invalid_ids = [cid for cid in target_set if cid < 0 or cid >= num_clients]
        if invalid_ids:
            raise ValueError(f"target_client_ids contains invalid client IDs: {invalid_ids}")
    else:
        target_set = None
    client_indices: List[List[int]]
    dirichlet_distributions: List[np.ndarray] = []
    if samples_per_client is not None or client_sample_counts is not None:
        sample_counts: List[int] = []
        for cid in range(num_clients):
            if client_sample_counts is not None and cid in client_sample_counts:
                count = client_sample_counts[cid]
            elif samples_per_client is not None:
                count = samples_per_client
            else:
                raise ValueError(
                    "Provide samples_per_client or client_sample_counts for every client when using Dirichlet-guided resampling."
                )
            if count <= 0:
                raise ValueError(f"Sample count for client {cid} must be positive.")
            sample_counts.append(count)
        label_sequences = _generate_label_sequences(
            num_clients=num_clients,
            sample_counts=sample_counts,
            num_classes=num_classes,
            alpha=alpha,
            seed=seed,
        )
        client_indices = _sample_from_label_sequences(label_sequences, targets, seed)
        for seq in label_sequences:
            if len(seq) == 0:
                dirichlet_distributions.append(np.zeros(num_classes))
            else:
                counts = np.bincount(np.array(seq, dtype=np.int64), minlength=num_classes)
                dirichlet_distributions.append(counts / counts.sum())
    else:
        client_indices = dirichlet_partition(targets, num_clients, alpha, seed)
        for cid_indices in client_indices:
            if not cid_indices:
                dirichlet_distributions.append(np.zeros(num_classes))
            else:
                counts = np.bincount(targets[cid_indices], minlength=num_classes)
                dirichlet_distributions.append(counts / counts.sum())
    rng = random.Random(seed)
    architectures = ["light", "standard", "strong"]
    arch_weights: Optional[List[float]] = None
    if architecture_distribution is not None:
        if len(architecture_distribution) != 3:
            raise ValueError("architecture_distribution must have three entries (light, standard, strong).")
        if any(weight < 0 for weight in architecture_distribution):
            raise ValueError("architecture_distribution weights must be non-negative.")
        if all(weight == 0 for weight in architecture_distribution):
            raise ValueError("At least one architecture weight must be positive.")
        arch_weights = list(architecture_distribution)

    if global_test_size <= 0 or global_test_size > len(test_set):
        raise ValueError("global_test_size must be within the range of the dataset size.")
    test_indices = list(range(global_test_size))
    if exclude_test_indices is not None:
        # The proxy set is drawn from this same test split, and clients distil on
        # it every round — any proxy image left in here is an image the models
        # were trained to fit, so reported accuracy stops measuring generalisation.
        # The pre-existing index file predates the hold-out convention and
        # overlaps; dropping those images is the evaluation-side repair that needs
        # no retraining (cluster.load_proxy_src_indices only warns about it).
        excluded = set(int(i) for i in exclude_test_indices)
        kept = [i for i in test_indices if i not in excluded]
        n_dropped = len(test_indices) - len(kept)
        if not kept:
            raise ValueError(
                "exclude_test_indices removed every global test sample; the proxy "
                "set covers the whole held-out region."
            )
        if n_dropped:
            print(
                f"[Eval] Dropped {n_dropped}/{len(test_indices)} global test images "
                f"that also appear in the proxy set; evaluating on {len(kept)}. "
                f"Numbers are NOT comparable with runs made before this exclusion."
            )
        test_indices = kept
    global_test_subset = Subset(test_set, test_indices)
    test_loader_shared = DataLoader(
        global_test_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    client_data: Dict[int, ClientData] = {}
    for cid in range(num_clients):
        if target_set is not None and cid not in target_set:
            continue
        indices = client_indices[cid]
        if not indices:
            continue
        if val_ratio > 0.0:
            train_ids, val_ids = _split_train_val(indices, targets, test_ratio=val_ratio, seed=seed + cid)
        else:
            train_ids, val_ids = list(indices), []
        train_subset = Subset(train_set, train_ids)
        val_subset = Subset(train_set, val_ids) if val_ids else Subset(train_set, train_ids)
        if _can_gpu_preload:
            train_loader = _make_gpu_loader(train_set, train_ids, batch_size, device, _gpu_aug)
        else:
            _in_ram = isinstance(train_set, MemDataset)
            _n_train_batches = math.ceil(max(len(train_ids), 1) / batch_size)
            _train_workers = 0 if _in_ram else (2 if _n_train_batches >= 8 else 0)
            if train_dataloader_workers is not None:
                _train_workers = train_dataloader_workers
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=_train_workers,
                pin_memory=True,
            )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        if arch_weights is not None:
            arch = rng.choices(architectures, weights=arch_weights, k=1)[0]
        else:
            arch = rng.choice(architectures)
        client_data[cid] = ClientData(train_loader, val_loader, test_loader_shared, arch)
    if dirichlet_distributions:
        dirichlet_array = np.stack(dirichlet_distributions, axis=0)
        output_npy = f"dirichlet_distributions_{dataset_name}.npy"
        output_txt = f"dirichlet_distributions_{dataset_name}.txt"
        np.save(output_npy, dirichlet_array)
        with open(output_txt, "w", encoding="utf-8") as f:
            for cid, dist in enumerate(dirichlet_array):
                # 把每个比例格式化为保留两位小数
                dist_str = "[" + ", ".join(f"{x:.2f}" for x in dist) + "]"
                line = f"Client {cid}: {dist_str}"
                print(line)
                f.write(line + "\n")
        print(
            f"[Dirichlet] Saved per-client distributions to {output_npy} and {output_txt}"
        )

    return client_data


def get_num_classes(dataset_name: str) -> int:
    return get_dataset_info(dataset_name)["num_classes"]


def get_input_channels(dataset_name: str) -> int:
    return get_dataset_info(dataset_name)["channels"]
