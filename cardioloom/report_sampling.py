"""Deterministic distributed weighted sampling for report-template tempering."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedTemplateWeightedSampler(Sampler[int]):
    """Draw a fixed-size weighted global epoch and shard it across DDP ranks."""

    def __init__(
        self,
        dataset: Sized,
        weights: torch.Tensor,
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed sampler rank/world size")
        weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if weights.ndim != 1 or len(weights) != len(dataset):
            raise ValueError("Sampling weights must be one-dimensional and match dataset")
        if not torch.isfinite(weights).all() or not bool((weights > 0).all()):
            raise ValueError("Sampling weights must be finite and strictly positive")

        self.dataset = dataset
        self.weights = weights
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.drop_last:
            self.num_samples = len(dataset) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        rank_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("Distributed weighted sampler shard size mismatch")
        return iter(rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
