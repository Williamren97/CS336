from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random contiguous blocks from a 1D dataset of token IDs.

    Returns (x, y) where y is x shifted by 1.
    """

    # Start indices in [0, len(dataset) - context_length)
    starting_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(dataset[start_idx : start_idx + context_length]) for start_idx in starting_indices]
    )
    y = torch.stack(
        [torch.from_numpy(dataset[start_idx + 1 : start_idx + context_length + 1]) for start_idx in starting_indices]
    )
    return x.to(device), y.to(device)


class Dataset:
    def __init__(self, dataset_name: str, context_length: int, batch_size: int, device: str, **kwargs):
        dataset_path = f"data/{dataset_name}"
        self.train_data = np.memmap(f"{dataset_path}/train.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.val_data = np.memmap(f"{dataset_path}/val.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        return get_batch(data, self.batch_size, self.context_length, self.device)

