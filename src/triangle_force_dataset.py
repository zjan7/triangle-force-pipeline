#This code isn't used
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TriangleForceDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path = PROJECT_ROOT / "outputs" / "pytorch_dataset" / "triangle_force_dataset.npz",
        normalize: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.normalize = normalize

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {self.dataset_path}")

        data = np.load(self.dataset_path)

        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.mask = data["mask"].astype(np.float32)

        self.sample_ids = data["sample_ids"].astype(np.int32)
        self.n_triangles = data["n_triangles"].astype(np.int32)

        if self.X.ndim != 3:
            raise ValueError(f"X must have shape (samples, triangles, features), got {self.X.shape}")

        if self.y.ndim != 3:
            raise ValueError(f"y must have shape (samples, triangles, targets), got {self.y.shape}")

        if self.mask.ndim != 2:
            raise ValueError(f"mask must have shape (samples, triangles), got {self.mask.shape}")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y have different number of samples")

        if self.X.shape[1] != self.y.shape[1]:
            raise ValueError("X and y have different number of triangles")

        if self.X.shape[:2] != self.mask.shape:
            raise ValueError("mask shape does not match X/y sample and triangle dimensions")

        self.X_mean = np.zeros((1, 1, self.X.shape[2]), dtype=np.float32)
        self.X_std = np.ones((1, 1, self.X.shape[2]), dtype=np.float32)

        self.y_mean = np.zeros((1, 1, self.y.shape[2]), dtype=np.float32)
        self.y_std = np.ones((1, 1, self.y.shape[2]), dtype=np.float32)

        if self.normalize:
            self._compute_normalization()

            self.X = (self.X - self.X_mean) / self.X_std
            self.y = (self.y - self.y_mean) / self.y_std

    def _compute_normalization(self) -> None:
        valid = self.mask > 0.5

        X_valid = self.X[valid]
        y_valid = self.y[valid]

        self.X_mean = X_valid.mean(axis=0, keepdims=True).reshape(1, 1, -1).astype(np.float32)
        self.X_std = X_valid.std(axis=0, keepdims=True).reshape(1, 1, -1).astype(np.float32)

        self.y_mean = y_valid.mean(axis=0, keepdims=True).reshape(1, 1, -1).astype(np.float32)
        self.y_std = y_valid.std(axis=0, keepdims=True).reshape(1, 1, -1).astype(np.float32)

        self.X_std[self.X_std < 1e-8] = 1.0
        self.y_std[self.y_std < 1e-8] = 1.0

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "X": torch.from_numpy(self.X[index]),
            "y": torch.from_numpy(self.y[index]),
            "mask": torch.from_numpy(self.mask[index]),
            "sample_id": torch.tensor(self.sample_ids[index], dtype=torch.long),
            "n_triangles": torch.tensor(self.n_triangles[index], dtype=torch.long),
        }

    def denormalize_y(self, y_normalized: torch.Tensor) -> torch.Tensor:
        y_mean = torch.tensor(self.y_mean, dtype=y_normalized.dtype, device=y_normalized.device)
        y_std = torch.tensor(self.y_std, dtype=y_normalized.dtype, device=y_normalized.device)

        return y_normalized * y_std + y_mean
