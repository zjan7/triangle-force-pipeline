from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_sample_dirs(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> list[Path]:
    accepted_samples_dir = Path(accepted_samples_dir)

    if not accepted_samples_dir.exists():
        raise FileNotFoundError(f"Accepted samples directory does not exist: {accepted_samples_dir}")

    sample_dirs = [
        path for path in accepted_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    def sample_number(path: Path) -> int:
        try:
            return int(path.name.split("_")[1])
        except Exception:
            return 10**12

    sample_dirs = sorted(sample_dirs, key=sample_number)

    if len(sample_dirs) == 0:
        raise RuntimeError(f"No sample folders found in: {accepted_samples_dir}")

    return sample_dirs


def load_sample_arrays(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:

    X_path = sample_dir / "pytorch" / "X_displacements.npy"
    y_path = sample_dir / "pytorch" / "y_forces.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing X file: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing y file: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D for {sample_dir.name}, got shape {X.shape}")

    if y.ndim != 2:
        raise ValueError(f"y must be 2D for {sample_dir.name}, got shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y row mismatch for {sample_dir.name}: "
            f"X has {X.shape[0]} rows, y has {y.shape[0]} rows"
        )

    if X.shape[1] != 5:
        raise ValueError(f"X must have 5 columns for {sample_dir.name}, got shape {X.shape}")

    if y.shape[1] != 3:
        raise ValueError(f"y must have 3 columns for {sample_dir.name}, got shape {y.shape}")

    return X.astype(np.float32), y.astype(np.float32)


def build_pytorch_dataset(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "pytorch_dataset",
    output_filename: str = "triangle_force_dataset.npz",
) -> dict:
    accepted_samples_dir = Path(accepted_samples_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = find_sample_dirs(accepted_samples_dir)

    X_list = []
    y_list = []
    sample_ids = []
    n_triangles_list = []

    print()
    print("=" * 80)
    print("BUILD PYTORCH DATASET")
    print("=" * 80)
    print(f"Accepted samples directory: {accepted_samples_dir}")
    print(f"Number of sample folders:   {len(sample_dirs)}")
    print()

    for sample_dir in sample_dirs:
        X, y = load_sample_arrays(sample_dir)

        try:
            sample_id = int(sample_dir.name.split("_")[1])
        except Exception:
            sample_id = -1

        X_list.append(X)
        y_list.append(y)
        sample_ids.append(sample_id)
        n_triangles_list.append(X.shape[0])

        print(f"Loaded {sample_dir.name}: X {X.shape}, y {y.shape}")

    n_samples = len(X_list)
    max_n_triangles = max(n_triangles_list)

    X_cols = X_list[0].shape[1]
    y_cols = y_list[0].shape[1]

    X_dataset = np.zeros((n_samples, max_n_triangles, X_cols), dtype=np.float32)
    y_dataset = np.zeros((n_samples, max_n_triangles, y_cols), dtype=np.float32)
    mask = np.zeros((n_samples, max_n_triangles), dtype=np.float32)

    for sample_index, (X, y) in enumerate(zip(X_list, y_list)):
        n_triangles = X.shape[0]

        X_dataset[sample_index, :n_triangles, :] = X
        y_dataset[sample_index, :n_triangles, :] = y
        mask[sample_index, :n_triangles] = 1.0

    sample_ids_array = np.array(sample_ids, dtype=np.int32)
    n_triangles_array = np.array(n_triangles_list, dtype=np.int32)

    output_path = output_dir / output_filename

    np.savez_compressed(
        output_path,
        X=X_dataset,
        y=y_dataset,
        mask=mask,
        sample_ids=sample_ids_array,
        n_triangles=n_triangles_array,
    )

    summary = {
        "dataset_path": str(output_path),
        "n_samples": int(n_samples),
        "max_n_triangles": int(max_n_triangles),
        "X_shape": list(X_dataset.shape),
        "y_shape": list(y_dataset.shape),
        "mask_shape": list(mask.shape),
        "sample_ids": sample_ids,
        "n_triangles_min": int(np.min(n_triangles_array)),
        "n_triangles_max": int(np.max(n_triangles_array)),
        "n_triangles_mean": float(np.mean(n_triangles_array)),
        "X_columns": [
            "x_ref_px",
            "y_ref_px",
            "dx_px",
            "dy_px",
            "rotation_deg",
        ],
        "y_columns": [
            "normal_force",
            "shear_force_x",
            "shear_force_y",
        ],
    }

    summary_path = output_dir / "triangle_force_dataset_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print()
    print("=" * 80)
    print("PYTORCH DATASET SAVED")
    print("=" * 80)
    print(f"Dataset path: {output_path}")
    print(f"Summary path: {summary_path}")
    print(f"X shape:      {X_dataset.shape}")
    print(f"y shape:      {y_dataset.shape}")
    print(f"mask shape:   {mask.shape}")
    print()

    return summary


if __name__ == "__main__":
    build_pytorch_dataset()
