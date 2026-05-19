from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_sample_dirs(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> list[Path]:
    accepted_samples_dir = Path(accepted_samples_dir)

    sample_dirs = [
        path for path in accepted_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    return sorted(sample_dirs, key=lambda p: int(p.name.split("_")[1]))


def check_one_sample(sample_dir: Path) -> dict:
    pytorch_dir = sample_dir / "pytorch"
    full_matrices_dir = sample_dir / "full_matrices"

    X_path = pytorch_dir / "X_displacements.npy"
    y_path = pytorch_dir / "y_forces.npy"

    opencv_to_generated_path = full_matrices_dir / "opencv_to_generated_index.npy"
    generated_to_opencv_path = full_matrices_dir / "generated_to_opencv_index.npy"
    mapping_distances_path = full_matrices_dir / "reference_mapping_distances_px.npy"
    y_reordered_debug_path = full_matrices_dir / "y_forces_reordered_to_opencv_order.npy"
    force_matrix_path = full_matrices_dir / "force_matrix_full.npy"

    required_paths = [
        X_path,
        y_path,
        opencv_to_generated_path,
        generated_to_opencv_path,
        mapping_distances_path,
        y_reordered_debug_path,
        force_matrix_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{sample_dir.name}: missing required file: {path}")

    X = np.load(X_path)
    y = np.load(y_path)

    opencv_to_generated = np.load(opencv_to_generated_path)
    generated_to_opencv = np.load(generated_to_opencv_path)
    mapping_distances = np.load(mapping_distances_path)

    y_reordered_debug = np.load(y_reordered_debug_path)
    force_matrix = np.load(force_matrix_path)

    generated_forces_from_matrix = force_matrix[:, 8:11]
    expected_y = generated_forces_from_matrix[opencv_to_generated]

    if X.ndim != 2:
        raise ValueError(f"{sample_dir.name}: X must be 2D, got {X.shape}")

    if y.ndim != 2:
        raise ValueError(f"{sample_dir.name}: y must be 2D, got {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{sample_dir.name}: X/y row mismatch: X={X.shape}, y={y.shape}")

    if X.shape[1] != 5:
        raise ValueError(f"{sample_dir.name}: X must have 5 columns, got {X.shape}")

    if y.shape[1] != 3:
        raise ValueError(f"{sample_dir.name}: y must have 3 columns, got {y.shape}")

    if len(opencv_to_generated) != X.shape[0]:
        raise ValueError(
            f"{sample_dir.name}: mapping length does not match X rows: "
            f"{len(opencv_to_generated)} vs {X.shape[0]}"
        )

    if len(np.unique(opencv_to_generated)) != len(opencv_to_generated):
        raise ValueError(f"{sample_dir.name}: opencv_to_generated is not one-to-one")

    if np.any(opencv_to_generated < 0):
        raise ValueError(f"{sample_dir.name}: opencv_to_generated contains negative indices")

    if np.any(generated_to_opencv < 0):
        raise ValueError(f"{sample_dir.name}: generated_to_opencv contains negative indices")

    if not np.allclose(y, y_reordered_debug, rtol=1e-6, atol=1e-12):
        raise ValueError(f"{sample_dir.name}: y_forces.npy does not match y_forces_reordered_to_opencv_order.npy")

    if not np.allclose(y, expected_y, rtol=1e-6, atol=1e-12):
        raise ValueError(f"{sample_dir.name}: y_forces.npy does not match force_matrix reordered by opencv_to_generated")

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{sample_dir.name}: X contains NaN or inf")

    if not np.all(np.isfinite(y)):
        raise ValueError(f"{sample_dir.name}: y contains NaN or inf")

    if not np.all(np.isfinite(mapping_distances)):
        raise ValueError(f"{sample_dir.name}: mapping distances contain NaN or inf")

    return {
        "sample_name": sample_dir.name,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "mean_mapping_distance_px": float(np.mean(mapping_distances)),
        "max_mapping_distance_px": float(np.max(mapping_distances)),
        "X_min": float(np.min(X)),
        "X_max": float(np.max(X)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "normal_force_min": float(np.min(y[:, 0])),
        "normal_force_max": float(np.max(y[:, 0])),
        "shear_x_min": float(np.min(y[:, 1])),
        "shear_x_max": float(np.max(y[:, 1])),
        "shear_y_min": float(np.min(y[:, 2])),
        "shear_y_max": float(np.max(y[:, 2])),
    }


def check_npz_dataset(
    npz_path: str | Path = PROJECT_ROOT / "outputs" / "pytorch_dataset" / "triangle_force_dataset.npz",
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> dict:
    npz_path = Path(npz_path)
    accepted_samples_dir = Path(accepted_samples_dir)

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz dataset: {npz_path}")

    data = np.load(npz_path)

    required_arrays = ["X", "y", "mask", "sample_ids", "n_triangles"]

    for name in required_arrays:
        if name not in data.files:
            raise ValueError(f"NPZ dataset missing array: {name}")

    X_all = data["X"]
    y_all = data["y"]
    mask = data["mask"]
    sample_ids = data["sample_ids"]
    n_triangles = data["n_triangles"]

    if X_all.ndim != 3:
        raise ValueError(f"NPZ X must be 3D, got {X_all.shape}")

    if y_all.ndim != 3:
        raise ValueError(f"NPZ y must be 3D, got {y_all.shape}")

    if mask.ndim != 2:
        raise ValueError(f"NPZ mask must be 2D, got {mask.shape}")

    if X_all.shape[0] != y_all.shape[0]:
        raise ValueError("NPZ X/y sample count mismatch")

    if X_all.shape[1] != y_all.shape[1]:
        raise ValueError("NPZ X/y triangle count mismatch")

    if X_all.shape[:2] != mask.shape:
        raise ValueError("NPZ mask shape does not match X/y")

    if X_all.shape[2] != 5:
        raise ValueError(f"NPZ X must have 5 features, got {X_all.shape}")

    if y_all.shape[2] != 3:
        raise ValueError(f"NPZ y must have 3 targets, got {y_all.shape}")

    if not np.all(np.isfinite(X_all)):
        raise ValueError("NPZ X contains NaN or inf")

    if not np.all(np.isfinite(y_all)):
        raise ValueError("NPZ y contains NaN or inf")

    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError("NPZ mask should contain only 0 and 1")

    # Check that NPZ arrays match individual sample files.
    for sample_index, sample_id in enumerate(sample_ids):
        sample_dir = accepted_samples_dir / f"sample_{int(sample_id):06d}"

        X_path = sample_dir / "pytorch" / "X_displacements.npy"
        y_path = sample_dir / "pytorch" / "y_forces.npy"

        if not X_path.exists():
            raise FileNotFoundError(f"Missing individual X file: {X_path}")

        if not y_path.exists():
            raise FileNotFoundError(f"Missing individual y file: {y_path}")

        X_single = np.load(X_path)
        y_single = np.load(y_path)

        n = X_single.shape[0]

        if int(n_triangles[sample_index]) != n:
            raise ValueError(
                f"NPZ n_triangles mismatch for sample {sample_id}: "
                f"{n_triangles[sample_index]} vs {n}"
            )

        if not np.allclose(X_all[sample_index, :n, :], X_single, rtol=1e-6, atol=1e-12):
            raise ValueError(f"NPZ X does not match individual sample {sample_id}")

        if not np.allclose(y_all[sample_index, :n, :], y_single, rtol=1e-6, atol=1e-12):
            raise ValueError(f"NPZ y does not match individual sample {sample_id}")

        if not np.all(mask[sample_index, :n] == 1.0):
            raise ValueError(f"NPZ mask valid region wrong for sample {sample_id}")

        if not np.all(mask[sample_index, n:] == 0.0):
            raise ValueError(f"NPZ mask padding region wrong for sample {sample_id}")

    return {
        "npz_path": str(npz_path),
        "files": list(data.files),
        "X_shape": list(X_all.shape),
        "y_shape": list(y_all.shape),
        "mask_shape": list(mask.shape),
        "sample_ids": sample_ids.astype(int).tolist(),
        "n_triangles": n_triangles.astype(int).tolist(),
        "X_min": float(np.min(X_all)),
        "X_max": float(np.max(X_all)),
        "y_min": float(np.min(y_all)),
        "y_max": float(np.max(y_all)),
    }


def check_dataset(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
    npz_path: str | Path = PROJECT_ROOT / "outputs" / "pytorch_dataset" / "triangle_force_dataset.npz",
    output_path: str | Path = PROJECT_ROOT / "outputs" / "dataset_integrity_check.json",
) -> dict:
    sample_dirs = find_sample_dirs(accepted_samples_dir)

    if len(sample_dirs) == 0:
        raise RuntimeError("No accepted samples found.")

    print()
    print("=" * 80)
    print("DATASET INTEGRITY CHECK")
    print("=" * 80)
    print(f"Accepted samples found: {len(sample_dirs)}")
    print()

    sample_summaries = []

    for sample_dir in sample_dirs:
        summary = check_one_sample(sample_dir)
        sample_summaries.append(summary)

        print(
            f"{summary['sample_name']}: "
            f"X {summary['X_shape']}, y {summary['y_shape']}, "
            f"mapping mean/max = "
            f"{summary['mean_mapping_distance_px']:.3f}/"
            f"{summary['max_mapping_distance_px']:.3f} px"
        )

    print()
    print("Checking compressed NPZ dataset...")
    npz_summary = check_npz_dataset(
        npz_path=npz_path,
        accepted_samples_dir=accepted_samples_dir,
    )

    mean_mapping_distances = [s["mean_mapping_distance_px"] for s in sample_summaries]
    max_mapping_distances = [s["max_mapping_distance_px"] for s in sample_summaries]

    full_summary = {
        "status": "PASS",
        "n_samples_checked": len(sample_summaries),
        "mean_mapping_distance_px_over_samples": float(np.mean(mean_mapping_distances)),
        "max_mapping_distance_px_over_samples": float(np.max(max_mapping_distances)),
        "samples": sample_summaries,
        "npz_dataset": npz_summary,
    }

    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(full_summary, file, indent=4)

    print()
    print("=" * 80)
    print("CHECK PASSED")
    print("=" * 80)
    print(f"Samples checked:                 {len(sample_summaries)}")
    print(f"Mean mapping distance over all:   {full_summary['mean_mapping_distance_px_over_samples']:.3f} px")
    print(f"Max mapping distance over all:    {full_summary['max_mapping_distance_px_over_samples']:.3f} px")
    print(f"NPZ X shape:                      {npz_summary['X_shape']}")
    print(f"NPZ y shape:                      {npz_summary['y_shape']}")
    print(f"NPZ mask shape:                   {npz_summary['mask_shape']}")
    print(f"Saved report:                     {output_path}")
    print()

    return full_summary


if __name__ == "__main__":
    check_dataset()
