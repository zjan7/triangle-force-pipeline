#This code isn't used
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_sample_folder(sample_id: int, accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",) -> Path:
    accepted_samples_dir = Path(accepted_samples_dir)
    sample_dir = accepted_samples_dir / f"sample_{sample_id:06d}"

    (sample_dir / "pytorch").mkdir(parents=True, exist_ok=True)
    (sample_dir / "full_matrices").mkdir(parents=True, exist_ok=True)
    (sample_dir / "check_images").mkdir(parents=True, exist_ok=True)
    (sample_dir / "raw").mkdir(parents=True, exist_ok=True)

    return sample_dir


def copy_file(src: str | Path, dst: str | Path) -> None:
    
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def write_accepted_sample(
    sample_id: int,
    reference_result: dict[str, Any],
    deformation_result: dict[str, Any],
    aruco_result: dict[str, Any],
    triangle_result: dict[str, Any],
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> Path:
    
    sample_dir = make_sample_folder(sample_id=sample_id, accepted_samples_dir=accepted_samples_dir,)

    copy_file(triangle_result["X_displacements_path"], sample_dir / "pytorch" / "X_displacements.npy",)

    copy_file(deformation_result["y_forces_path"], sample_dir / "pytorch" / "y_forces.npy",)

    copy_file(triangle_result["triangle_matrix_full_path"], sample_dir / "full_matrices" / "triangle_matrix_full.npy",)

    copy_file(deformation_result["force_matrix_path"], sample_dir / "full_matrices" / "force_matrix_full.npy",)

    copy_file(aruco_result["homography_path"], sample_dir / "full_matrices" / "homography.npy",)

    copy_file(reference_result["reference_image_path"], sample_dir / "check_images" / "reference.png",)

    copy_file(deformation_result["deformed_image_path"], sample_dir / "check_images" / "deformed.png",)

    copy_file(aruco_result["aligned_deformed_image_path"], sample_dir / "check_images" / "aligned_deformed.png",)

    copy_file(aruco_result["overlay_path"], sample_dir / "check_images" / "overlay_reference_vs_aligned.png",)

    copy_file(deformation_result["force_plot_path"], sample_dir / "check_images" / "force_plot.png",)

    # These are created by triangle_detection.py
    triangle_output_dir = Path(triangle_result["output_dir"])

    copy_file(triangle_output_dir / "displacement_arrows.png", sample_dir / "check_images" / "displacement_arrows.png",)

    copy_file(triangle_output_dir / "side_by_side_reference_vs_deformed.png", sample_dir / "check_images" / "side_by_side_reference_vs_deformed.png",)

    copy_file(triangle_output_dir / "reference_annotated_stable_ids.png", sample_dir / "check_images" / "reference_annotated_stable_ids.png",)

    copy_file(triangle_output_dir / "deformed_inherited_reference_ids.png", sample_dir / "check_images" / "deformed_inherited_reference_ids.png",)

    copy_file(deformation_result["corners_path"], sample_dir / "raw" / "deformed_corners.npy",)

    copy_file(reference_result["reference_corners_path"], sample_dir / "raw" / "reference_zero_force_corners.npy",)

    X = np.load(sample_dir / "pytorch" / "X_displacements.npy")
    y = np.load(sample_dir / "pytorch" / "y_forces.npy")

    if X.ndim != 2:
        raise RuntimeError(f"X_displacements must be 2D, got shape {X.shape}")

    if y.ndim != 2:
        raise RuntimeError(f"y_forces must be 2D, got shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise RuntimeError(
            "X and y row count mismatch. "
            f"X has {X.shape[0]} rows, y has {y.shape[0]} rows."
        )

    if X.shape[1] != 5:
        raise RuntimeError(f"X_displacements must have 5 columns, got {X.shape[1]}")

    if y.shape[1] != 3:
        raise RuntimeError(f"y_forces must have 3 columns, got {y.shape[1]}")

    metadata = {
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "accepted": True,
        "n_triangles": int(X.shape[0]),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
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
        "reference_metadata": reference_result.get("metadata", {}),
        "deformation_metadata": deformation_result.get("metadata", {}),
        "aruco_metadata": aruco_result.get("metadata", {}),
        "triangle_detection_metadata": triangle_result.get("metadata", {}),
    }

    save_json(sample_dir / "metadata.json", metadata)

    print(f"Accepted sample saved to: {sample_dir}")

    return sample_dir