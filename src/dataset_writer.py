#(pipe)This code creates the dataset with as final code the force_alignment.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.force_alignment import align_forces_to_opencv_order


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_sample_folder(sample_id: int, accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",) -> dict[str, Path]:
    accepted_samples_dir = Path(accepted_samples_dir)

    sample_dir = accepted_samples_dir / f"sample_{sample_id:06d}"

    folders = {
        "sample_dir": sample_dir,
        "pytorch_dir": sample_dir / "pytorch",
        "full_matrices_dir": sample_dir / "full_matrices",
        "check_images_dir": sample_dir / "check_images",
        "raw_dir": sample_dir / "raw",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def copy_file(source: str | Path | None, destination: str | Path) -> Path | None:
    if source is None:
        return None

    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    return destination


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def make_json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, dict):
            return {str(k): make_json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [make_json_safe(v) for v in value]

        if isinstance(value, tuple):
            return [make_json_safe(v) for v in value]

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        return value

    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, indent=4)


def get_path_from_result(result: dict[str, Any], keys: list[str]) -> Path | None:
    for key in keys:
        if key in result and result[key] is not None:
            path = Path(result[key])

            if path.exists():
                return path

    return None


def write_accepted_sample(
    sample_id: int,
    reference_result: dict[str, Any],
    deformation_result: dict[str, Any],
    aruco_result: dict[str, Any],
    triangle_result: dict[str, Any],
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> Path:
    
    folders = make_sample_folder(sample_id=sample_id, accepted_samples_dir=accepted_samples_dir,)

    sample_dir = folders["sample_dir"]
    pytorch_dir = folders["pytorch_dir"]
    full_matrices_dir = folders["full_matrices_dir"]
    check_images_dir = folders["check_images_dir"]
    raw_dir = folders["raw_dir"]
    X_displacements_path = get_path_from_result(triangle_result, ["X_displacements_path", "x_displacements_path"],)

    triangle_matrix_full_path = get_path_from_result(triangle_result, ["triangle_matrix_full_path", "triangle_matrix_path"],)

    reference_corners_path = get_path_from_result(reference_result, ["reference_corners_path", "reference_zero_force_corners_path", "corners_path"],)

    y_forces_path = get_path_from_result(deformation_result, ["y_forces_path", "forces_path"],)

    force_matrix_full_path = get_path_from_result(deformation_result, ["force_matrix_full_path", "force_matrix_path"],)

    if X_displacements_path is None:
        raise FileNotFoundError("Could not find X_displacements path in triangle_result.")

    if triangle_matrix_full_path is None:
        raise FileNotFoundError("Could not find triangle_matrix_full path in triangle_result.")

    if reference_corners_path is None:
        raise FileNotFoundError("Could not find reference corners path in reference_result.")

    if y_forces_path is None and force_matrix_full_path is None:
        raise FileNotFoundError("Could not find generated forces path in deformation_result.")

    X_output_path = pytorch_dir / "X_displacements.npy"
    copy_file(X_displacements_path, X_output_path)

    alignment_result = align_forces_to_opencv_order(
        reference_corners_path=reference_corners_path,
        triangle_matrix_full_path=triangle_matrix_full_path,
        y_forces_path=y_forces_path,
        force_matrix_full_path=force_matrix_full_path,
        output_dir=full_matrices_dir,
        mean_distance_threshold_px=1.0,
        max_distance_threshold_px=2.0,
    )

    y_forces_reordered = alignment_result["y_forces_reordered"]

    y_output_path = pytorch_dir / "y_forces.npy"
    np.save(y_output_path, y_forces_reordered)

    # Keep original generated-order forces for debugging.
    if y_forces_path is not None:
        copy_file(y_forces_path, full_matrices_dir / "y_forces_generated_order.npy")

    X = np.load(X_output_path)
    y = np.load(y_output_path)

    if X.ndim != 2:
        raise ValueError(f"X_displacements.npy must be 2D, got shape {X.shape}")

    if y.ndim != 2:
        raise ValueError(f"y_forces.npy must be 2D, got shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y row counts do not match after force alignment: "
            f"X={X.shape}, y={y.shape}"
        )

    if X.shape[1] != 5:
        raise ValueError(f"X_displacements.npy must have 5 columns, got shape {X.shape}")

    if y.shape[1] != 3:
        raise ValueError(f"y_forces.npy must have 3 columns, got shape {y.shape}")
    copy_file(triangle_matrix_full_path, full_matrices_dir / "triangle_matrix_full.npy")
    copy_file(force_matrix_full_path, full_matrices_dir / "force_matrix_full.npy")

    homography_path = get_path_from_result(aruco_result, ["homography_path"],)
    copy_file(homography_path, full_matrices_dir / "homography.npy")

    copy_file(reference_corners_path, raw_dir / "reference_zero_force_corners.npy")

    deformed_corners_path = get_path_from_result(deformation_result, ["deformed_corners_path", "corners_path"],)
    copy_file(deformed_corners_path, raw_dir / "deformed_corners.npy")

    reference_data_path = get_path_from_result(reference_result, ["reference_data_path", "reference_zero_force_data_path"],)
    copy_file(reference_data_path, raw_dir / "reference_zero_force_data.npy")
    copy_file(get_path_from_result(reference_result, ["reference_image_path", "image_path"]), check_images_dir / "reference_zero_force_with_aruco.png",)

    copy_file(get_path_from_result(deformation_result, ["deformed_image_path", "image_path"]), check_images_dir / "deformed_with_aruco.png",)

    copy_file(get_path_from_result(aruco_result, ["aligned_image_path", "aligned_deformed_path"]), check_images_dir / "aligned_deformed.png",)

    copy_file(get_path_from_result(aruco_result, ["overlay_path"]), check_images_dir / "overlay_reference_vs_aligned.png",)

    copy_file(get_path_from_result(deformation_result, ["force_plot_path"]), check_images_dir / "force_plot.png",)

    triangle_output_dir = triangle_result.get("output_dir", None)

    if triangle_output_dir is not None:
        triangle_output_dir = Path(triangle_output_dir)

        for image_name in [
            "reference_annotated_stable_ids.png",
            "deformed_inherited_reference_ids.png",
            "deformed_detected_no_ids.png",
            "reference_binary_mask.png",
            "deformed_binary_mask.png",
            "displacement_arrows.png",
            "side_by_side_reference_vs_deformed.png",
        ]:
            copy_file(triangle_output_dir / image_name, check_images_dir / image_name,)

        copy_file(triangle_output_dir / "triangle_matrix.csv", full_matrices_dir / "triangle_matrix.csv",)

        copy_file(triangle_output_dir / "triangle_analysis.json", full_matrices_dir / "triangle_analysis.json",)
    metadata = {
        "sample_id": int(sample_id),
        "sample_dir": str(sample_dir),
        "X_displacements_path": str(X_output_path),
        "y_forces_path": str(y_output_path),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "force_alignment": alignment_result["metadata"],
        "important_note": (
            "y_forces.npy has been reordered to match X_displacements.npy/OpenCV triangle order "
            "using reference-centroid position matching."
        ),
    }

    save_json(sample_dir / "metadata.json", metadata)

    print()
    print(f"Saved accepted sample: {sample_dir}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(
        "Force alignment mean/max distance: "
        f"{alignment_result['metadata']['mean_reference_mapping_distance_px']:.3f} px / "
        f"{alignment_result['metadata']['max_reference_mapping_distance_px']:.3f} px"
    )
    print()

    return sample_dir
