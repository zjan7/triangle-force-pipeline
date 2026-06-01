from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_real_sample_folder(
    sample_id: int,
    real_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "real_inference_samples",
) -> dict[str, Path]:
    real_samples_dir = Path(real_samples_dir)
    sample_dir = real_samples_dir / f"sample_{sample_id:06d}"

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


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, indent=4)


def get_path_from_result(result: dict[str, Any], keys: list[str]) -> Path | None:
    for key in keys:
        if key in result and result[key] is not None:
            path = Path(result[key])

            if path.exists():
                return path

    return None


def write_real_inference_sample(
    sample_id: int,
    real_pipeline_result: dict[str, Any],
    real_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "real_inference_samples",
) -> Path:
    folders = make_real_sample_folder(
        sample_id=sample_id,
        real_samples_dir=real_samples_dir,
    )

    sample_dir = folders["sample_dir"]
    pytorch_dir = folders["pytorch_dir"]
    full_matrices_dir = folders["full_matrices_dir"]
    check_images_dir = folders["check_images_dir"]
    raw_dir = folders["raw_dir"]

    X_displacements_path = get_path_from_result(
        real_pipeline_result,
        ["X_displacements_path"],
    )

    triangle_matrix_full_path = get_path_from_result(
        real_pipeline_result,
        ["triangle_matrix_full_path"],
    )

    checks_path = get_path_from_result(
        real_pipeline_result,
        ["checks_path"],
    )

    if X_displacements_path is None:
        raise FileNotFoundError("Could not find X_displacements_path in real_pipeline_result.")

    if triangle_matrix_full_path is None:
        raise FileNotFoundError("Could not find triangle_matrix_full_path in real_pipeline_result.")

    X_output_path = pytorch_dir / "X_displacements.npy"
    triangle_matrix_output_path = full_matrices_dir / "triangle_matrix_full.npy"

    copy_file(X_displacements_path, X_output_path)
    copy_file(triangle_matrix_full_path, triangle_matrix_output_path)

    X = np.load(X_output_path)

    if X.ndim != 2:
        raise ValueError(f"X_displacements.npy must be 2D, got shape {X.shape}")

    if X.shape[1] != 5:
        raise ValueError(f"X_displacements.npy must have 5 columns, got shape {X.shape}")

    aruco_result = real_pipeline_result.get("aruco_result", {})
    copy_file(get_path_from_result(aruco_result, ["homography_path"]), full_matrices_dir / "homography.npy")
    copy_file(get_path_from_result(aruco_result, ["overlay_path"]), check_images_dir / "overlay_reference_vs_aligned.png")
    copy_file(get_path_from_result(aruco_result, ["reference_detected_path"]), check_images_dir / "aruco_reference_detected.png")
    copy_file(get_path_from_result(aruco_result, ["deformed_detected_path"]), check_images_dir / "aruco_deformed_detected.png")
    copy_file(get_path_from_result(aruco_result, ["aligned_deformed_image_path"]), check_images_dir / "aligned_deformed.png")

    output_dir = real_pipeline_result.get("output_dir", None)

    if output_dir is not None:
        output_dir = Path(output_dir)

        for image_name in [
            "reference_annotated_stable_ids.png",
            "deformed_inherited_reference_ids.png",
            "deformed_detected_no_ids.png",
            "reference_orange_binary_mask.png",
            "deformed_orange_binary_mask.png",
            "displacement_arrows.png",
            "side_by_side_reference_vs_deformed.png",
        ]:
            copy_file(output_dir / image_name, check_images_dir / image_name)

        copy_file(output_dir / "triangle_matches.csv", full_matrices_dir / "triangle_matches.csv")
        copy_file(output_dir / "triangle_matches.json", full_matrices_dir / "triangle_matches.json")
        copy_file(output_dir / "visual_check_report.json", full_matrices_dir / "visual_check_report.json")

    copy_file(checks_path, full_matrices_dir / "visual_check_report.json")

    checks = real_pipeline_result.get("checks", {})

    metadata = {
        "sample_id": int(sample_id),
        "sample_dir": str(sample_dir),
        "X_displacements_path": str(X_output_path),
        "X_shape": list(X.shape),
        "X_columns": [
            "x_ref_px",
            "y_ref_px",
            "dx_px",
            "dy_px",
            "rotation_deg",
        ],
        "ready_for_neural_network": bool(real_pipeline_result.get("ready_for_neural_network", False)),
        "important_note": (
            "This is a real-camera inference sample. It contains X_displacements.npy only. "
            "No y_forces.npy is saved because real camera images do not contain known ground-truth forces."
        ),
        "checks": checks,
    }

    save_json(sample_dir / "metadata.json", metadata)

    print()
    print(f"Saved real inference sample: {sample_dir}")
    print(f"X shape: {X.shape}")
    print(f"Ready for neural network: {metadata['ready_for_neural_network']}")
    print()

    return sample_dir