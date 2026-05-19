from __future__ import annotations

import json
from pathlib import Path
from itertools import permutations

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def flatten_corner_array(corners: np.ndarray) -> np.ndarray:
    """
    Accept:
        (N, 3, 3)
    or:
        (n_x, n_y, 3, 3)

    Return:
        (N, 3, 3)
    """
    corners = np.asarray(corners, dtype=float)

    if corners.ndim == 3:
        return corners

    if corners.ndim == 4:
        n_x, n_y, n_corners, n_coords = corners.shape

        if n_corners != 3:
            raise ValueError(f"Expected 3 corners per triangle, got shape {corners.shape}")

        return corners.reshape(n_x * n_y, n_corners, n_coords)

    raise ValueError(
        f"Corner array should have shape (N, 3, 3) or (n_x, n_y, 3, 3), got {corners.shape}"
    )


def rotation_from_corners_clockwise_0_60(
    reference_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
) -> float:
    
    ref = np.asarray(reference_vertices, dtype=float)[:, :2]
    deformed = np.asarray(deformed_vertices, dtype=float)[:, :2]

    ref_center = ref.mean(axis=0)
    def_center = deformed.mean(axis=0)

    ref_centered = ref - ref_center
    def_centered = deformed - def_center

    # Kabsch-like 2D rotation estimate using corresponding corners.
    numerator = np.sum(
        ref_centered[:, 0] * def_centered[:, 1]
        - ref_centered[:, 1] * def_centered[:, 0]
    )

    denominator = np.sum(
        ref_centered[:, 0] * def_centered[:, 0]
        + ref_centered[:, 1] * def_centered[:, 1]
    )

    raw_angle_ccw = np.degrees(np.arctan2(numerator, denominator))
    raw_angle_ccw = wrap_angle_deg(raw_angle_ccw)

    # Convert physical counterclockwise angle to clockwise-positive angle.
    rotation_clockwise = -raw_angle_ccw

    # Bring into [0, 360).
    rotation_clockwise = rotation_clockwise % 360.0

    # The physical model only allows clockwise rotations from 0 to 60 degrees.
    # If numerical noise gives something close to 360, treat it as 0.
    if rotation_clockwise > 300.0:
        rotation_clockwise = 0.0

    # Safety clamp.
    rotation_clockwise = float(np.clip(rotation_clockwise, 0.0, 60.0))

    return rotation_clockwise
def compute_generated_rotations(
    reference_corners_path: str | Path,
    deformed_corners_path: str | Path,
) -> np.ndarray:
    reference_corners = np.load(reference_corners_path, allow_pickle=True)
    deformed_corners = np.load(deformed_corners_path, allow_pickle=True)

    reference_corners = flatten_corner_array(reference_corners)
    deformed_corners = flatten_corner_array(deformed_corners)

    if reference_corners.shape != deformed_corners.shape:
        raise ValueError(
            f"Reference and deformed corner shapes differ: "
            f"{reference_corners.shape} vs {deformed_corners.shape}"
        )

    rotations = np.zeros(reference_corners.shape[0], dtype=np.float32)

    for index in range(reference_corners.shape[0]):
        rotations[index] = rotation_from_corners_clockwise_0_60(
            reference_corners[index],
            deformed_corners[index],
        )

    return rotations


def update_one_sample_rotation(sample_dir: str | Path) -> dict:
    sample_dir = Path(sample_dir)

    raw_dir = sample_dir / "raw"
    pytorch_dir = sample_dir / "pytorch"
    full_matrices_dir = sample_dir / "full_matrices"

    reference_corners_path = raw_dir / "reference_zero_force_corners.npy"
    deformed_corners_path = raw_dir / "deformed_corners.npy"
    mapping_path = full_matrices_dir / "opencv_to_generated_index.npy"

    X_path = pytorch_dir / "X_displacements.npy"
    X_backup_path = pytorch_dir / "X_displacements_before_generated_rotation.npy"

    triangle_matrix_path = full_matrices_dir / "triangle_matrix_full.npy"
    triangle_matrix_backup_path = full_matrices_dir / "triangle_matrix_full_before_generated_rotation.npy"

    if not reference_corners_path.exists():
        raise FileNotFoundError(f"Missing: {reference_corners_path}")

    if not deformed_corners_path.exists():
        raise FileNotFoundError(f"Missing: {deformed_corners_path}")

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing: {mapping_path}. "
            "Run the force alignment step first so OpenCV-to-generated mapping exists."
        )

    if not X_path.exists():
        raise FileNotFoundError(f"Missing: {X_path}")

    if not triangle_matrix_path.exists():
        raise FileNotFoundError(f"Missing: {triangle_matrix_path}")

    generated_rotations = compute_generated_rotations(
        reference_corners_path=reference_corners_path,
        deformed_corners_path=deformed_corners_path,
    )

    opencv_to_generated = np.load(mapping_path).astype(int)

    X = np.load(X_path)
    triangle_matrix = np.load(triangle_matrix_path)

    if X.shape[0] != len(opencv_to_generated):
        raise ValueError(
            f"X rows and mapping length differ: X={X.shape}, mapping={len(opencv_to_generated)}"
        )

    if triangle_matrix.shape[0] != len(opencv_to_generated):
        raise ValueError(
            f"triangle_matrix rows and mapping length differ: "
            f"triangle_matrix={triangle_matrix.shape}, mapping={len(opencv_to_generated)}"
        )

    if not X_backup_path.exists():
        np.save(X_backup_path, X)

    if not triangle_matrix_backup_path.exists():
        np.save(triangle_matrix_backup_path, triangle_matrix)

    rotations_opencv_order = generated_rotations[opencv_to_generated]

    # Replace only the rotation column.
    # X columns:
    # 0 x_ref_px
    # 1 y_ref_px
    # 2 dx_px
    # 3 dy_px
    # 4 rotation_deg
    X[:, 4] = rotations_opencv_order

    # triangle_matrix_full columns:
    # 10 angle_ref_deg
    # 11 angle_def_deg
    # 12 rotation_change_deg
    #
    # We only replace the rotation_change_deg column.
    triangle_matrix[:, 12] = rotations_opencv_order

    np.save(X_path, X)
    np.save(triangle_matrix_path, triangle_matrix)

    np.save(full_matrices_dir / "generated_rotation_opencv_order.npy", rotations_opencv_order)
    np.save(full_matrices_dir / "generated_rotation_generated_order.npy", generated_rotations)

    metadata = {
        "sample_dir": str(sample_dir),
        "rotation_source": "generated_reference_and_deformed_corner_arrays",
        "rotation_order": "opencv_order",
        "rotation_column_replaced": True,
        "X_rotation_column": 4,
        "triangle_matrix_rotation_column": 12,
        "rotation_min_deg": float(np.min(rotations_opencv_order)),
        "rotation_max_deg": float(np.max(rotations_opencv_order)),
        "rotation_mean_deg": float(np.mean(rotations_opencv_order)),
        "rotation_nonzero_count_gt_1e_minus_6": int(np.sum(np.abs(rotations_opencv_order) > 1e-6)),
        "note": (
            "dx_px and dy_px remain OpenCV-detected. Only rotation_deg was replaced "
            "by generated-corner rotation, reordered to OpenCV triangle order."
        ),
    }

    metadata_path = full_matrices_dir / "generated_rotation_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print()
    print(f"Updated rotation for {sample_dir.name}")
    print(f"rotation min/max: {metadata['rotation_min_deg']:.6f} / {metadata['rotation_max_deg']:.6f} deg")
    print(f"nonzero rotations: {metadata['rotation_nonzero_count_gt_1e_minus_6']}")
    print()

    return metadata


def update_all_samples(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> list[dict]:
    accepted_samples_dir = Path(accepted_samples_dir)

    sample_dirs = [
        path for path in accepted_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    sample_dirs = sorted(sample_dirs, key=lambda path: int(path.name.split("_")[1]))

    if len(sample_dirs) == 0:
        raise RuntimeError(f"No sample folders found in {accepted_samples_dir}")

    results = []

    print()
    print("=" * 80)
    print("UPDATE ROTATION FROM GENERATED CORNERS")
    print("=" * 80)
    print(f"Samples found: {len(sample_dirs)}")

    for sample_dir in sample_dirs:
        results.append(update_one_sample_rotation(sample_dir))

    summary_path = PROJECT_ROOT / "outputs" / "generated_rotation_update_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print()
    print("=" * 80)
    print("ROTATION UPDATE FINISHED")
    print("=" * 80)
    print(f"Saved summary: {summary_path}")
    print()

    return results


if __name__ == "__main__":
    update_all_samples()
