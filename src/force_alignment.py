#(pipe)This is the final code of the loop it makes sure the opencv ids match the generated ids so the Y value in the PyTorch dataset is valid
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def flatten_corner_array(corners: np.ndarray) -> np.ndarray:
   
    corners = np.asarray(corners, dtype=float)

    if corners.ndim == 3:
        return corners

    if corners.ndim == 4:
        n_x, n_y, n_corners, n_coords = corners.shape

        if n_corners != 3:
            raise ValueError(f"Expected 3 triangle corners, got shape {corners.shape}")

        return corners.reshape(n_x * n_y, n_corners, n_coords)

    raise ValueError(
        f"Corner array must have shape (N, 3, 2/3) or (n_x, n_y, 3, 2/3), got {corners.shape}"
    )


def initial_physical_to_pixel_guess(
    physical_points: np.ndarray,
    pixel_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    x_phys = physical_points[:, 0]
    y_phys = physical_points[:, 1]

    x_pix = pixel_points[:, 0]
    y_pix = pixel_points[:, 1]

    sx = (x_pix.max() - x_pix.min()) / max(x_phys.max() - x_phys.min(), 1e-12)
    sy = (y_pix.min() - y_pix.max()) / max(y_phys.max() - y_phys.min(), 1e-12)

    bx = x_pix.min() - sx * x_phys.min()
    by = y_pix.max() - sy * y_phys.min()

    A = np.array(
        [
            [sx, 0.0],
            [0.0, sy],
        ],
        dtype=float,
    )

    b = np.array([bx, by], dtype=float)

    return A, b


def apply_affine(points: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    return points @ A + b


def fit_affine(
    physical_points: np.ndarray,
    pixel_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones((physical_points.shape[0], 1), dtype=float)
    design = np.hstack([physical_points, ones])

    coeffs, _, _, _ = np.linalg.lstsq(design, pixel_points, rcond=None)

    A = coeffs[:2, :]
    b = coeffs[2, :]

    return A, b


def one_to_one_nearest_mapping(
    generated_points_px: np.ndarray,
    opencv_points_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_generated = generated_points_px.shape[0]
    n_opencv = opencv_points_px.shape[0]

    diff = opencv_points_px[:, None, :] - generated_points_px[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=2)

    try:
        from scipy.optimize import linear_sum_assignment

        opencv_indices, generated_indices = linear_sum_assignment(distance_matrix)

        opencv_to_generated = np.full(n_opencv, -1, dtype=int)
        distances = np.full(n_opencv, np.nan, dtype=float)

        for opencv_index, generated_index in zip(opencv_indices, generated_indices):
            opencv_to_generated[opencv_index] = generated_index
            distances[opencv_index] = distance_matrix[opencv_index, generated_index]

        return opencv_to_generated, distances

    except Exception:
        # Greedy fallback if scipy is unavailable.
        pairs = []

        for opencv_index in range(n_opencv):
            for generated_index in range(n_generated):
                pairs.append(
                    (
                        distance_matrix[opencv_index, generated_index],
                        opencv_index,
                        generated_index,
                    )
                )

        pairs = sorted(pairs, key=lambda item: item[0])

        opencv_to_generated = np.full(n_opencv, -1, dtype=int)
        distances = np.full(n_opencv, np.nan, dtype=float)

        used_opencv = set()
        used_generated = set()

        for distance, opencv_index, generated_index in pairs:
            if opencv_index in used_opencv or generated_index in used_generated:
                continue

            opencv_to_generated[opencv_index] = generated_index
            distances[opencv_index] = distance

            used_opencv.add(opencv_index)
            used_generated.add(generated_index)

            if len(used_opencv) == min(n_opencv, n_generated):
                break

        return opencv_to_generated, distances


def load_generated_forces(
    y_forces_path: str | Path | None = None,
    force_matrix_full_path: str | Path | None = None,
) -> np.ndarray:
   
    if force_matrix_full_path is not None and Path(force_matrix_full_path).exists():
        force_matrix = np.load(force_matrix_full_path, allow_pickle=True)
        force_matrix = np.asarray(force_matrix, dtype=float)

        if force_matrix.ndim != 2 or force_matrix.shape[1] < 11:
            raise ValueError(f"force_matrix_full.npy should have at least 11 columns, got {force_matrix.shape}")

        return force_matrix[:, 8:11].astype(np.float32)

    if y_forces_path is not None and Path(y_forces_path).exists():
        y = np.load(y_forces_path, allow_pickle=True)
        y = np.asarray(y, dtype=float)

        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError(f"y_forces.npy should have shape (N, 3), got {y.shape}")

        return y.astype(np.float32)

    raise FileNotFoundError("Could not load generated forces from force_matrix_full_path or y_forces_path.")


def align_forces_to_opencv_order(
    reference_corners_path: str | Path,
    triangle_matrix_full_path: str | Path,
    y_forces_path: str | Path | None = None,
    force_matrix_full_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    mean_distance_threshold_px: float = 1.0,
    max_distance_threshold_px: float = 2.0,
    n_refinement_steps: int = 3,
) -> dict:
    reference_corners_path = Path(reference_corners_path)
    triangle_matrix_full_path = Path(triangle_matrix_full_path)

    reference_corners = np.load(reference_corners_path, allow_pickle=True)
    reference_corners = flatten_corner_array(reference_corners)

    triangle_matrix = np.load(triangle_matrix_full_path, allow_pickle=True)
    triangle_matrix = np.asarray(triangle_matrix, dtype=float)

    generated_forces = load_generated_forces(y_forces_path=y_forces_path, force_matrix_full_path=force_matrix_full_path,)

    generated_ref_centroids_physical = reference_corners[:, :, :2].mean(axis=1)

    # triangle_matrix_full columns:
    # 3 cx_ref_px
    # 4 cy_ref_px
    opencv_ref_centroids_px = triangle_matrix[:, 3:5]

    n_generated = generated_ref_centroids_physical.shape[0]
    n_opencv = opencv_ref_centroids_px.shape[0]

    if generated_forces.shape[0] != n_generated:
        raise ValueError(
            f"Generated force rows do not match generated triangles: "
            f"forces={generated_forces.shape[0]}, triangles={n_generated}"
        )

    if n_generated != n_opencv:
        raise ValueError(
            f"Generated and OpenCV triangle counts differ: "
            f"generated={n_generated}, opencv={n_opencv}"
        )

    # Initial transform.
    A, b = initial_physical_to_pixel_guess(physical_points=generated_ref_centroids_physical, pixel_points=opencv_ref_centroids_px,)

    # Refine transform and mapping.
    for _ in range(n_refinement_steps):
        generated_ref_centroids_px_est = apply_affine(generated_ref_centroids_physical, A, b,)

        opencv_to_generated, distances = one_to_one_nearest_mapping(generated_points_px=generated_ref_centroids_px_est, opencv_points_px=opencv_ref_centroids_px,)

        valid_opencv_rows = np.where(opencv_to_generated >= 0)[0]
        matched_generated_indices = opencv_to_generated[valid_opencv_rows]

        A, b = fit_affine(physical_points=generated_ref_centroids_physical[matched_generated_indices], pixel_points=opencv_ref_centroids_px[valid_opencv_rows],)

    # Final mapping.
    generated_ref_centroids_px_final = apply_affine(generated_ref_centroids_physical, A, b,)

    opencv_to_generated, distances = one_to_one_nearest_mapping(generated_points_px=generated_ref_centroids_px_final, opencv_points_px=opencv_ref_centroids_px,)

    if np.any(opencv_to_generated < 0):
        raise RuntimeError("Not all OpenCV triangles could be matched to generated triangles.")

    unique_generated = np.unique(opencv_to_generated)

    if len(unique_generated) != n_generated:
        raise RuntimeError(
            f"Mapping is not one-to-one: unique generated matches={len(unique_generated)}, "
            f"expected={n_generated}"
        )

    mean_distance = float(np.mean(distances))
    max_distance = float(np.max(distances))

    if mean_distance > mean_distance_threshold_px:
        raise RuntimeError(
            f"Mean reference centroid mapping distance too large: "
            f"{mean_distance:.3f} px > {mean_distance_threshold_px:.3f} px"
        )

    if max_distance > max_distance_threshold_px:
        raise RuntimeError(
            f"Max reference centroid mapping distance too large: "
            f"{max_distance:.3f} px > {max_distance_threshold_px:.3f} px"
        )

    # This is the important line:
    # row k in y_forces_reordered now matches row k in OpenCV X_displacements.
    y_forces_reordered = generated_forces[opencv_to_generated].astype(np.float32)

    generated_to_opencv = np.full(n_generated, -1, dtype=int)

    for opencv_index, generated_index in enumerate(opencv_to_generated):
        generated_to_opencv[generated_index] = opencv_index

    metadata = {
        "n_generated_triangles": int(n_generated),
        "n_opencv_triangles": int(n_opencv),
        "mean_reference_mapping_distance_px": mean_distance,
        "max_reference_mapping_distance_px": max_distance,
        "mean_distance_threshold_px": float(mean_distance_threshold_px),
        "max_distance_threshold_px": float(max_distance_threshold_px),
        "mapping_is_one_to_one": True,
        "force_order": "opencv_reference_centroid_order",
        "method": "reference_centroid_position_matching_with_affine_physical_to_pixel_transform",
        "affine_A_physical_to_pixel": A.tolist(),
        "affine_b_physical_to_pixel": b.tolist(),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "opencv_to_generated_index.npy", opencv_to_generated)
        np.save(output_dir / "generated_to_opencv_index.npy", generated_to_opencv)
        np.save(output_dir / "reference_mapping_distances_px.npy", distances)
        np.save(output_dir / "y_forces_reordered_to_opencv_order.npy", y_forces_reordered)

        with open(output_dir / "force_alignment_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

    return {
        "y_forces_reordered": y_forces_reordered,
        "opencv_to_generated_index": opencv_to_generated,
        "generated_to_opencv_index": generated_to_opencv,
        "reference_mapping_distances_px": distances,
        "metadata": metadata,
    }
