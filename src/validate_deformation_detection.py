from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_latest_sample(accepted_samples_dir: Path = PROJECT_ROOT / "outputs" / "accepted_samples",) -> Path:
    sample_dirs = [
        path for path in accepted_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    if len(sample_dirs) == 0:
        raise RuntimeError(f"No accepted samples found in {accepted_samples_dir}")

    sample_dirs = sorted(sample_dirs, key=lambda p: int(p.name.split("_")[1]))
    return sample_dirs[-1]


def find_file(folder: Path, names: list[str]) -> Path:
    for name in names:
        path = folder / name
        if path.exists():
            return path

    for name in names:
        matches = list(folder.rglob(name))
        if len(matches) > 0:
            return matches[0]

    raise FileNotFoundError(f"Could not find any of {names} in {folder}")


def flatten_corner_array(corners: np.ndarray) -> np.ndarray:
    
    corners = np.asarray(corners, dtype=float)

    if corners.ndim == 3:
        return corners

    if corners.ndim == 4:
        n_x, n_y, n_corners, n_coords = corners.shape

        if n_corners != 3:
            raise ValueError(f"Expected 3 triangle corners, got shape {corners.shape}")

        return corners.reshape(n_x * n_y, n_corners, n_coords)

    raise ValueError(f"Corner array must have shape (N, 3, 2/3) or (n_x, n_y, 3, 2/3), got {corners.shape}")


def initial_physical_to_pixel_guess(physical_points: np.ndarray, pixel_points: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    
    x_phys = physical_points[:, 0]
    y_phys = physical_points[:, 1]

    x_pix = pixel_points[:, 0]
    y_pix = pixel_points[:, 1]

    x_phys_min = np.min(x_phys)
    x_phys_max = np.max(x_phys)
    y_phys_min = np.min(y_phys)
    y_phys_max = np.max(y_phys)

    x_pix_min = np.min(x_pix)
    x_pix_max = np.max(x_pix)
    y_pix_min = np.min(y_pix)
    y_pix_max = np.max(y_pix)

    sx = (x_pix_max - x_pix_min) / max(x_phys_max - x_phys_min, 1e-12)
    sy = (y_pix_min - y_pix_max) / max(y_phys_max - y_phys_min, 1e-12)

    # x_pixel = sx * x_phys + bx
    # y_pixel = sy * y_phys + by
    # sy is negative because image y increases downward.
    bx = x_pix_min - sx * x_phys_min
    by = y_pix_max - sy * y_phys_min

    A = np.array([[sx, 0.0], [0.0, sy],], dtype=float,)

    b = np.array([bx, by], dtype=float)

    return A, b


def apply_affine(points: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    return points @ A + b


def fit_affine(physical_points: np.ndarray, pixel_points: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    
    ones = np.ones((physical_points.shape[0], 1), dtype=float)
    design = np.hstack([physical_points, ones])

    coeffs, _, _, _ = np.linalg.lstsq(design, pixel_points, rcond=None)

    A = coeffs[:2, :]
    b = coeffs[2, :]

    return A, b


def one_to_one_nearest_mapping(generated_points_px: np.ndarray, detected_points_px: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    n_generated = generated_points_px.shape[0]
    n_detected = detected_points_px.shape[0]

    if n_generated != n_detected:
        print()
        print("WARNING:")
        print(f"Generated points: {n_generated}")
        print(f"Detected points:  {n_detected}")
        print("The mapping will use the smallest common one-to-one set.")
        print()

    # Distance matrix:
    # rows = detected points
    # cols = generated points
    diff = detected_points_px[:, None, :] - generated_points_px[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=2)

    try:
        from scipy.optimize import linear_sum_assignment

        detected_indices, generated_indices = linear_sum_assignment(distance_matrix)

        detected_to_generated = np.full(n_detected, -1, dtype=int)
        distances = np.full(n_detected, np.nan, dtype=float)

        for d_idx, g_idx in zip(detected_indices, generated_indices):
            detected_to_generated[d_idx] = g_idx
            distances[d_idx] = distance_matrix[d_idx, g_idx]

        return detected_to_generated, distances

    except Exception:
        # Fallback greedy matcher.
        all_pairs = []

        for d_idx in range(n_detected):
            for g_idx in range(n_generated):
                all_pairs.append((distance_matrix[d_idx, g_idx], d_idx, g_idx))

        all_pairs = sorted(all_pairs, key=lambda item: item[0])

        detected_to_generated = np.full(n_detected, -1, dtype=int)
        distances = np.full(n_detected, np.nan, dtype=float)

        used_detected = set()
        used_generated = set()

        for distance, d_idx, g_idx in all_pairs:
            if d_idx in used_detected or g_idx in used_generated:
                continue

            detected_to_generated[d_idx] = g_idx
            distances[d_idx] = distance

            used_detected.add(d_idx)
            used_generated.add(g_idx)

            if len(used_detected) == min(n_detected, n_generated):
                break

        return detected_to_generated, distances


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

    if ss_tot < 1e-20:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-20 or np.std(b) < 1e-20:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])
def plot_component_validation(
    generated_values,
    detected_values,
    component_name,
    output_path,
    r2=None,
    rmse=None,
    corr=None,
):
    generated_values = np.asarray(generated_values, dtype=float)
    detected_values = np.asarray(detected_values, dtype=float)

    finite_mask = np.isfinite(generated_values) & np.isfinite(detected_values)
    generated_values = generated_values[finite_mask]
    detected_values = detected_values[finite_mask]

    data_min = min(np.min(generated_values), np.min(detected_values))
    data_max = max(np.max(generated_values), np.max(detected_values))

    data_range = max(data_max - data_min, 1.0)
    margin = 0.12 * data_range

    axis_min = data_min - margin
    axis_max = data_max + margin

    line = np.linspace(axis_min, axis_max, 300)

    plt.figure(figsize=(7, 7))

    plt.scatter(generated_values, detected_values, s=22, alpha=0.85, label="Triangles",)

    # Perfect agreement line.
    plt.plot(line, line, linewidth=2.0, label="Perfect agreement",)

    # ±RMSE band.
    if rmse is not None and np.isfinite(rmse):
        plt.fill_between(line, line - rmse, line + rmse, alpha=0.20, label=f"±RMSE = {rmse:.3f} px",)

    # ±0.5 pixel band. This is useful because the achieved errors are sub-pixel.
    plt.fill_between(line, line - 0.5, line + 0.5, alpha=0.10, label="±0.5 px",)

    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel(f"Generated {component_name} [px]")
    plt.ylabel(f"OpenCV detected {component_name} [px]")

    title = f"{component_name} validation, position matched"

    metric_lines = []

    if r2 is not None and np.isfinite(r2):
        metric_lines.append(f"R² = {r2:.6f}")

    if corr is not None and np.isfinite(corr):
        metric_lines.append(f"corr = {corr:.6f}")

    if rmse is not None and np.isfinite(rmse):
        metric_lines.append(f"RMSE = {rmse:.3f} px")

    if len(metric_lines) > 0:
        title += "\n" + ", ".join(metric_lines)

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_component_residuals(generated_values, detected_values, component_name, output_path,):
    generated_values = np.asarray(generated_values, dtype=float)
    detected_values = np.asarray(detected_values, dtype=float)

    finite_mask = np.isfinite(generated_values) & np.isfinite(detected_values)
    generated_values = generated_values[finite_mask]
    detected_values = detected_values[finite_mask]

    residuals = detected_values - generated_values

    plt.figure(figsize=(8, 5))

    plt.scatter(generated_values, residuals, s=22, alpha=0.85, label="Triangles",)

    plt.axhline(0.0, linewidth=2.0, label="Zero error",)

    plt.axhline(0.5, linestyle="--", linewidth=1.2, label="±0.5 px",)

    plt.axhline(-0.5, linestyle="--", linewidth=1.2,)

    plt.xlabel(f"Generated {component_name} [px]")
    plt.ylabel(f"Residual error in {component_name} [px]")
    plt.title(f"{component_name} residuals, position matched")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def validate_detected_deformations(
    sample_dir: str | Path | None = None,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "deformation_validation",
    n_refinement_steps: int = 3,
) -> dict:
    if sample_dir is None:
        sample_dir = load_latest_sample()

    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("VALIDATE OPENCV DETECTED DEFORMATIONS")
    print("=" * 80)
    print(f"Sample folder: {sample_dir}")

    raw_dir = sample_dir / "raw"
    full_matrices_dir = sample_dir / "full_matrices"

    reference_corners_path = find_file(raw_dir, ["reference_zero_force_corners.npy", "reference_corners.npy"],)

    deformed_corners_path = find_file(raw_dir, ["deformed_corners.npy", "hoeken_start.npy"],)

    triangle_matrix_path = find_file(full_matrices_dir, ["triangle_matrix_full.npy"],)

    reference_corners = np.load(reference_corners_path, allow_pickle=True)
    deformed_corners = np.load(deformed_corners_path, allow_pickle=True)
    triangle_matrix = np.load(triangle_matrix_path, allow_pickle=True)

    reference_corners = flatten_corner_array(reference_corners)
    deformed_corners = flatten_corner_array(deformed_corners)
    triangle_matrix = np.asarray(triangle_matrix, dtype=float)

    if reference_corners.shape != deformed_corners.shape:
        raise ValueError(
            f"Reference and deformed corner arrays have different shapes: "
            f"{reference_corners.shape} vs {deformed_corners.shape}"
        )

    print(f"Reference corners shape: {reference_corners.shape}")
    print(f"Deformed corners shape:  {deformed_corners.shape}")
    print(f"Triangle matrix shape:   {triangle_matrix.shape}")

    generated_ref_centroids_physical = reference_corners[:, :, :2].mean(axis=1)
    generated_def_centroids_physical = deformed_corners[:, :, :2].mean(axis=1)

    generated_displacement_physical = (generated_def_centroids_physical - generated_ref_centroids_physical)

    opencv_ids = triangle_matrix[:, 0].astype(int)
    opencv_ref_centroids_px = triangle_matrix[:, 3:5]
    opencv_def_centroids_px = triangle_matrix[:, 5:7]
    opencv_displacement_px = triangle_matrix[:, 7:9]

    n_generated = generated_ref_centroids_physical.shape[0]
    n_detected = opencv_ref_centroids_px.shape[0]
    A, b = initial_physical_to_pixel_guess(physical_points=generated_ref_centroids_physical, pixel_points=opencv_ref_centroids_px,)
    detected_to_generated = None
    mapping_distances = None

    for step in range(n_refinement_steps):
        generated_ref_centroids_px_est = apply_affine(generated_ref_centroids_physical, A, b,)

        detected_to_generated, mapping_distances = one_to_one_nearest_mapping(generated_points_px=generated_ref_centroids_px_est, detected_points_px=opencv_ref_centroids_px,)

        valid_detected_rows = np.where(detected_to_generated >= 0)[0]
        matched_generated_indices = detected_to_generated[valid_detected_rows]

        A, b = fit_affine(physical_points=generated_ref_centroids_physical[matched_generated_indices], pixel_points=opencv_ref_centroids_px[valid_detected_rows],)
    generated_ref_centroids_px_final = apply_affine(generated_ref_centroids_physical, A, b,)

    detected_to_generated, mapping_distances = one_to_one_nearest_mapping(generated_points_px=generated_ref_centroids_px_final, detected_points_px=opencv_ref_centroids_px,)

    valid_detected_rows = np.where(detected_to_generated >= 0)[0]
    matched_generated_indices = detected_to_generated[valid_detected_rows]
    generated_ref_px_matched = apply_affine(generated_ref_centroids_physical[matched_generated_indices], A, b,)

    generated_def_px_matched = apply_affine(generated_def_centroids_physical[matched_generated_indices], A, b,)

    generated_displacement_px_predicted = (generated_def_px_matched - generated_ref_px_matched)

    opencv_displacement_px_matched = opencv_displacement_px[valid_detected_rows]

    residual = opencv_displacement_px_matched - generated_displacement_px_predicted
    residual_norm = np.linalg.norm(residual, axis=1)

    r2 = r2_score(opencv_displacement_px_matched, generated_displacement_px_predicted)
    rmse_px = float(np.sqrt(np.mean(residual_norm**2)))
    mean_residual_px = float(np.mean(residual_norm))
    median_residual_px = float(np.median(residual_norm))
    max_residual_px = float(np.max(residual_norm))

    corr_dx = safe_corr(opencv_displacement_px_matched[:, 0], generated_displacement_px_predicted[:, 0],)

    corr_dy = safe_corr(opencv_displacement_px_matched[:, 1], generated_displacement_px_predicted[:, 1],)

    mean_mapping_distance_px = float(np.nanmean(mapping_distances))
    max_mapping_distance_px = float(np.nanmax(mapping_distances))

    table = pd.DataFrame(
        {
            "opencv_row_index": valid_detected_rows,
            "opencv_id": opencv_ids[valid_detected_rows],
            "matched_generated_index": matched_generated_indices,
            "reference_mapping_distance_px": mapping_distances[valid_detected_rows],
            "generated_ref_x_physical": generated_ref_centroids_physical[matched_generated_indices, 0],
            "generated_ref_y_physical": generated_ref_centroids_physical[matched_generated_indices, 1],
            "opencv_ref_x_px": opencv_ref_centroids_px[valid_detected_rows, 0],
            "opencv_ref_y_px": opencv_ref_centroids_px[valid_detected_rows, 1],
            "generated_dx_physical": generated_displacement_physical[matched_generated_indices, 0],
            "generated_dy_physical": generated_displacement_physical[matched_generated_indices, 1],
            "generated_dx_predicted_px": generated_displacement_px_predicted[:, 0],
            "generated_dy_predicted_px": generated_displacement_px_predicted[:, 1],
            "opencv_dx_px": opencv_displacement_px_matched[:, 0],
            "opencv_dy_px": opencv_displacement_px_matched[:, 1],
            "residual_dx_px": residual[:, 0],
            "residual_dy_px": residual[:, 1],
            "residual_norm_px": residual_norm,
        }
    )

    validation_csv_path = output_dir / "deformation_validation_table_position_matched.csv"
    table.to_csv(validation_csv_path, index=False)

    dx_plot_path = output_dir / "dx_validation_position_matched.png"
    dy_plot_path = output_dir / "dy_validation_position_matched.png"

    dx_residual_plot_path = output_dir / "dx_residuals_position_matched.png"
    dy_residual_plot_path = output_dir / "dy_residuals_position_matched.png"

    plot_component_validation(
        generated_values=table["generated_dx_predicted_px"],
        detected_values=table["opencv_dx_px"],
        component_name="dx",
        output_path=dx_plot_path,
        r2=r2,
        rmse=rmse_px,
        corr=corr_dx,
    )

    plot_component_validation(
        generated_values=table["generated_dy_predicted_px"],
        detected_values=table["opencv_dy_px"],
        component_name="dy",
        output_path=dy_plot_path,
        r2=r2,
        rmse=rmse_px,
        corr=corr_dy,
    )

    plot_component_residuals(
        generated_values=table["generated_dx_predicted_px"],
        detected_values=table["opencv_dx_px"],
        component_name="dx",
        output_path=dx_residual_plot_path,
    )

    plot_component_residuals(
        generated_values=table["generated_dy_predicted_px"],
        detected_values=table["opencv_dy_px"],
        component_name="dy",
        output_path=dy_residual_plot_path,
    )
    plt.figure(figsize=(7, 5))
    plt.hist(residual_norm, bins=30)
    plt.xlabel("Residual magnitude [px]")
    plt.ylabel("Number of triangles")
    plt.title("OpenCV deformation residual error, position matched")
    plt.grid(True)
    plt.tight_layout()
    residual_plot_path = output_dir / "residual_histogram_position_matched.png"
    plt.savefig(residual_plot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(mapping_distances[valid_detected_rows], bins=30)
    plt.xlabel("Reference centroid mapping distance [px]")
    plt.ylabel("Number of triangles")
    plt.title("Generated reference centroid to OpenCV reference centroid mapping")
    plt.grid(True)
    plt.tight_layout()
    mapping_plot_path = output_dir / "reference_centroid_mapping_distance.png"
    plt.savefig(mapping_plot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.quiver(
        opencv_ref_centroids_px[valid_detected_rows, 0],
        opencv_ref_centroids_px[valid_detected_rows, 1],
        generated_displacement_px_predicted[:, 0],
        generated_displacement_px_predicted[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        label="generated converted to px",
    )
    plt.quiver(
        opencv_ref_centroids_px[valid_detected_rows, 0],
        opencv_ref_centroids_px[valid_detected_rows, 1],
        opencv_displacement_px_matched[:, 0],
        opencv_displacement_px_matched[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002,
        alpha=0.6,
        label="OpenCV detected",
    )
    plt.gca().invert_yaxis()
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.title("Generated vs OpenCV displacement vectors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    vector_plot_path = output_dir / "vector_comparison_position_matched.png"
    plt.savefig(vector_plot_path, dpi=200)
    plt.close()

    summary = {
        "sample_dir": str(sample_dir),
        "reference_corners_path": str(reference_corners_path),
        "deformed_corners_path": str(deformed_corners_path),
        "triangle_matrix_path": str(triangle_matrix_path),
        "n_generated_triangles": int(n_generated),
        "n_detected_triangles": int(n_detected),
        "n_valid_compared_triangles": int(len(valid_detected_rows)),
        "affine_A_physical_to_pixel": A.tolist(),
        "affine_b_physical_to_pixel": b.tolist(),
        "mean_reference_mapping_distance_px": mean_mapping_distance_px,
        "max_reference_mapping_distance_px": max_mapping_distance_px,
        "r2_pixel_displacement": r2,
        "rmse_residual_px": rmse_px,
        "mean_residual_px": mean_residual_px,
        "median_residual_px": median_residual_px,
        "max_residual_px": max_residual_px,
        "corr_dx": corr_dx,
        "corr_dy": corr_dy,
        "validation_csv": str(validation_csv_path),
        "dx_plot": str(dx_plot_path),
        "dy_plot": str(dy_plot_path),
        "dx_residual_plot": str(dx_residual_plot_path),
        "dy_residual_plot": str(dy_residual_plot_path),
        "residual_histogram": str(residual_plot_path),
        "mapping_plot": str(mapping_plot_path),
        "vector_plot": str(vector_plot_path),
    }

    summary_path = output_dir / "deformation_validation_summary_position_matched.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print()
    print("=" * 80)
    print("VALIDATION RESULTS, POSITION MATCHED")
    print("=" * 80)
    print(f"Generated triangles:                  {n_generated}")
    print(f"Detected triangles:                   {n_detected}")
    print(f"Compared triangles:                   {len(valid_detected_rows)}")
    print(f"Mean reference mapping distance:       {mean_mapping_distance_px:.3f} px")
    print(f"Max reference mapping distance:        {max_mapping_distance_px:.3f} px")
    print(f"R2 displacement:                      {r2:.6f}")
    print(f"RMSE residual:                        {rmse_px:.3f} px")
    print(f"Mean residual:                        {mean_residual_px:.3f} px")
    print(f"Median residual:                      {median_residual_px:.3f} px")
    print(f"Max residual:                         {max_residual_px:.3f} px")
    print(f"Correlation dx:                       {corr_dx:.6f}")
    print(f"Correlation dy:                       {corr_dy:.6f}")
    print()
    print(f"Saved table:                          {validation_csv_path}")
    print(f"Saved dx residual plot:               {dx_residual_plot_path}")
    print(f"Saved dy residual plot:               {dy_residual_plot_path}")
    print(f"Saved residual histogram:             {residual_plot_path}")
    print(f"Saved mapping plot:                   {mapping_plot_path}")
    print(f"Saved vector comparison plot:         {vector_plot_path}")
    print(f"Saved summary:                        {summary_path}")
    print()

    return summary


if __name__ == "__main__":
    validate_detected_deformations()
