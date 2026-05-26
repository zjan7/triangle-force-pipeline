from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.pipeline_loop import run_dataset_generation
from src.validate_deformation_detection import validate_detected_deformations


PROJECT_ROOT = Path(__file__).resolve().parent


def safe_remove_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_triangle_count_test(
    triangle_counts: list[int],
    n_accepted_samples_per_count: int = 1,
    max_attempts_per_count: int = 30,
    grid_width: float = 0.1,
    grid_height: float = 0.1,
) -> pd.DataFrame:
    results_root = PROJECT_ROOT / "outputs_triangle_count_tests"
    results_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for target_triangles in triangle_counts:
        print()
        print("#" * 100)
        print(f"TESTING target_triangles = {target_triangles}")
        print("#" * 100)
        print()

        # Use a clean outputs folder for each test.
        outputs_dir = PROJECT_ROOT / "outputs"

        if outputs_dir.exists():
            backup_dir = results_root / f"outputs_before_test_{target_triangles}"

            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            shutil.move(str(outputs_dir), str(backup_dir))

        try:
            generation_summary = run_dataset_generation(
                n_accepted_samples=n_accepted_samples_per_count,
                max_attempts=max_attempts_per_count,
                grid_width=grid_width,
                grid_height=grid_height,
                target_triangles=target_triangles,
            )

            accepted_samples_dir = PROJECT_ROOT / "outputs" / "accepted_samples"

            sample_dirs = [
                p for p in accepted_samples_dir.iterdir()
                if p.is_dir() and p.name.startswith("sample_")
            ]

            sample_dirs = sorted(sample_dirs, key=lambda p: int(p.name.split("_")[1]))

            if len(sample_dirs) == 0:
                raise RuntimeError(
                    f"No accepted samples were generated for target_triangles={target_triangles}"
                )

            latest_sample_dir = sample_dirs[-1]

            validation_output_dir = (
                results_root
                / f"triangle_count_{target_triangles}"
                / "deformation_validation"
            )

            validation_summary = validate_detected_deformations(
                sample_dir=latest_sample_dir,
                output_dir=validation_output_dir,
            )

            row = {
                "target_triangles": int(target_triangles),
                "accepted_sample_dir": str(latest_sample_dir),
                "generation_success": bool(generation_summary.get("success", True))
                if isinstance(generation_summary, dict)
                else True,
                "n_generated_triangles": validation_summary["n_generated_triangles"],
                "n_detected_triangles": validation_summary["n_detected_triangles"],
                "n_valid_compared_triangles": validation_summary["n_valid_compared_triangles"],
                "mean_reference_mapping_distance_px": validation_summary[
                    "mean_reference_mapping_distance_px"
                ],
                "max_reference_mapping_distance_px": validation_summary[
                    "max_reference_mapping_distance_px"
                ],
                "r2_pixel_displacement": validation_summary["r2_pixel_displacement"],
                "rmse_residual_px": validation_summary["rmse_residual_px"],
                "mean_residual_px": validation_summary["mean_residual_px"],
                "median_residual_px": validation_summary["median_residual_px"],
                "max_residual_px": validation_summary["max_residual_px"],
                "corr_dx": validation_summary["corr_dx"],
                "corr_dy": validation_summary["corr_dy"],
            }

            # Estimate relative errors using mean nearest-neighbour spacing
            # from OpenCV reference centroids in the triangle matrix.
            triangle_matrix_path = latest_sample_dir / "full_matrices" / "triangle_matrix_full.npy"

            if triangle_matrix_path.exists():
                import numpy as np
                from scipy.spatial import cKDTree

                triangle_matrix = np.load(triangle_matrix_path)
                ref_centroids = triangle_matrix[:, 3:5]

                tree = cKDTree(ref_centroids)
                distances, _ = tree.query(ref_centroids, k=2)

                nearest_spacing_px = distances[:, 1]
                mean_spacing_px = float(np.mean(nearest_spacing_px))
                min_spacing_px = float(np.min(nearest_spacing_px))

                row["mean_nearest_triangle_spacing_px"] = mean_spacing_px
                row["min_nearest_triangle_spacing_px"] = min_spacing_px

                row["rmse_residual_percent_of_mean_spacing"] = (
                    row["rmse_residual_px"] / mean_spacing_px * 100.0
                )

                row["mean_residual_percent_of_mean_spacing"] = (
                    row["mean_residual_px"] / mean_spacing_px * 100.0
                )

                row["max_residual_percent_of_mean_spacing"] = (
                    row["max_residual_px"] / mean_spacing_px * 100.0
                )

                row["max_mapping_percent_of_mean_spacing"] = (
                    row["max_reference_mapping_distance_px"] / mean_spacing_px * 100.0
                )

            all_rows.append(row)

            count_output_dir = results_root / f"triangle_count_{target_triangles}"
            count_output_dir.mkdir(parents=True, exist_ok=True)

            with open(count_output_dir / "summary.json", "w", encoding="utf-8") as file:
                json.dump(row, file, indent=4)

            # Save a copy of the accepted sample for later inspection.
            sample_copy_dir = count_output_dir / latest_sample_dir.name

            if sample_copy_dir.exists():
                shutil.rmtree(sample_copy_dir)

            shutil.copytree(latest_sample_dir, sample_copy_dir)

        except Exception as error:
            row = {
                "target_triangles": int(target_triangles),
                "error": repr(error),
            }

            all_rows.append(row)

            print()
            print(f"FAILED target_triangles={target_triangles}")
            print(f"Reason: {repr(error)}")
            print()

        finally:
            # Move this test's outputs into the result folder.
            current_outputs_dir = PROJECT_ROOT / "outputs"

            if current_outputs_dir.exists():
                final_outputs_dir = results_root / f"outputs_triangle_count_{target_triangles}"

                if final_outputs_dir.exists():
                    shutil.rmtree(final_outputs_dir)

                shutil.move(str(current_outputs_dir), str(final_outputs_dir))

            # Restore previous outputs if they existed.
            backup_dir = results_root / f"outputs_before_test_{target_triangles}"

            if backup_dir.exists():
                shutil.move(str(backup_dir), str(PROJECT_ROOT / "outputs"))

    df = pd.DataFrame(all_rows)

    summary_csv = results_root / "triangle_count_validation_summary.csv"
    df.to_csv(summary_csv, index=False)

    print()
    print("#" * 100)
    print("TRIANGLE COUNT TEST FINISHED")
    print("#" * 100)
    print(f"Saved summary CSV: {summary_csv}")
    print()
    print(df)

    return df


if __name__ == "__main__":
    run_triangle_count_test(
        triangle_counts=[20, 500, 1000],
        n_accepted_samples_per_count=1,
        max_attempts_per_count=50,
        grid_width=0.1,
        grid_height=0.1,
    )
