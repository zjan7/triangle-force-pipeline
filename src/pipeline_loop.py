#(pipe)This code creates the entire pipeline! next file is reference_generator.py
from __future__ import annotations
import sys
import csv
import json
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))
from trigrid import TriangleGrid
from src.reference_generator import generate_reference
from src.deformation_generator import generate_deformation_sample
from src.aruco_warp import warp_deformed_to_reference
from src.triangle_detection import run_triangle_detection
from src.dataset_writer import write_accepted_sample
from src.sample_quality import evaluate_sample_quality, write_quality_report, move_rejected_sample
from src.validate_deformation_detection import validate_detected_deformations
from src.grid_config import grid_size_from_bep2_triangle_area


def get_next_sample_id(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> int:
    
    accepted_samples_dir = Path(accepted_samples_dir)

    if not accepted_samples_dir.exists():
        return 1

    sample_dirs = [
        path for path in accepted_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    if len(sample_dirs) == 0:
        return 1

    ids = []

    for path in sample_dirs:
        try:
            ids.append(int(path.name.split("_")[1]))
        except Exception:
            continue

    if len(ids) == 0:
        return 1

    return max(ids) + 1


def append_row_to_csv(csv_path: str | Path, row: dict[str, Any]) -> None:
    
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def run_one_sample(
    sample_id: int,
    attempt_id: int | None = None,
    tris: TriangleGrid | None = None,
    reference_result: dict[str, Any] | None = None,
) -> dict[str, Any]:

    if tris is None:
        raise ValueError("run_one-sample() requires trianglegrid object")
        

    if attempt_id is None:
        attempt_id = sample_id
    grid_width = tris.width
    grid_height = tris.height
    triangle_area = tris.t_area
    physical_triangle_area = getattr(tris, "physical_triangle_area", tris.t_area)
    actual_triangles = tris.n_x * tris.n_y
    print()
    print("=" * 80)
    print(f"Attempt {attempt_id:06d} -> sample {sample_id:06d}")
    print("=" * 80)
    print(f"Grid width:              {grid_width:.6f} m")
    print(f"Grid height:             {grid_height:.6f} m")
    print(f"TriangleGrid area:       {triangle_area:.8e} m²")
    print(f"Physical triangle area:  {physical_triangle_area:.8e} m²")
    print(f"Actual triangles:        {actual_triangles}")
    attempt_dir = PROJECT_ROOT / "outputs" / "attempts" / f"attempt{attempt_id:06d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if reference_result is None: 
        reference_result = generate_reference(tris=tris, output_dir=PROJECT_ROOT / "outputs" / "reference", show_plot=False,)
    print("Starting deformation generation..")

    deformation_result = generate_deformation_sample(tris=tris, output_dir=attempt_dir / "deformation", show_plot=False,)
    print("Finished deformation generation")
    
    
    aruco_result = warp_deformed_to_reference(
        reference_image_path=reference_result["reference_image_path"],
        deformed_image_path=deformation_result["deformed_image_path"],
        output_dir=PROJECT_ROOT / "outputs" / "aruco_warp_test",
        required_ids=[1, 2, 3],
        require_all_markers=True,
    )

    triangle_result = run_triangle_detection(show=False)

    sample_dir = write_accepted_sample(
        sample_id=sample_id,
        reference_result=reference_result,
        deformation_result=deformation_result,
        aruco_result=aruco_result,
        triangle_result=triangle_result,
        accepted_samples_dir=PROJECT_ROOT / "outputs" / "accepted_samples",
    )

    return {
        "accepted": True,
        "attempt_id": attempt_id,
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "reference_result": reference_result,
        "deformation_result": deformation_result,
        "aruco_result": aruco_result,
        "triangle_result": triangle_result,
    }


def run_dataset_generation(
    n_accepted_samples: int = 10,
    max_attempts: int = 100,
    target_triangles: int = 200,
    aspect_ratio: float = 1.0,
    packing_factor: float = 2.5,
) -> dict[str, Any]:
    
    outputs_dir = PROJECT_ROOT / "outputs"
    accepted_samples_dir = outputs_dir / "accepted_samples"
    accepted_samples_dir.mkdir(parents=True, exist_ok=True)

    dataset_index_path = outputs_dir / "dataset_index.csv"
    rejected_attempts_path = outputs_dir / "rejected_attempts.csv"
    summary_path = outputs_dir / "generation_summary.json"

    start_sample_id = get_next_sample_id(accepted_samples_dir)

    accepted_this_run = 0
    rejected_this_run = 0
    attempts = 0
    grid_width, grid_height, physical_triangle_area, grid_cell_area = grid_size_from_bep2_triangle_area(
        target_triangles=target_triangles,
        aspect_ratio=aspect_ratio,
        packing_factor=packing_factor,
    )

    tris = TriangleGrid(
        grid_width,
        grid_height,
        grid_cell_area,
    )
    tris.physical_triangle_area = physical_triangle_area

    actual_triangles = tris.n_x * tris.n_y
    print()
    print("#" * 80)
    print("START DATASET GENERATION")
    print("#" * 80)
    print(f"Target accepted samples this run: {n_accepted_samples}")
    print(f"Maximum attempts this run:        {max_attempts}")
    print(f"First sample ID this run:         {start_sample_id}")
    print()

    reference_result = generate_reference(tris=tris, output_dir=PROJECT_ROOT / "outputs" / "reference", show_plot=False,)
    while accepted_this_run < n_accepted_samples and attempts < max_attempts:
        attempts += 1
        attempt_id = attempts
        sample_id = start_sample_id + accepted_this_run

        try:
            result = run_one_sample(sample_id=sample_id, attempt_id=attempt_id, tris=tris, reference_result=reference_result,)

            sample_dir = Path(result["sample_dir"])
            validation_output_dir = sample_dir / "validation"

            summary = validate_detected_deformations(
                sample_dir=sample_dir,
                output_dir=validation_output_dir,
            )

            quality_passed, rejection_reasons = evaluate_sample_quality(summary)

            print("quality_passed:", quality_passed)
            print("rejection_reasons:", rejection_reasons)

            write_quality_report(
                output_dir=sample_dir,
                passed=quality_passed,
                reasons=rejection_reasons,
                summary=summary,
            )

            if not quality_passed:
                rejected_this_run += 1

                rejected_dir = move_rejected_sample(
                    sample_dir=sample_dir,
                    rejected_root=PROJECT_ROOT / "outputs" / "rejected_samples",
                    reasons=rejection_reasons,
                    summary=summary,
                )

                rejected_row = {
                    "attempt_id": attempt_id,
                    "sample_id_if_accepted": sample_id,
                    "error_type": "QualityValidationFailed",
                    "error_message": "; ".join(rejection_reasons),
                    "rejected_dir": str(rejected_dir),
                }

                append_row_to_csv(rejected_attempts_path, rejected_row)

                print()
                print(f"REJECTED sample {sample_id:06d} after quality validation")
                print(f"Moved to: {rejected_dir}")
                for reason in rejection_reasons:
                    print(f"  - {reason}")
                print(f"Accepted so far this run: {accepted_this_run}/{n_accepted_samples}")
                print(f"Rejected so far this run: {rejected_this_run}")
                print()

                continue

            accepted_this_run += 1

            X = result["triangle_result"]["X_displacements"]
            y = result["deformation_result"]["y_forces"]

            accepted_row = {
                "sample_id": sample_id,
                "attempt_id": attempt_id,
                "sample_dir": str(result["sample_dir"]),
                "n_triangles": int(X.shape[0]),
                "X_rows": int(X.shape[0]),
                "X_cols": int(X.shape[1]),
                "y_rows": int(y.shape[0]),
                "y_cols": int(y.shape[1]),
                "rmse_residual_px": summary.get("rmse_residual_px"),
                "mean_residual_px": summary.get("mean_residual_px"),
                "max_residual_px": summary.get("max_residual_px"),
                "r2_displacement": summary.get("r2_pixel_displacement"),
            }

            append_row_to_csv(dataset_index_path, accepted_row)

            print()
            print(f"ACCEPTED sample {sample_id:06d}")
            print(f"Accepted so far this run: {accepted_this_run}/{n_accepted_samples}")
            print(f"Rejected so far this run: {rejected_this_run}")
            print()

        except Exception as error:
            rejected_this_run += 1

            rejected_row = {
                "attempt_id": attempt_id,
                "sample_id_if_accepted": sample_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            append_row_to_csv(rejected_attempts_path, rejected_row)

            print()
            print(f"REJECTED attempt {attempt_id:06d}")
            print(f"Reason: {type(error).__name__}: {error}")
            print(f"Accepted so far this run: {accepted_this_run}/{n_accepted_samples}")
            print(f"Rejected so far this run: {rejected_this_run}")
            print()

    success = accepted_this_run == n_accepted_samples

    summary = {
        "success": success,
        "requested_accepted_samples": n_accepted_samples,
        "accepted_this_run": accepted_this_run,
        "rejected_this_run": rejected_this_run,
        "attempts_this_run": attempts,
        "max_attempts": max_attempts,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "target_triangles": target_triangles,
        "actual_triangles": actual_triangles,
        "physical_triangle_area": physical_triangle_area,
        "grid_cell_area": grid_cell_area,
        "packing_factor": packing_factor,
        "dataset_index_csv": str(dataset_index_path),
        "rejected_attempts_csv": str(rejected_attempts_path),
        "accepted_samples_dir": str(accepted_samples_dir),
    }

    save_summary_json(summary_path, summary)

    print()
    print("#" * 80)
    print("DATASET GENERATION FINISHED")
    print("#" * 80)
    print(f"Success:               {success}")
    print(f"Accepted this run:     {accepted_this_run}")
    print(f"Rejected this run:     {rejected_this_run}")
    print(f"Attempts this run:     {attempts}")
    print(f"dataset_index.csv:     {dataset_index_path}")
    print(f"rejected_attempts.csv: {rejected_attempts_path}")
    print(f"summary json:          {summary_path}")
    print()

    return summary