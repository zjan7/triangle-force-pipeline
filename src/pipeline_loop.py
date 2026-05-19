from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.reference_generator import generate_reference
from src.deformation_generator import generate_deformation_sample
from src.aruco_warp import warp_deformed_to_reference
from src.triangle_detection import run_triangle_detection
from src.dataset_writer import write_accepted_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    grid_width: float = 0.1,
    grid_height: float = 0.1,
    target_triangles: int = 200,
    reference_result: dict[str, Any] | None = None,
) -> dict[str, Any]:

    if attempt_id is None:
        attempt_id = sample_id

    print()
    print("=" * 80)
    print(f"Attempt {attempt_id:06d} -> sample {sample_id:06d}")
    print("=" * 80)

    if reference_result is None:
        reference_result = generate_reference(
            output_dir=PROJECT_ROOT / "outputs" / "reference",
            grid_width=grid_width,
            grid_height=grid_height,
            target_triangles=target_triangles,
            show_plot=False,
        )

    deformation_result = generate_deformation_sample(
        output_dir=PROJECT_ROOT / "outputs" / "deformation_test",
        grid_width=grid_width,
        grid_height=grid_height,
        target_triangles=target_triangles,
        show_plot=False,
    )

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
    grid_width: float = 0.1,
    grid_height: float = 0.1,
    target_triangles: int = 200,
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

    print()
    print("#" * 80)
    print("START DATASET GENERATION")
    print("#" * 80)
    print(f"Target accepted samples this run: {n_accepted_samples}")
    print(f"Maximum attempts this run:        {max_attempts}")
    print(f"First sample ID this run:         {start_sample_id}")
    print()

    reference_result = generate_reference(
        output_dir=PROJECT_ROOT / "outputs" / "reference",
        grid_width=grid_width,
        grid_height=grid_height,
        target_triangles=target_triangles,
        show_plot=False,
    )

    while accepted_this_run < n_accepted_samples and attempts < max_attempts:
        attempts += 1
        attempt_id = attempts
        sample_id = start_sample_id + accepted_this_run

        try:
            result = run_one_sample(
                sample_id=sample_id,
                attempt_id=attempt_id,
                grid_width=grid_width,
                grid_height=grid_height,
                target_triangles=target_triangles,
                reference_result=reference_result,
            )

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