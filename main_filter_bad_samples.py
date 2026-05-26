from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.validate_deformation_detection import validate_detected_deformations


PROJECT_ROOT = Path(__file__).resolve().parent

ACCEPTED_DIR = PROJECT_ROOT / "outputs" / "accepted_samples"
REJECTED_DIR = PROJECT_ROOT / "outputs" / "rejected_samples" / "high_error"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "error_filter_validation"

REJECTED_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
MAX_ALLOWED_RMSE_PX = 0.50
MAX_ALLOWED_MEAN_RESIDUAL_PX = 0.25
MAX_ALLOWED_MAX_RESIDUAL_PX = 1.50

MAX_ALLOWED_MEAN_MAPPING_PX = 1.00
MAX_ALLOWED_MAX_MAPPING_PX = 2.00

# Use these as warnings first, not hard rejection.
R2_WARNING_LIMIT = 0.98
CORR_WARNING_LIMIT = 0.98


def get_sample_dirs() -> list[Path]:
    sample_dirs = [
        path for path in ACCEPTED_DIR.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    return sorted(sample_dirs, key=lambda p: int(p.name.split("_")[1]))


def evaluate_summary(summary: dict) -> tuple[bool, list[str], list[str]]:
    reasons = []
    warnings = []

    n_generated = summary["n_generated_triangles"]
    n_detected = summary["n_detected_triangles"]
    n_compared = summary["n_valid_compared_triangles"]

    mean_mapping = summary["mean_reference_mapping_distance_px"]
    max_mapping = summary["max_reference_mapping_distance_px"]

    r2 = summary["r2_pixel_displacement"]
    rmse = summary["rmse_residual_px"]
    mean_residual = summary["mean_residual_px"]
    max_residual = summary["max_residual_px"]

    corr_dx = summary["corr_dx"]
    corr_dy = summary["corr_dy"]

    # Hard geometric consistency checks.
    if n_generated != n_detected:
        reasons.append(
            f"generated triangles ({n_generated}) != detected triangles ({n_detected})"
        )

    if n_generated != n_compared:
        reasons.append(
            f"generated triangles ({n_generated}) != compared triangles ({n_compared})"
        )

    # Hard force-label alignment checks.
    if mean_mapping > MAX_ALLOWED_MEAN_MAPPING_PX:
        reasons.append(
            f"mean mapping distance {mean_mapping:.3f} px > {MAX_ALLOWED_MEAN_MAPPING_PX:.3f} px"
        )

    if max_mapping > MAX_ALLOWED_MAX_MAPPING_PX:
        reasons.append(
            f"max mapping distance {max_mapping:.3f} px > {MAX_ALLOWED_MAX_MAPPING_PX:.3f} px"
        )

    # Hard displacement-error checks.
    if rmse > MAX_ALLOWED_RMSE_PX:
        reasons.append(
            f"RMSE {rmse:.3f} px > {MAX_ALLOWED_RMSE_PX:.3f} px"
        )

    if mean_residual > MAX_ALLOWED_MEAN_RESIDUAL_PX:
        reasons.append(
            f"mean residual {mean_residual:.3f} px > {MAX_ALLOWED_MEAN_RESIDUAL_PX:.3f} px"
        )

    if max_residual > MAX_ALLOWED_MAX_RESIDUAL_PX:
        reasons.append(
            f"max residual {max_residual:.3f} px > {MAX_ALLOWED_MAX_RESIDUAL_PX:.3f} px"
        )

    # Diagnostic warnings.
    if r2 < R2_WARNING_LIMIT:
        warnings.append(
            f"R2 {r2:.6f} < {R2_WARNING_LIMIT:.3f}"
        )

    if corr_dx < CORR_WARNING_LIMIT:
        warnings.append(
            f"corr_dx {corr_dx:.6f} < {CORR_WARNING_LIMIT:.3f}"
        )

    if corr_dy < CORR_WARNING_LIMIT:
        warnings.append(
            f"corr_dy {corr_dy:.6f} < {CORR_WARNING_LIMIT:.3f}"
        )

    reject = len(reasons) > 0

    return reject, reasons, warnings


def move_rejected_sample(sample_dir: Path, reasons: list[str]) -> Path:
    destination = REJECTED_DIR / sample_dir.name

    if destination.exists():
        shutil.rmtree(destination)

    shutil.move(str(sample_dir), str(destination))

    reason_file = destination / "rejection_reason.txt"

    with open(reason_file, "w", encoding="utf-8") as file:
        file.write("Rejected because:\n")
        for reason in reasons:
            file.write(f"- {reason}\n")

    return destination


def main() -> None:
    sample_dirs = get_sample_dirs()

    if len(sample_dirs) == 0:
        raise RuntimeError(f"No accepted samples found in {ACCEPTED_DIR}")

    rows = []

    print()
    print("=" * 80)
    print("FILTER BAD ACCEPTED SAMPLES BY VALIDATION ERROR")
    print("=" * 80)
    print(f"Accepted samples found: {len(sample_dirs)}")
    print()

    for sample_dir in sample_dirs:
        print("-" * 80)
        print(f"Checking {sample_dir.name}")

        sample_validation_dir = VALIDATION_DIR / sample_dir.name

        try:
            summary = validate_detected_deformations(
                sample_dir=sample_dir,
                output_dir=sample_validation_dir,
            )

            reject, reasons, warnings = evaluate_summary(summary)

            row = {
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "rejected": reject,
                "reasons": " | ".join(reasons),
                "warnings": " | ".join(warnings),
                "n_generated_triangles": summary["n_generated_triangles"],
                "n_detected_triangles": summary["n_detected_triangles"],
                "n_valid_compared_triangles": summary["n_valid_compared_triangles"],
                "mean_reference_mapping_distance_px": summary["mean_reference_mapping_distance_px"],
                "max_reference_mapping_distance_px": summary["max_reference_mapping_distance_px"],
                "r2_pixel_displacement": summary["r2_pixel_displacement"],
                "rmse_residual_px": summary["rmse_residual_px"],
                "mean_residual_px": summary["mean_residual_px"],
                "median_residual_px": summary["median_residual_px"],
                "max_residual_px": summary["max_residual_px"],
                "corr_dx": summary["corr_dx"],
                "corr_dy": summary["corr_dy"],
            }

            if reject:
                destination = move_rejected_sample(sample_dir, reasons)
                row["moved_to"] = str(destination)

                print("REJECTED")
                for reason in reasons:
                    print(f"  - {reason}")
            else:
                row["moved_to"] = ""
                print("ACCEPTED")

                if len(warnings) > 0:
                    print("Warnings:")
                    for warning in warnings:
                        print(f"  - {warning}")

        except Exception as error:
            row = {
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "rejected": True,
                "reasons": f"validation crashed: {repr(error)}",
                "warnings": "",
                "moved_to": "",
            }

            destination = move_rejected_sample(sample_dir, [f"validation crashed: {repr(error)}"])
            row["moved_to"] = str(destination)

            print("REJECTED because validation crashed")
            print(repr(error))

        rows.append(row)

    df = pd.DataFrame(rows)

    summary_csv = VALIDATION_DIR / "sample_error_filter_summary.csv"
    df.to_csv(summary_csv, index=False)

    n_rejected = int(df["rejected"].sum())
    n_total = len(df)
    n_remaining = n_total - n_rejected

    print()
    print("=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)
    print(f"Total checked:     {n_total}")
    print(f"Rejected:          {n_rejected}")
    print(f"Remaining accepted:{n_remaining}")
    print(f"Saved summary:     {summary_csv}")
    print()
    print("Rejected samples moved to:")
    print(REJECTED_DIR)
    print()


if __name__ == "__main__":
    main()
