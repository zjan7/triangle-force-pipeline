from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


QUALITY_THRESHOLDS = {
    "mean_reference_mapping_distance_px": 1.0,
    "max_reference_mapping_distance_px": 2.0,
    "rmse_residual_px": 0.5,
    "mean_residual_px": 0.25,
    "max_residual_px": 1.5,
    "abs_mean_residual_dx_px": 0.5,
    "abs_mean_residual_dy_px": 0.5,
}


def evaluate_sample_quality(summary: dict[str, Any]) -> tuple[bool, list[str]]:

    reasons: list[str] = []

    generated = summary.get("n_generated_triangles")
    detected = summary.get("n_detected_triangles")
    compared = summary.get("n_valid_compared_triangles")

    if generated is None or detected is None or compared is None:
        reasons.append("Missing triangle-count information in validation summary.")
    else:
        if not (generated == detected == compared):
            reasons.append(
                f"Triangle count mismatch: generated={generated}, "
                f"detected={detected}, compared={compared}."
            )

    mean_mapping = summary.get("mean_reference_mapping_distance_px")
    max_mapping = summary.get("max_reference_mapping_distance_px")
    rmse = summary.get("rmse_residual_px")
    mean_residual = summary.get("mean_residual_px")
    max_residual = summary.get("max_residual_px")

    mean_residual_dx = summary.get("mean_residual_dx_px", 0.0)
    mean_residual_dy = summary.get("mean_residual_dy_px", 0.0)

    if mean_mapping is not None:
        if mean_mapping > QUALITY_THRESHOLDS["mean_reference_mapping_distance_px"]:
            reasons.append(
                f"Mean reference mapping distance too high: "
                f"{mean_mapping:.3f} px."
            )

    if max_mapping is not None:
        if max_mapping > QUALITY_THRESHOLDS["max_reference_mapping_distance_px"]:
            reasons.append(
                f"Max reference mapping distance too high: "
                f"{max_mapping:.3f} px."
            )

    if rmse is not None:
        if rmse > QUALITY_THRESHOLDS["rmse_residual_px"]:
            reasons.append(f"RMSE residual too high: {rmse:.3f} px.")

    if mean_residual is not None:
        if mean_residual > QUALITY_THRESHOLDS["mean_residual_px"]:
            reasons.append(f"Mean residual too high: {mean_residual:.3f} px.")

    if max_residual is not None:
        if max_residual > QUALITY_THRESHOLDS["max_residual_px"]:
            reasons.append(f"Max residual too high: {max_residual:.3f} px.")

    if abs(mean_residual_dx) > QUALITY_THRESHOLDS["abs_mean_residual_dx_px"]:
        reasons.append(
            f"Mean residual dx too high: {mean_residual_dx:.3f} px."
        )

    if abs(mean_residual_dy) > QUALITY_THRESHOLDS["abs_mean_residual_dy_px"]:
        reasons.append(
            f"Mean residual dy too high: {mean_residual_dy:.3f} px."
        )

    passed = len(reasons) == 0
    return passed, reasons


def write_quality_report(
    output_dir: str | Path,
    passed: bool,
    reasons: list[str],
    summary: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "quality_passed": passed,
        "rejection_reasons": reasons,
        "thresholds": QUALITY_THRESHOLDS,
        "validation_summary": summary,
    }

    report_path = output_dir / "quality_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report_path


def move_rejected_sample(
    sample_dir: str | Path,
    rejected_root: str | Path,
    reasons: list[str],
    summary: dict[str, Any],
) -> Path:
    sample_dir = Path(sample_dir)
    rejected_root = Path(rejected_root)
    rejected_root.mkdir(parents=True, exist_ok=True)

    rejected_dir = rejected_root / sample_dir.name

    if rejected_dir.exists():
        shutil.rmtree(rejected_dir)

    shutil.move(str(sample_dir), str(rejected_dir))

    write_quality_report(
        output_dir=rejected_dir,
        passed=False,
        reasons=reasons,
        summary=summary,
    )

    reason_path = rejected_dir / "rejection_reason.txt"
    with open(reason_path, "w", encoding="utf-8") as f:
        for reason in reasons:
            f.write(reason + "\n")

    return rejected_dir