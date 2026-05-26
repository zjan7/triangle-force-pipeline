from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.validate_deformation_detection import validate_detected_deformations


PROJECT_ROOT = Path(__file__).resolve().parent


def get_sample_dirs(
    accepted_samples_dir: Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> list[Path]:
    if not accepted_samples_dir.exists():
        raise RuntimeError(f"Accepted samples folder does not exist: {accepted_samples_dir}")

    sample_dirs = [
        path for path in accepted_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    sample_dirs = sorted(sample_dirs, key=lambda p: int(p.name.split("_")[1]))

    if len(sample_dirs) == 0:
        raise RuntimeError(f"No accepted samples found in {accepted_samples_dir}")

    return sample_dirs


def summarize_metric(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def validate_all_samples(
    max_samples: int | None = None,
    output_root: Path = PROJECT_ROOT / "outputs" / "sample_size_validation",
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = get_sample_dirs()

    if max_samples is not None:
        sample_dirs = sample_dirs[:max_samples]

    rows = []

    print()
    print("=" * 80)
    print("VALIDATING ACCEPTED SAMPLES")
    print("=" * 80)
    print(f"Number of samples to validate: {len(sample_dirs)}")
    print()

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name

        print()
        print("-" * 80)
        print(f"Validating {sample_id}")
        print("-" * 80)

        sample_output_dir = output_root / "per_sample_plots" / sample_id

        try:
            summary = validate_detected_deformations(
                sample_dir=sample_dir,
                output_dir=sample_output_dir,
            )

            row = {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
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
                "success": True,
                "error": "",
            }

        except Exception as error:
            row = {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "success": False,
                "error": repr(error),
            }

            print(f"Validation failed for {sample_id}: {repr(error)}")

        rows.append(row)

    df = pd.DataFrame(rows)

    per_sample_csv = output_root / "per_sample_validation_results.csv"
    df.to_csv(per_sample_csv, index=False)

    print()
    print("=" * 80)
    print("PER-SAMPLE VALIDATION FINISHED")
    print("=" * 80)
    print(f"Saved: {per_sample_csv}")
    print()

    return df


def compute_sample_size_convergence(
    df: pd.DataFrame,
    sample_sizes: list[int] | None = None,
    output_root: Path = PROJECT_ROOT / "outputs" / "sample_size_validation",
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)

    valid_df = df[df["success"] == True].copy()

    if len(valid_df) == 0:
        raise RuntimeError("No successful validation results available.")

    valid_df = valid_df.reset_index(drop=True)

    n_available = len(valid_df)

    if sample_sizes is None:
        candidate_sizes = [5, 10, 20, 30, 50, 75, 100, 150, 200]
        sample_sizes = [n for n in candidate_sizes if n <= n_available]

        if n_available not in sample_sizes:
            sample_sizes.append(n_available)

    metrics = [
        "rmse_residual_px",
        "mean_residual_px",
        "median_residual_px",
        "max_residual_px",
        "r2_pixel_displacement",
        "corr_dx",
        "corr_dy",
        "mean_reference_mapping_distance_px",
        "max_reference_mapping_distance_px",
    ]

    rows = []

    for n in sample_sizes:
        subset = valid_df.iloc[:n]

        row = {
            "sample_size": int(n),
            "n_available": int(n_available),
        }

        for metric in metrics:
            if metric not in subset.columns:
                continue

            stats = summarize_metric(subset[metric].to_numpy(dtype=float))

            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_sem"] = stats["sem"]
            row[f"{metric}_min"] = stats["min"]
            row[f"{metric}_max"] = stats["max"]

        rows.append(row)

    convergence_df = pd.DataFrame(rows)

    convergence_csv = output_root / "sample_size_convergence.csv"
    convergence_df.to_csv(convergence_csv, index=False)

    print()
    print("=" * 80)
    print("SAMPLE SIZE CONVERGENCE")
    print("=" * 80)
    print(f"Saved: {convergence_csv}")
    print()
    print(convergence_df)
    print()

    return convergence_df


def plot_convergence(
    convergence_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = convergence_df["sample_size"].to_numpy(dtype=float)
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"

    if mean_col not in convergence_df.columns:
        print(f"Skipping plot for {metric}: missing column {mean_col}")
        return

    y = convergence_df[mean_col].to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", label="Mean")

    if sem_col in convergence_df.columns:
        sem = convergence_df[sem_col].to_numpy(dtype=float)
        plt.fill_between(x, y - sem, y + sem, alpha=0.25, label="±SEM")

    plt.xlabel("Number of accepted samples used")
    plt.ylabel(ylabel)
    plt.title(f"Sample-size convergence: {ylabel}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_convergence_plots(
    convergence_df: pd.DataFrame,
    output_root: Path = PROJECT_ROOT / "outputs" / "sample_size_validation",
) -> None:
    plot_convergence(
        convergence_df,
        metric="rmse_residual_px",
        ylabel="RMSE residual [px]",
        output_path=output_root / "rmse_convergence.png",
    )

    plot_convergence(
        convergence_df,
        metric="mean_residual_px",
        ylabel="Mean residual [px]",
        output_path=output_root / "mean_residual_convergence.png",
    )

    plot_convergence(
        convergence_df,
        metric="max_residual_px",
        ylabel="Maximum residual [px]",
        output_path=output_root / "max_residual_convergence.png",
    )

    plot_convergence(
        convergence_df,
        metric="r2_pixel_displacement",
        ylabel="R² displacement [-]",
        output_path=output_root / "r2_convergence.png",
    )

    print("Saved convergence plots.")


def main() -> None:
    output_root = PROJECT_ROOT / "outputs" / "sample_size_validation"

    # Set this to 100 if you want to validate exactly the first 100 accepted samples.
    max_samples = 100

    df = validate_all_samples(
        max_samples=max_samples,
        output_root=output_root,
    )

    convergence_df = compute_sample_size_convergence(
        df,
        sample_sizes=[5, 10, 20, 30, 50, 75, 100],
        output_root=output_root,
    )

    create_convergence_plots(
        convergence_df,
        output_root=output_root,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Per-sample results: {output_root / 'per_sample_validation_results.csv'}")
    print(f"Convergence table:  {output_root / 'sample_size_convergence.csv'}")
    print()


if __name__ == "__main__":
    main()
