from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "outputs" / "sample_size_validation" / "sample_size_convergence.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sample_size_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_stable_sample_size(
    sample_sizes: np.ndarray,
    values: np.ndarray,
    final_value: float,
    abs_tolerance: float,
) -> int | None:
    """
    Return the first sample size N for which the metric stays within
    abs_tolerance of the final value for all larger sample sizes.
    """
    diffs = np.abs(values - final_value)

    for i in range(len(sample_sizes)):
        if np.all(diffs[i:] <= abs_tolerance):
            return int(sample_sizes[i])

    return None


def plot_metric_convergence(
    df: pd.DataFrame,
    metric_prefix: str,
    ylabel: str,
    filename: str,
    abs_tolerance: float | None = None,
) -> int | None:
    x = df["sample_size"].to_numpy(dtype=float)
    mean_col = f"{metric_prefix}_mean"
    sem_col = f"{metric_prefix}_sem"

    if mean_col not in df.columns:
        raise RuntimeError(f"Column not found: {mean_col}")

    y = df[mean_col].to_numpy(dtype=float)
    sem = df[sem_col].to_numpy(dtype=float) if sem_col in df.columns else None

    final_value = float(y[-1])

    stable_n = None
    if abs_tolerance is not None:
        stable_n = find_stable_sample_size(
            sample_sizes=x,
            values=y,
            final_value=final_value,
            abs_tolerance=abs_tolerance,
        )

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", label=f"Mean {ylabel}")

    if sem is not None:
        plt.fill_between(x, y - sem, y + sem, alpha=0.25, label="± SEM")

    plt.axhline(final_value, linestyle="--", label=f"Final mean = {final_value:.6f}")

    if abs_tolerance is not None:
        plt.axhline(final_value + abs_tolerance, linestyle=":", label=f"Tolerance band")
        plt.axhline(final_value - abs_tolerance, linestyle=":")

    if stable_n is not None:
        plt.axvline(stable_n, linestyle="-.", label=f"Stable sample size = {stable_n}")

    plt.xlabel("Number of accepted samples used")
    plt.ylabel(ylabel)
    plt.title(f"Sample-size convergence of {ylabel}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

    return stable_n


def plot_difference_to_final(
    df: pd.DataFrame,
    metric_prefix: str,
    ylabel: str,
    filename: str,
) -> None:
    x = df["sample_size"].to_numpy(dtype=float)
    mean_col = f"{metric_prefix}_mean"

    if mean_col not in df.columns:
        raise RuntimeError(f"Column not found: {mean_col}")

    y = df[mean_col].to_numpy(dtype=float)
    final_value = float(y[-1])
    delta = np.abs(y - final_value)

    plt.figure(figsize=(8, 5))
    plt.plot(x, delta, marker="o")
    plt.xlabel("Number of accepted samples used")
    plt.ylabel(f"|mean - final mean| of {ylabel}")
    plt.title(f"Distance to final estimate: {ylabel}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Could not find: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = df.sort_values("sample_size").reset_index(drop=True)

    # Adjustable tolerances:
    # RMSE considered stable if within 0.005 px of the final mean.
    rmse_tol = 0.005

    # R² considered stable if within 0.001 of the final mean.
    r2_tol = 0.001

    stable_rmse_n = plot_metric_convergence(
        df=df,
        metric_prefix="rmse_residual_px",
        ylabel="RMSE residual [px]",
        filename="rmse_convergence_with_stability.png",
        abs_tolerance=rmse_tol,
    )

    stable_r2_n = plot_metric_convergence(
        df=df,
        metric_prefix="r2_pixel_displacement",
        ylabel="R² displacement [-]",
        filename="r2_convergence_with_stability.png",
        abs_tolerance=r2_tol,
    )

    plot_metric_convergence(
        df=df,
        metric_prefix="max_residual_px",
        ylabel="Maximum residual [px]",
        filename="max_residual_convergence_with_stability.png",
        abs_tolerance=None,
    )

    plot_difference_to_final(
        df=df,
        metric_prefix="rmse_residual_px",
        ylabel="RMSE residual [px]",
        filename="rmse_difference_to_final.png",
    )

    plot_difference_to_final(
        df=df,
        metric_prefix="r2_pixel_displacement",
        ylabel="R² displacement [-]",
        filename="r2_difference_to_final.png",
    )

    print()
    print("=" * 80)
    print("SAMPLE SIZE SELECTION SUMMARY")
    print("=" * 80)
    print(f"Input: {INPUT_CSV}")
    print()

    if stable_rmse_n is not None:
        print(f"Stable sample size based on RMSE tolerance ({rmse_tol} px): {stable_rmse_n}")
    else:
        print("No stable sample size found for RMSE with current tolerance.")

    if stable_r2_n is not None:
        print(f"Stable sample size based on R² tolerance ({r2_tol}): {stable_r2_n}")
    else:
        print("No stable sample size found for R² with current tolerance.")

    print()
    print("Saved figures:")
    print(OUTPUT_DIR / "rmse_convergence_with_stability.png")
    print(OUTPUT_DIR / "r2_convergence_with_stability.png")
    print(OUTPUT_DIR / "max_residual_convergence_with_stability.png")
    print(OUTPUT_DIR / "rmse_difference_to_final.png")
    print(OUTPUT_DIR / "r2_difference_to_final.png")
    print()


if __name__ == "__main__":
    main()
