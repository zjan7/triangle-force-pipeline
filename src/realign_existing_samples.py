from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.force_alignment import align_forces_to_opencv_order


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_sample_dirs(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> list[Path]:
    accepted_samples_dir = Path(accepted_samples_dir)

    sample_dirs = [
        path for path in accepted_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    ]

    return sorted(sample_dirs, key=lambda p: int(p.name.split("_")[1]))


def realign_existing_samples(
    accepted_samples_dir: str | Path = PROJECT_ROOT / "outputs" / "accepted_samples",
) -> None:
    sample_dirs = find_sample_dirs(accepted_samples_dir)

    print()
    print("=" * 80)
    print("REALIGN EXISTING ACCEPTED SAMPLES")
    print("=" * 80)
    print(f"Found samples: {len(sample_dirs)}")
    print()

    for sample_dir in sample_dirs:
        print("-" * 80)
        print(f"Processing {sample_dir.name}")

        raw_dir = sample_dir / "raw"
        pytorch_dir = sample_dir / "pytorch"
        full_matrices_dir = sample_dir / "full_matrices"

        reference_corners_path = raw_dir / "reference_zero_force_corners.npy"
        triangle_matrix_full_path = full_matrices_dir / "triangle_matrix_full.npy"
        force_matrix_full_path = full_matrices_dir / "force_matrix_full.npy"

        y_path = pytorch_dir / "y_forces.npy"
        y_backup_path = pytorch_dir / "y_forces_before_opencv_alignment.npy"

        if not reference_corners_path.exists():
            raise FileNotFoundError(f"Missing: {reference_corners_path}")

        if not triangle_matrix_full_path.exists():
            raise FileNotFoundError(f"Missing: {triangle_matrix_full_path}")

        if not force_matrix_full_path.exists():
            raise FileNotFoundError(
                f"Missing: {force_matrix_full_path}. "
                "This is needed to recover original generated-order forces safely."
            )

        if y_path.exists() and not y_backup_path.exists():
            original_y = np.load(y_path)
            np.save(y_backup_path, original_y)

        alignment_result = align_forces_to_opencv_order(
            reference_corners_path=reference_corners_path,
            triangle_matrix_full_path=triangle_matrix_full_path,
            force_matrix_full_path=force_matrix_full_path,
            output_dir=full_matrices_dir,
            mean_distance_threshold_px=1.0,
            max_distance_threshold_px=2.0,
        )

        y_reordered = alignment_result["y_forces_reordered"]
        np.save(y_path, y_reordered)

        metadata_path = sample_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        else:
            metadata = {}

        metadata["force_alignment"] = alignment_result["metadata"]
        metadata["important_note"] = (
            "Existing sample was realigned. y_forces.npy is now reordered to match "
            "X_displacements.npy/OpenCV triangle order."
        )

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        print(
            "Aligned force rows. Mean/max mapping distance: "
            f"{alignment_result['metadata']['mean_reference_mapping_distance_px']:.3f} px / "
            f"{alignment_result['metadata']['max_reference_mapping_distance_px']:.3f} px"
        )

    print()
    print("=" * 80)
    print("REALIGNMENT FINISHED")
    print("=" * 80)
    print()


if __name__ == "__main__":
    realign_existing_samples()
