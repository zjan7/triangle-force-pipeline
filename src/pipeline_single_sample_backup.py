#Deze code is enkel als test om veranderingen te proberen met 1 sample
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.reference_generator import generate_reference
from src.deformation_generator import generate_deformation_sample
from src.aruco_warp import warp_deformed_to_reference
from src.triangle_detection import run_triangle_detection
from src.dataset_writer import write_accepted_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_one_sample(
    sample_id: int = 1,
    grid_width: float = 0.1,
    grid_height: float = 0.1,
    target_triangles: int = 200,
) -> dict[str, Any]:
    print()
    print("=" * 80)
    print(f"Running sample {sample_id:06d}")
    print("=" * 80)
    reference_result = generate_reference(output_dir=PROJECT_ROOT / "outputs" / "reference", grid_width=grid_width, grid_height=grid_height, target_triangles=target_triangles, show_plot=False,)

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
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "reference_result": reference_result,
        "deformation_result": deformation_result,
        "aruco_result": aruco_result,
        "triangle_result": triangle_result,
    }


if __name__ == "__main__":
    run_one_sample(sample_id=1)