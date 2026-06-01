#(pipe)This code creates the reference image next code is deformation_generator.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external"

if str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))

from trigrid import TriangleGrid  
import bep2 as bep  

def generate_aruco_markers(
    output_dir: str | Path,
    marker_ids: list[int] | None = None,
    tag_size: int = 300,
) -> list[Path]:

    if marker_ids is None:
        marker_ids = [1, 2, 3]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aruco_type = "DICT_4X4_100"
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    marker_paths: list[Path] = []

    for marker_id in marker_ids:
        if hasattr(cv2.aruco, "generateImageMarker"):
            tag = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tag_size, borderBits=1,)
        else:
            tag = np.zeros((tag_size, tag_size), dtype=np.uint8)
            cv2.aruco.drawMarker(aruco_dict, marker_id, tag_size, tag, 1,)

        tag_path = output_dir / f"{aruco_type}_{marker_id}.png"
        cv2.imwrite(str(tag_path), tag)
        marker_paths.append(tag_path)

    return marker_paths


def place_marker(
    ax: plt.Axes,
    image_path: str | Path,
    center_x: float,
    center_y: float,
    size: float,
    label: str | None = None,
) -> None:
    image_path = Path(image_path)
    img = mpimg.imread(image_path)

    x0 = center_x - size / 2
    x1 = center_x + size / 2
    y0 = center_y - size / 2
    y1 = center_y + size / 2

    ax.imshow(img, extent=[x0, x1, y0, y1], cmap="gray", zorder=20,)

    if label is not None:
        ax.text(center_x, y0 - size * 0.15, label, ha="center", va="top", fontsize=8,)

def generate_reference(
    tris: TriangleGrid,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "reference",
    marker_ids: list[int] | None = None,
    dpi: int = 300,
    show_plot: bool = False,
) -> dict[str, Any]:
   
    mpl.interactive(False)

    if marker_ids is None:
        marker_ids = [1, 2, 3]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_width = tris.width
    grid_height = tris.height
    triangle_area = tris.t_area
    actual_triangles = tris.n_x * tris.n_y
    grid_cell_area = tris.t_area
    physical_triangle_area = getattr(tris, "physical_triangle_area", tris.t_area)

    marker_dir = output_dir / "arucoMarkers"
    marker_paths = generate_aruco_markers(output_dir=marker_dir, marker_ids=marker_ids,)

    print("Reference grid")
    print("n_x:", tris.n_x)
    print("n_y:", tris.n_y)
    print("Aantal driehoeken:", tris.n_x * tris.n_y)
    print("Triangle area:", tris.t_area)

    fig, ax = plt.subplots(figsize=(10, 10))

    zero_force = [0, 0, 0]

    corners: list[list[list[np.ndarray]]] = []
    reference_data: list[dict[str, Any]] = []
    reference_matrix_rows: list[list[float]] = []

    triangle_id = 0

    for i in tqdm.tqdm(range(tris.n_x), desc="Generating reference triangles"):
        corners_row = []

        for j in tqdm.tqdm(range(tris.n_y), leave=False):
            upside_down = ((i + j) % 2 == 0)

            c1, c2, c3 = bep.solve_module(zero_force, upside_down=upside_down,)

            tc_x, tc_y = tris.get_triangle_center(i, j)
            original_center = np.array([tc_x, tc_y, 0.0])

            c1 = c1 + original_center
            c2 = c2 + original_center
            c3 = c3 + original_center

            ax.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color="turquoise",)

            corners_row.append([c1, c2, c3])

            reference_data.append(
                {
                    "id": triangle_id,
                    "i": i,
                    "j": j,
                    "center_x": float(tc_x),
                    "center_y": float(tc_y),
                    "upside_down": bool(upside_down),
                    "corners": [
                        c1.tolist(),
                        c2.tolist(),
                        c3.tolist(),
                    ],
                }
            )

            # Full numerical reference matrix.
            #
            # Columns:
            # [
            #     triangle_id,
            #     i,
            #     j,
            #     center_x,
            #     center_y,
            #     upside_down,
            #     c1_x,
            #     c1_y,
            #     c1_z,
            #     c2_x,
            #     c2_y,
            #     c2_z,
            #     c3_x,
            #     c3_y,
            #     c3_z
            # ]
            reference_matrix_rows.append(
                [
                    float(triangle_id),
                    float(i),
                    float(j),
                    float(tc_x),
                    float(tc_y),
                    float(int(upside_down)),
                    float(c1[0]),
                    float(c1[1]),
                    float(c1[2]),
                    float(c2[0]),
                    float(c2[1]),
                    float(c2[2]),
                    float(c3[0]),
                    float(c3[1]),
                    float(c3[2]),
                ]
            )

            triangle_id += 1

        corners.append(corners_row)

    corners_array = np.array(corners, dtype=float)
    reference_matrix_full = np.array(reference_matrix_rows, dtype=np.float64)

    x_min = 0.0
    x_max = grid_width
    y_min = 0.0
    y_max = grid_height

    grid_plot_width = grid_width
    marker_size = grid_plot_width * 0.12
    margin = marker_size * 0.45

    # Top marker: ID 1
    place_marker(ax, marker_paths[0], center_x=(x_min + x_max) / 2, center_y=y_max + margin + marker_size / 2, size=marker_size,)

    # Left marker: ID 2
    place_marker(ax, marker_paths[1], center_x=x_min - margin - marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size,)

    # Right marker: ID 3
    place_marker(ax, marker_paths[2], center_x=x_max + margin + marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size,)

    # Expand limits so markers are visible.
    plot_margin = marker_size * 2.5
    plot_x_min = x_min - plot_margin
    plot_x_max = x_max + plot_margin
    plot_y_min = y_min - plot_margin
    plot_y_max = y_max + plot_margin

    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    reference_image_path = output_dir / "reference_zero_force_with_aruco.png"
    reference_corners_path = output_dir / "reference_zero_force_corners.npy"
    reference_data_path = output_dir / "reference_zero_force_data.npy"
    reference_matrix_path = output_dir / "reference_triangle_matrix_full.npy"
    metadata_path = output_dir / "reference_metadata.json"

    fig.savefig(reference_image_path, dpi=dpi, pad_inches=0,)

    if show_plot:
        plt.show()

    plt.close(fig)

    np.save(reference_corners_path, corners_array)
    np.save(reference_data_path, np.array(reference_data, dtype=object))
    np.save(reference_matrix_path, reference_matrix_full)

    metadata = {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "actual_triangles": int(actual_triangles),
        "grid_cell_area": float(grid_cell_area),
        "n_x": int(tris.n_x),
        "n_y": int(tris.n_y),
        "n_triangles": int(tris.n_x * tris.n_y),
        "marker_ids": marker_ids,
        "reference_image": str(reference_image_path),
        "reference_corners": str(reference_corners_path),
        "reference_data": str(reference_data_path),
        "reference_matrix_full": str(reference_matrix_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved image: {reference_image_path}")
    print(f"Saved corners: {reference_corners_path}")
    print(f"Saved metadata array: {reference_data_path}")
    print(f"Saved full reference matrix: {reference_matrix_path}")
    print(f"Saved metadata JSON: {metadata_path}")

    return {
        "reference_image_path": reference_image_path,
        "reference_corners_path": reference_corners_path,
        "reference_data_path": reference_data_path,
        "reference_matrix_path": reference_matrix_path,
        "metadata_path": metadata_path,
        "corners": corners_array,
        "reference_data": reference_data,
        "reference_matrix_full": reference_matrix_full,
        "metadata": metadata,
    }


if __name__ == "__main__":
    generate_reference(show_plot=True)