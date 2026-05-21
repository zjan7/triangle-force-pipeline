from __future__ import annotations

import json
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

from forces2 import ForceGenerator2  # noqa: E402
from trigrid import TriangleGrid  # noqa: E402
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
            tag = cv2.aruco.generateImageMarker(
                aruco_dict,
                marker_id,
                tag_size,
                borderBits=1,
            )
        else:
            tag = np.zeros((tag_size, tag_size), dtype=np.uint8)
            cv2.aruco.drawMarker(
                aruco_dict,
                marker_id,
                tag_size,
                tag,
                1,
            )

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
        ax.text(center_x, y0 - size * 0.15, nlabel, ha="center", va="top", fontsize=8,)

def save_force_plot(
    force_generator: ForceGenerator2,
    output_path: str | Path,
    grid_width: float,
    grid_height: float,
    resolution: float = 0.0025,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_range = np.arange(0, grid_width + resolution, resolution)
    y_range = np.arange(0, grid_height + resolution, resolution)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    normal_grid = np.zeros_like(x_grid, dtype=float)
    shear_x_grid = np.zeros_like(x_grid, dtype=float)
    shear_y_grid = np.zeros_like(x_grid, dtype=float)

    for row in range(x_grid.shape[0]):
        for col in range(x_grid.shape[1]):
            normal, shear_x, shear_y = force_generator(
                float(x_grid[row, col]),
                float(y_grid[row, col]),
            )

            normal_grid[row, col] = normal
            shear_x_grid[row, col] = shear_x
            shear_y_grid[row, col] = shear_y

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im1 = axes[0].pcolormesh(
        x_grid,
        y_grid,
        normal_grid,
        cmap="viridis",
        shading="auto",
    )
    axes[0].set_title("Normal force / pressure")
    axes[0].set_aspect("equal")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(
        x_grid,
        y_grid,
        shear_x_grid,
        cmap="RdBu_r",
        shading="auto",
    )
    axes[1].set_title("Shear force X")
    axes[1].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].pcolormesh(
        x_grid,
        y_grid,
        shear_y_grid,
        cmap="RdBu_r",
        shading="auto",
    )
    axes[2].set_title("Shear force Y")
    axes[2].set_aspect("equal")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_deformation_sample(
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "deformation_test",
    grid_width: float = 0.1,
    grid_height: float = 0.1,
    target_triangles: int = 200,
    marker_ids: list[int] | None = None,
    dpi: int = 300,
    show_plot: bool = False,
) -> dict[str, Any]:
  
    mpl.interactive(False)

    if marker_ids is None:
        marker_ids = [1, 2, 3]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_dir = output_dir / "arucoMarkers"
    marker_paths = generate_aruco_markers(output_dir=marker_dir, marker_ids=marker_ids,)

    module_area = (grid_width * grid_height) / target_triangles
    tris = TriangleGrid(grid_width, grid_height, module_area)

    print("Deformation grid")
    print("n_x:", tris.n_x)
    print("n_y:", tris.n_y)
    print("Aantal driehoeken:", tris.n_x * tris.n_y)
    print("Triangle area:", tris.t_area)

    normal_peak = 125000.0
    shear_peak = 2500.0
    force_scale = 0.4
    force_generator = ForceGenerator2(grid_width, grid_height, 0.03, 0.02, 0.03, smoothness=1.2, sphere_factor=1.2, normal_peak=normal_peak, shear_peak=shear_peak,)

    force_plot_path = output_dir / "force_plot.png"
    save_force_plot(
        force_generator=force_generator,
        output_path=force_plot_path,
        grid_width=grid_width,
        grid_height=grid_height,
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    forces_grid: list[list[list[float]]] = []
    force_matrix_rows: list[list[float]] = []
    y_force_rows: list[list[float]] = []

    triangle_id = 0

    for i in range(tris.n_x):
        forces_row = []

        for j in range(tris.n_y):
            center_x, center_y = tris.get_triangle_center(i, j)

            normal, shear_x, shear_y = force_generator(center_x, center_y)
            normal = force_scale * normal
            shear_x = force_scale * shear_x
            shear_y = force_scale * shear_y
            actual_normal = tris.t_area * normal
            actual_shear_x = tris.t_area * shear_x
            actual_shear_y = tris.t_area * shear_y

            solver_force_vector = [
                actual_normal,
                actual_shear_x,
                actual_shear_y,
            ]

            forces_row.append(solver_force_vector)

            y_force_rows.append(
                [
                    float(actual_normal),
                    float(actual_shear_x),
                    float(actual_shear_y),
                ]
            )

            force_matrix_rows.append(
                [
                    float(triangle_id),
                    float(i),
                    float(j),
                    float(center_x),
                    float(center_y),
                    float(normal),
                    float(shear_x),
                    float(shear_y),
                    float(actual_normal),
                    float(actual_shear_x),
                    float(actual_shear_y),
                ]
            )

            triangle_id += 1

        forces_grid.append(forces_row)

    force_matrix_full = np.array(force_matrix_rows, dtype=np.float64)
    y_forces = np.array(y_force_rows, dtype=np.float64)
    corners: list[list[list[np.ndarray]]] = []

    for i in tqdm.tqdm(range(tris.n_x), desc="Generating deformed triangles"):
        corners_row = []

        for j in tqdm.tqdm(range(tris.n_y), leave=False):
            force_vector = forces_grid[i][j]
            upside_down = ((i + j) % 2 == 0)

            c1, c2, c3 = bep.solve_module(
                force_vector,
                upside_down=upside_down,
            )

            center_x, center_y = tris.get_triangle_center(i, j)
            original_center = np.array([center_x, center_y, 0.0])

            c1 = c1 + original_center
            c2 = c2 + original_center
            c3 = c3 + original_center

            ax.fill(
                [c1[0], c2[0], c3[0]],
                [c1[1], c2[1], c3[1]],
                color="turquoise",
            )

            corners_row.append([c1, c2, c3])

        corners.append(corners_row)

    corners_array = np.array(corners, dtype=float)

    ax.set_aspect("equal")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    grid_plot_width = x_max - x_min
    marker_size = grid_plot_width * 0.12
    margin = marker_size * 0.45

    # Top marker: ID 1
    place_marker(
        ax,
        marker_paths[0],
        center_x=(x_min + x_max) / 2,
        center_y=y_max + margin + marker_size / 2,
        size=marker_size,
    )

    # Left marker: ID 2
    place_marker(
        ax,
        marker_paths[1],
        center_x=x_min - margin - marker_size / 2,
        center_y=(y_min + y_max) / 2,
        size=marker_size,
    )

    # Right marker: ID 3
    place_marker(
        ax,
        marker_paths[2],
        center_x=x_max + margin + marker_size / 2,
        center_y=(y_min + y_max) / 2,
        size=marker_size,
    )

    # Expand limits so markers are visible.
    ax.set_xlim(x_min - 2 * marker_size, x_max + 2 * marker_size)
    ax.set_ylim(y_min - marker_size, y_max + 2 * marker_size)

    ax.axis("equal")
    ax.axis("off")

    deformed_image_path = output_dir / "deformed_with_aruco.png"
    corners_path = output_dir / "deformed_corners.npy"
    force_matrix_path = output_dir / "force_matrix_full.npy"
    y_forces_path = output_dir / "y_forces.npy"
    metadata_path = output_dir / "deformation_metadata.json"

    fig.savefig(
        deformed_image_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.6,
    )

    if show_plot:
        plt.show()

    plt.close(fig)

    np.save(corners_path, corners_array)
    np.save(force_matrix_path, force_matrix_full)
    np.save(y_forces_path, y_forces)

    metadata = {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "target_triangles": target_triangles,
        "module_area": module_area,
        "n_x": int(tris.n_x),
        "n_y": int(tris.n_y),
        "n_triangles": int(tris.n_x * tris.n_y),
        "triangle_area": float(tris.t_area),
        "marker_ids": marker_ids,
        "normal_peak": float(normal_peak),
        "deformed_image": str(deformed_image_path),
        "deformed_corners": str(corners_path),
        "force_matrix_full": str(force_matrix_path),
        "y_forces": str(y_forces_path),
        "force_plot": str(force_plot_path),
        "force_matrix_full_columns": [
            "triangle_id",
            "i",
            "j",
            "center_x",
            "center_y",
            "normal_raw",
            "shear_x_raw",
            "shear_y_raw",
            "normal_force",
            "shear_force_x",
            "shear_force_y",
        ],
        "y_forces_columns": [
            "normal_force",
            "shear_force_x",
            "shear_force_y",
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved deformed image: {deformed_image_path}")
    print(f"Saved deformed corners: {corners_path}")
    print(f"Saved full force matrix: {force_matrix_path}")
    print(f"Saved PyTorch target matrix: {y_forces_path}")
    print(f"Saved force plot: {force_plot_path}")
    print(f"Saved metadata: {metadata_path}")

    return {
        "deformed_image_path": deformed_image_path,
        "corners_path": corners_path,
        "force_matrix_path": force_matrix_path,
        "y_forces_path": y_forces_path,
        "force_plot_path": force_plot_path,
        "metadata_path": metadata_path,
        "corners": corners_array,
        "force_matrix_full": force_matrix_full,
        "y_forces": y_forces,
        "metadata": metadata,
    }


if __name__ == "__main__":
    generate_deformation_sample(show_plot=True)