from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external"

if str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))

import bep2  # noqa: E402


def equilateral_triangle_area_from_radius(radius: float) -> float:
    if radius <= 0:
        raise ValueError("radius must be positive.")

    return (3.0 * math.sqrt(3.0) / 4.0) * radius**2


def get_triangle_area_from_bep2() -> float:
    return equilateral_triangle_area_from_radius(float(bep2.l))


def grid_size_from_bep2_triangle_area(
    target_triangles: int,
    aspect_ratio: float = 1.0,
    packing_factor: float = 2.5,
) -> tuple[float, float, float]:
    physical_triangle_area = get_triangle_area_from_bep2()

    if target_triangles <= 0:
        raise ValueError("target_triangles must be positive.")

    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive.")

    if packing_factor <= 0:
        raise ValueError("packing_factor must be positive.")

    grid_cell_area = physical_triangle_area *packing_factor

    grid_area = target_triangles * grid_cell_area

    grid_width = math.sqrt(grid_area * aspect_ratio)
    grid_height = math.sqrt(grid_area / aspect_ratio)

    return grid_width, grid_height, physical_triangle_area, grid_cell_area   


