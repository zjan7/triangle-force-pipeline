import os
import cv2
import numpy as np
from forces2 import ForceGenerator2
from trigrid import TriangleGrid
import bep
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tqdm


def generate_aruco_markers():
    output_dir = "arucoMarkers"
    os.makedirs(output_dir, exist_ok=True)

    aruco_type = "DICT_4X4_100"
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    marker_ids = [1, 2, 3]
    tag_size = 300

    marker_paths = []

    for marker_id in marker_ids:
        tag = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tag_size, borderBits=1)
        tag_name = os.path.join(output_dir, f"{aruco_type}_{marker_id}.png")
        cv2.imwrite(tag_name, tag)
        marker_paths.append(tag_name)

    return marker_paths


def place_marker(ax, image_path, center_x, center_y, size, label=None):
    img = mpimg.imread(image_path)

    x0 = center_x - size / 2
    x1 = center_x + size / 2
    y0 = center_y - size / 2
    y1 = center_y + size / 2

    ax.imshow(img, extent=[x0, x1, y0, y1], cmap="gray", zorder=20)

    if label is not None:
        ax.text(center_x, y0 - size * 0.15, label, ha="center", va="top", fontsize=8)


def main():
    mpl.interactive(True)

    marker_paths = generate_aruco_markers()
    # Totale oppervlak van sensor grid
    grid_width = 0.1
    grid_height = 0.1
    target_triangles = 200

    # Eigenschappen driehoekig oppervlak aan de bovenkant
    module_area = (grid_width * grid_height) / target_triangles # 1 cm^2 modules
    tris = TriangleGrid(grid_width, grid_height, module_area)

    peak_normal = 5 / (0.01 * 0.01)  # N/m^2, dus 5 N / cm^2

    forceg = ForceGenerator2(0.1, 0.1, 0.03, 0.01, 0.02, normal_peak=peak_normal, shear_peak=peak_normal * 0.2)

    forceg.show()

    fig, ax = plt.subplots(figsize=(10, 10))

    forces = []

    for i in range(tris.n_x):
        forces_row = []

        for j in range(tris.n_y):
            tx, ty = tris.get_triangle_center(i, j)
            fx, fy, fz = forceg(tx, ty)

            # Convert pressure/load to actual force on one triangle
            force_vector = [tris.t_area * fx, tris.t_area * fy, tris.t_area * fz]

            forces_row.append(force_vector)

        forces.append(forces_row)

    corners = []

    for i in tqdm.tqdm(range(tris.n_x)):
        corners_row = []

        for j in tqdm.tqdm(range(tris.n_y), leave=False):
            f_vec = forces[i][j]

            c1, c2, c3 = bep.solve_module(f_vec, upside_down=((i + j) % 2 == 0))

            tc_x, tc_y = tris.get_triangle_center(i, j)
            original_center = np.array([tc_x, tc_y, 0])

            c1 = c1 + original_center
            c2 = c2 + original_center
            c3 = c3 + original_center

            ax.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color="turquoise")

            corners_row.append([c1, c2, c3])

        corners.append(corners_row)

    # Save all corner positions.
    # Shape: [triangle_x, triangle_y, corner_number, coordinate_xyz]
    np.save("hoeken_start.npy", np.array(corners))

    ax.set_aspect("equal")

    # Get current triangle-grid limits before adding markers
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    grid_plot_width = x_max - x_min
    marker_size = grid_plot_width * 0.12
    margin = marker_size * 0.45

    # Top marker: ID 1
    place_marker(ax, marker_paths[0], center_x=(x_min + x_max) / 2, center_y=y_max + margin + marker_size / 2, size=marker_size)

    # Left marker: ID 2
    place_marker(ax, marker_paths[1], center_x=x_min - margin - marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size)

    # Right marker: ID 3
    place_marker(ax, marker_paths[2], center_x=x_max + margin + marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size)

    # Expand limits so markers are visible
    ax.set_xlim(x_min - 2 * marker_size, x_max + 2 * marker_size)
    ax.set_ylim(y_min - marker_size, y_max + 2 * marker_size)

    ax.axis("equal")
    ax.axis("off")

    plt.savefig("triangle_grid_with_aruco.png", dpi=300, bbox_inches="tight")
    plt.show()

    input()


if __name__ == "__main__":
    main()