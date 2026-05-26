#this code isn't used
from trigrid import TriangleGrid
import bep

import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
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
        if hasattr(cv2.aruco, "generateImageMarker"):
            tag = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tag_size, borderBits=1)
        else:
            tag = np.zeros((tag_size, tag_size), dtype=np.uint8)
            cv2.aruco.drawMarker(aruco_dict, marker_id, tag_size, tag, 1)

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

    grid_width = 0.1
    grid_height = 0.1
    target_triangles = 200

    module_area = (grid_width * grid_height) / target_triangles
    tris = TriangleGrid(grid_width, grid_height, module_area)

    print("n_x:", tris.n_x)
    print("n_y:", tris.n_y)
    print("Aantal driehoeken:", tris.n_x * tris.n_y)
    print("Triangle area:", tris.t_area)
   
    fig, ax = plt.subplots(figsize=(10, 10))

    # Zero-force reference
    zero_force = [0, 0, 0]

    corners = []
    reference_data = []

    triangle_id = 0

    for i in tqdm.tqdm(range(tris.n_x)):
        corners_row = []

        for j in tqdm.tqdm(range(tris.n_y), leave=False):
            upside_down = ((i + j) % 2 == 0)

            c1, c2, c3 = bep.solve_module(zero_force, upside_down=upside_down)

            tc_x, tc_y = tris.get_triangle_center(i, j)
            original_center = np.array([tc_x, tc_y, 0.0])

            c1 = c1 + original_center
            c2 = c2 + original_center
            c3 = c3 + original_center

            ax.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color="turquoise")

            corners_row.append([c1, c2, c3])

            # Referentie-info voor latere beeldanalyse
            reference_data.append({
                "id": triangle_id,
                "i": i,
                "j": j,
                "center_x": tc_x,
                "center_y": tc_y,
                "upside_down": upside_down,
                "corners": [c1, c2, c3]
            })

            triangle_id += 1

        corners.append(corners_row)

    # Hoekpunten opslaan voor latere vergelijking
    np.save("reference_zero_force_corners.npy", np.array(corners, dtype=float))

    # Extra metadata opslaan als object-array
    np.save("reference_zero_force_data.npy", np.array(reference_data, dtype=object))

    ax.set_aspect("equal")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    plot_width = x_max - x_min
    marker_size = plot_width * 0.12
    margin = marker_size * 0.45

    # Top marker: ID 1
    place_marker(ax, marker_paths[0], center_x=(x_min + x_max) / 2, center_y=y_max + margin + marker_size / 2, size=marker_size)

    # Left marker: ID 2
    place_marker(ax, marker_paths[1], center_x=x_min - margin - marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size)

    # Right marker: ID 3
    place_marker(ax, marker_paths[2], center_x=x_max + margin + marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size)

    # Limits uitbreiden zodat markers zichtbaar zijn
    ax.set_xlim(x_min - 2 * marker_size, x_max + 2 * marker_size)
    ax.set_ylim(y_min - marker_size, y_max + 2 * marker_size)

    ax.axis("equal")
    ax.axis("off")

    output_name = "reference_zero_force_with_aruco.png"
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved image: {output_name}")
    print("Saved corners: reference_zero_force_corners.npy")
    print("Saved metadata: reference_zero_force_data.npy")

    input()


if __name__ == "__main__":
    main()