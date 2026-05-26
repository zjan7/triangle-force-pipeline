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
    tri_grid_size = 10
    tris = TriangleGrid(0.0001, tri_grid_size)
    fig, ax = plt.subplots(figsize=(10, 10))
    zero_force = (0.0, 0.0, 0.0)

    # Draw triangles using the SAME geometry pipeline as main.py,
    # but now with zero force instead of real forces
    for i in tqdm.tqdm(range(tri_grid_size * 2)):
        for j in tqdm.tqdm(range(tri_grid_size), leave=False):
            c1, c2, c3 = bep.solve_module(zero_force)

            tc_x, tc_y = tris.get_triangle_center(i, j)
            original_center = np.array([tc_x, tc_y, 0.0])

            c1 = c1 + original_center
            c2 = c2 + original_center
            c3 = c3 + original_center

            ax.fill(
                [c1[0], c2[0], c3[0]],
                [c1[1], c2[1], c3[1]],
                color="#00A6A6"
            )

    ax.set_aspect("equal")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    grid_width = x_max - x_min
    marker_size = grid_width * 0.12
    margin = marker_size * 0.45

    # Top marker: ID 1
    place_marker(ax, marker_paths[0], center_x=(x_min + x_max) / 2, center_y=y_max + margin + marker_size / 2, size=marker_size)

    # Left marker: ID 2
    place_marker(ax, marker_paths[1], center_x=x_min - margin - marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size)

    # Right marker: ID 3
    place_marker(ax, marker_paths[2], center_x=x_max + margin + marker_size / 2, center_y=(y_min + y_max) / 2, size=marker_size)

    # SAME axis expansion as main.py
    ax.set_xlim(x_min - 2 * marker_size, x_max + 2 * marker_size)
    ax.set_ylim(y_min - marker_size, y_max + 2 * marker_size)

    ax.axis("equal")
    ax.axis("off")

    plt.savefig("reference_zero_force_with_aruco.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved: reference_zero_force_with_aruco.png")

    input()


if __name__ == "__main__":
    main()