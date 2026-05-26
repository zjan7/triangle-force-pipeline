#this code isn't used
from trigrid import TriangleGrid
import bep

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tqdm


def make_aruco_marker(marker_id, tag_size=300):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    if hasattr(cv2.aruco, "generateImageMarker"):
        tag = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tag_size, borderBits=1)
    else:
        tag = np.zeros((tag_size, tag_size), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, marker_id, tag_size, tag, 1)

    return tag


def add_marker_axis(fig, position, marker_img, title_text):
    ax_marker = fig.add_axes(position)
    ax_marker.imshow(marker_img, cmap="gray")
    ax_marker.set_title(title_text, fontsize=10)
    ax_marker.axis("off")
    return ax_marker


def main():
    mpl.interactive(True)
    tri_grid_size = 10
    tris = TriangleGrid(0.0001, tri_grid_size)
    triangle_color = "#00A6A6"
    marker_top = make_aruco_marker(1, 300)
    marker_left = make_aruco_marker(2, 300)
    marker_right = make_aruco_marker(3, 300)
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor("#e6e6e6")

    ax = fig.add_axes([0.18, 0.18, 0.64, 0.64])
    ax.set_facecolor("#e6e6e6")

    add_marker_axis(fig, [0.42, 0.84, 0.16, 0.16], marker_top, "ID 1")
    add_marker_axis(fig, [0.01, 0.42, 0.16, 0.16], marker_left, "ID 2")
    add_marker_axis(fig, [0.83, 0.42, 0.16, 0.16], marker_right, "ID 3")
    zero_force = (0.0, 0.0, 0.0)

    for i in tqdm.tqdm(range(tri_grid_size * 2)):
        for j in tqdm.tqdm(range(tri_grid_size), leave=False):
            c1, c2, c3 = bep.solve_module(zero_force)

            tc_x, tc_y = tris.get_triangle_center(i, j)
            original_center = np.array([tc_x, tc_y, 0.0])

            c1 = c1 + original_center
            c2 = c2 + original_center
            c3 = c3 + original_center

            ax.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color= "#00A6A6")
    

    ax.set_aspect("equal")
    ax.axis("off")


    plt.savefig("reference_zero_force_with_aruco.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved: reference_zero_force_with_aruco.png")


if __name__ == "__main__":
    main()
