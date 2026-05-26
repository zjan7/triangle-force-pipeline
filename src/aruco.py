import matplotlib.pyplot as plt
import os
import cv2

aruco_type = "DICT_4X4_100"
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

output_dir = "arucoMarkers"
os.makedirs(output_dir, exist_ok=True)

marker_ids = [6, 7, 8]
tag_size = 300

for marker_id in marker_ids:
    tag = cv2.aruco.generateImageMarker(arucoDict, marker_id, tag_size, borderBits=1)
    filename = os.path.join(output_dir, f"{aruco_type}_{marker_id}.png")
    cv2.imwrite(filename, tag)

def place_marker(ax, image_path, center_x, center_y, size):
    img = mpimg.imread(image_path)
    x0 = center_x - size / 2
    x1 = center_x + size / 2
    y0 = center_y - size / 2
    y1 = center_y + size / 2
    ax.imshow(img, extent=[x0, x1, y0, y1], cmap="gray", zorder=10)
fig, ax = plt.subplots(figsize=(10, 10))
tri_grid_size = 10

for i in range(tri_grid_size * 2):
    for j in range(tri_grid_size):
        f_vec = forces[i][j]
        (c1, c2, c3) = bep.solve_module(f_vec)
        (tc_x, tc_y) = tris.get_triangle_center(i, j)
        original_center = [tc_x, tc_y, 0]
        c1 += original_center
        c2 += original_center
        c3 += original_center

        ax.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color='turquoise')


ax.set_aspect('equal')

x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

marker_size = (x_max - x_min) * 0.12
margin = marker_size * 0.25

place_marker(ax, "arucoMarkers/DICT_4X4_100_1.png",
             (x_min + x_max)/2, y_max + margin + marker_size/2, marker_size)

place_marker(ax, "arucoMarkers/DICT_4X4_100_2.png",
             x_min - margin - marker_size/2, (y_min + y_max)/2, marker_size)

place_marker(ax, "arucoMarkers/DICT_4X4_100_3.png",
             x_max + margin + marker_size/2, (y_min + y_max)/2, marker_size)

ax.set_xlim(x_min - 2*marker_size, x_max + 2*marker_size)
ax.set_ylim(y_min - marker_size, y_max + 2*marker_size)

plt.show()
