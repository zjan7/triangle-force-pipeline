#(pipe)This is the final triangle detection code for the loop next is dataset_writer.py
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
import json
from itertools import permutations

PROJECT_ROOT = Path(__file__).resolve().parents[1]

reference_path = PROJECT_ROOT / "outputs" / "reference" / "reference_zero_force_with_aruco.png"
deformed_path = PROJECT_ROOT / "outputs" / "aruco_warp_test" / "aligned_deformed.png"

output_dir = PROJECT_ROOT / "outputs" / "triangle_detection_test"
output_dir.mkdir(parents=True, exist_ok=True)

# In model 1.73 cm uit elkaar dus ongeveer 65.4 pixels.
min_area = 1000
max_area = 50000

matching_max_distance = 60

use_grid_ids = True
row_tolerance_px = 30


# Max distance between reference centroid and deformed centroid.



def load_image(path):
    # Dit laad de afbeelding in OpenCV.
    img = cv2.imread(str(path))

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return img


def show_image(img, title="", figsize=(8, 8), show=False):

    if not show:
        return

    plt.figure(figsize=figsize)

    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title(title)
    plt.axis("off")
    plt.show()


def wrap_angle_deg(angle):
    # Hoeken altijd in range [-180, 180].
    return (angle + 180) % 360 - 180
def clamp_rotation_to_clockwise_0_60(angle_deg):
    
    candidates = []

    for k in range(-3, 4):
        candidate = angle_deg + 120.0 * k
        candidate_wrapped = wrap_angle_deg(candidate)
        candidates.append(candidate_wrapped)

    valid = [c for c in candidates if 0.0 <= c <= 60.0]

    if len(valid) > 0:
        return min(valid, key=lambda c: abs(c - angle_deg))

    # If no equivalent angle is exactly inside the range, choose the closest
    # value to the allowed interval. This should only happen for noisy detections.
    def distance_to_allowed_range(c):
        if c < 0.0:
            return abs(c - 0.0)
        if c > 60.0:
            return abs(c - 60.0)
        return 0.0

    closest = min(candidates, key=distance_to_allowed_range)

    return float(np.clip(closest, 0.0, 60.0))


def optimal_rotation_from_vertices(ref_vertices, def_vertices):
    
    ref = np.asarray(ref_vertices, dtype=np.float64)
    deformed = np.asarray(def_vertices, dtype=np.float64)

    ref_center = ref.mean(axis=0)
    def_center = deformed.mean(axis=0)

    ref_centered = ref - ref_center
    def_centered_original = deformed - def_center

    best_rotation = 0.0
    best_error = np.inf

    for perm in permutations(range(3)):
        def_centered = def_centered_original[list(perm)]

        # Estimate 2D rotation angle from ref_centered to def_centered.
        # In image coordinates, positive angle is visually clockwise.
        numerator = np.sum(
            ref_centered[:, 0] * def_centered[:, 1]
            - ref_centered[:, 1] * def_centered[:, 0]
        )

        denominator = np.sum(
            ref_centered[:, 0] * def_centered[:, 0]
            + ref_centered[:, 1] * def_centered[:, 1]
        )

        raw_angle = np.degrees(np.arctan2(numerator, denominator))
        raw_angle = wrap_angle_deg(raw_angle)

        corrected_angle = clamp_rotation_to_clockwise_0_60(raw_angle)

        theta = np.radians(corrected_angle)

        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )

        predicted = ref_centered @ rotation_matrix.T
        error = np.sqrt(np.mean(np.sum((predicted - def_centered) ** 2, axis=1)))

        if error < best_error:
            best_error = error
            best_rotation = corrected_angle

    return float(best_rotation), float(best_error)

def triangle_angle(vertices):
    pts = np.array(vertices, dtype=np.float32)

    edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])]

    edge_info = []

    for p1, p2 in edges:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        angle_wrapped = wrap_angle_deg(angle)

        horizontal_score = min(abs(angle_wrapped), abs(abs(angle_wrapped) - 180))
        edge_info.append((horizontal_score, p1, p2))

    if len(edge_info) == 0:
        return 0.0

    _, p1, p2 = min(edge_info, key=lambda item: item[0])

    if p2[0] < p1[0]:
        p1, p2 = p2, p1

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle_deg = np.degrees(np.arctan2(dy, dx))

    return angle_deg


def counterclockwise_angle_difference(angle_ref, angle_def, zero_tolerance_deg=1.0):
    d_angle = (angle_ref - angle_def) % 360

    # Treat tiny numerical noise near 0 or 360 as zero.
    if d_angle <= zero_tolerance_deg:
        d_angle = 0.0

    if d_angle >= 360 - zero_tolerance_deg:
        d_angle = 0.0

    return d_angle


def assign_ids_by_grid(triangles, row_tolerance_px=30):
    if len(triangles) == 0:
        return triangles

    # Sort all y-values into rows.
    triangles_sorted_y = sorted(triangles, key=lambda t: t["centroid"][1])
    rows = []

    for tri in triangles_sorted_y:
        cx, cy = tri["centroid"]
        placed = False

        for row in rows:
            row_y_values = [item["centroid"][1] for item in row]
            row_mean_y = np.mean(row_y_values)

            # Compare whether it fits in the row.
            if abs(cy - row_mean_y) <= row_tolerance_px:
                row.append(tri)
                placed = True
                break

        if not placed:
            rows.append([tri])

    # Rows sorted from top to bottom.
    rows = sorted(rows, key=lambda row: np.mean([item["centroid"][1] for item in row]))

    output = []
    current_id = 0

    for row_index, row in enumerate(rows):
        # Inside each row, sort left to right.
        row_sorted = sorted(row, key=lambda t: t["centroid"][0])

        for col_index, tri in enumerate(row_sorted):
            tri["row"] = row_index
            tri["col"] = col_index
            tri["id"] = current_id

            output.append(tri)
            current_id += 1

    return output


def assign_ids_simple_sort(triangles):
    triangles = sorted(triangles, key=lambda t: (t["centroid"][1], t["centroid"][0]))

    for idx, tri in enumerate(triangles):
        tri["id"] = idx
        tri["row"] = None
        tri["col"] = None

    return triangles


def draw_detected_triangles(image, triangles, draw_ids=True):
    # Teken de driehoeken.
    annotated = image.copy()

    for tri in triangles:
        cx, cy = tri["centroid"]
        vertices = np.array(tri["vertices"], dtype=np.int32)

        # Green contour.
        cv2.drawContours(annotated, [vertices], -1, (0, 255, 0), 2)

        # Red centroid.
        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        if draw_ids and tri["id"] is not None:
            cv2.putText(
                annotated,
                str(tri["id"]),
                (int(cx) + 5, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return annotated


def detect_triangles(image, min_area=40, max_area=2000, assign_ids=True, use_grid_ids=True, row_tolerance_px=30,):
    #originele detectiefunctie
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,)[1]

    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5,)

    binary = cv2.bitwise_or(binary_otsu, binary_adaptive)

    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,)

    triangles = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        if len(contour) < 5:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / hull_area

        if solidity < 0.65:
            continue

        enclosing_area, enclosing_triangle = cv2.minEnclosingTriangle(hull)

        if enclosing_triangle is None:
            continue

        if enclosing_area <= 0:
            continue

        triangle_fit = area / enclosing_area

        if triangle_fit < 0.55:
            continue

        vertices = enclosing_triangle.reshape(3, 2)

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        angle_deg = triangle_angle(vertices)

        triangles.append(
            {
                "id": None,
                "row": None,
                "col": None,
                "centroid": (cx, cy),
                "angle_deg": angle_deg,
                "area_px2": area,
                "contour": contour,
                "vertices": vertices,
                "triangle_fit": triangle_fit,
                "solidity": solidity,
            }
        )

    if assign_ids:
        if use_grid_ids:
            triangles = assign_ids_by_grid(triangles, row_tolerance_px=row_tolerance_px,)
        else:
            triangles = assign_ids_simple_sort(triangles)

    annotated = draw_detected_triangles(image, triangles, draw_ids=assign_ids)

    return triangles, annotated, binary


def match_deformed_to_reference(reference_triangles, deformed_triangles, max_distance=60):
    # Match deformed triangles to reference triangles.
    if len(reference_triangles) == 0:
        return []

    if len(deformed_triangles) == 0:
        return []

    ref_centroids = np.array([tri["centroid"] for tri in reference_triangles], dtype=np.float32,)

    def_centroids = np.array([tri["centroid"] for tri in deformed_triangles], dtype=np.float32,)

    tree = cKDTree(def_centroids)

    matches = []
    used_deformed_indices = set()

    for ref_index, ref_tri in enumerate(reference_triangles):
        distance, def_index = tree.query(ref_centroids[ref_index])

        if distance > max_distance:
            continue

        if def_index in used_deformed_indices:
            continue

        used_deformed_indices.add(def_index)

        def_tri = deformed_triangles[def_index]

        cx_ref, cy_ref = ref_tri["centroid"]
        cx_def, cy_def = def_tri["centroid"]

        dx = cx_def - cx_ref
        dy = cy_def - cy_ref

        displacement = np.sqrt(dx**2 + dy**2)

        d_angle, rotation_fit_error_px = optimal_rotation_from_vertices(ref_tri["vertices"], def_tri["vertices"],)

        matches.append(
            {
                "ID": ref_tri["id"],
                "row": ref_tri["row"],
                "col": ref_tri["col"],
                "cx_ref": cx_ref,
                "cy_ref": cy_ref,
                "angle_ref_deg": ref_tri["angle_deg"],
                "area_ref_px2": ref_tri["area_px2"],
                "cx_def": cx_def,
                "cy_def": cy_def,
                "angle_def_deg": def_tri["angle_deg"],
                "area_def_px2": def_tri["area_px2"],
                "dx_px": dx,
                "dy_px": dy,
                "displacement_px": displacement,
                "rotation_change_deg": d_angle,
                "rotation_fit_error_px": rotation_fit_error_px,
                "matching_distance_px": distance,
                "reference_triangle_fit": ref_tri.get("triangle_fit", None),
                "deformed_triangle_fit": def_tri.get("triangle_fit", None),
                "reference_solidity": ref_tri.get("solidity", None),
                "deformed_solidity": def_tri.get("solidity", None),
                "deformed_internal_index": def_index,
            }
        )

    matches = sorted(matches, key=lambda m: m["ID"])

    return matches


def draw_matched_deformed_ids(image, matches):
    # Teken de IDs op de gematchte deformed driehoeken.
    annotated = image.copy()

    for m in matches:
        cx = m["cx_def"]
        cy = m["cy_def"]

        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.putText(annotated, str(m["ID"]), (int(cx) + 5, int(cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA,)

    return annotated


def draw_displacement_arrows(image, matches, scale=1.0):
    annotated = image.copy()

    for m in matches:
        start = (int(round(m["cx_ref"])), int(round(m["cy_ref"])))

        end = (int(round(m["cx_ref"] + scale * m["dx_px"])), int(round(m["cy_ref"] + scale * m["dy_px"])),)

        cv2.arrowedLine(annotated, start, end, (0, 0, 255), 2, tipLength=0.25,)

        cv2.circle(annotated, start, 3, (255, 0, 0), -1,)

    return annotated


def make_side_by_side(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = max(h1, h2)

    def pad_to_height(img, target_h):
        h, w = img.shape[:2]

        if h == target_h:
            return img

        pad_bottom = target_h - h

        return cv2.copyMakeBorder(img, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255),)

    img1_padded = pad_to_height(img1, target_h)
    img2_padded = pad_to_height(img2, target_h)

    return np.hstack([img1_padded, img2_padded])


def print_detection_summary(reference_triangles, deformed_triangles, matches):
    print("Detection summary")
    print("-----------------")
    print(f"Reference triangles detected: {len(reference_triangles)}")
    print(f"Deformed triangles detected:  {len(deformed_triangles)}")
    print(f"Matched triangles:            {len(matches)}")
    print()

    if len(reference_triangles) != len(deformed_triangles):
        print("Warning:")
        print("The reference and deformed image do not have the same number of detected triangles.")
        print("This can indicate detection errors, occlusions, or false positives.")
        print()

    unmatched_reference = len(reference_triangles) - len(matches)

    if unmatched_reference > 0:
        print(f"Unmatched reference triangles: {unmatched_reference}")
        print("Try increasing matching_max_distance if the triangles are visibly close but not accepted.")
        print()


def make_json_safe(obj):
    
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)

    return obj


def build_matrices_from_matches(matches):
    
    triangle_matrix_full = np.array(
        [
            [
                m["ID"],
                m["row"],
                m["col"],
                m["cx_ref"],
                m["cy_ref"],
                m["cx_def"],
                m["cy_def"],
                m["dx_px"],
                m["dy_px"],
                m["displacement_px"],
                m["angle_ref_deg"],
                m["angle_def_deg"],
                m["rotation_change_deg"],
                m["area_ref_px2"],
                m["area_def_px2"],
                m["matching_distance_px"],
            ]
            for m in matches
        ],
        dtype=float,
    )

    X_displacements = np.array(
        [
            [
                m["cx_ref"],
                m["cy_ref"],
                m["dx_px"],
                m["dy_px"],
                m["rotation_change_deg"],
            ]
            for m in matches
        ],
        dtype=float,
    )

    return triangle_matrix_full, X_displacements

def estimate_dynamic_thresholds(reference_triangles):
    areas = np.array([tri["area_px2"] for tri in reference_triangles], dtype=float)
    centroids = np.array([tri["centroid"] for tri in reference_triangles], dtype=float)

    median_area = float(np.median(areas))

    tree = cKDTree(centroids)
    distances, _ = tree.query(centroids, k=2)
    nearest_distances = distances[:, 1]

    median_spacing = float(np.median(nearest_distances))

    dynamic_min_area = max(10.0, 0.25 * median_area)
    dynamic_max_area = 3.00 * median_area
    dynamic_row_tolerance_px = max(5.0, 0.45 * median_spacing)
    dynamic_matching_max_distance = max(10.0, 0.75 * median_spacing)

    return {
        "median_area": median_area,
        "median_spacing": median_spacing,
        "min_area": dynamic_min_area,
        "max_area": dynamic_max_area,
        "row_tolerance_px": dynamic_row_tolerance_px,
        "matching_max_distance": dynamic_matching_max_distance,
    }
def run_triangle_detection(show=False):
    reference_img = load_image(reference_path)
    deformed_img = load_image(deformed_path)

    show_image(reference_img, "Reference image", show=show)
    show_image(deformed_img, "Deformed / warped image", show=show)

    # Reference image gets stable IDs.
    reference_triangles_rough, _, _ = detect_triangles(reference_img, min_area=10, max_area=1000000, assign_ids=False, use_grid_ids=False, row_tolerance_px=30,)

    dynamic = estimate_dynamic_thresholds(reference_triangles_rough)

    print("Dynamic thresholds")
    print("------------------")
    print(f"median area:       {dynamic['median_area']:.2f} px²")
    print(f"median spacing:    {dynamic['median_spacing']:.2f} px")
    print(f"min_area:          {dynamic['min_area']:.2f}")
    print(f"max_area:          {dynamic['max_area']:.2f}")
    print(f"row_tolerance_px:  {dynamic['row_tolerance_px']:.2f}")
    print(f"matching_distance: {dynamic['matching_max_distance']:.2f}")
    print()

# Final reference detection with dynamic thresholds
    reference_triangles, reference_annotated, reference_binary = detect_triangles(reference_img, min_area=dynamic["min_area"], max_area=dynamic["max_area"], assign_ids=True, use_grid_ids=use_grid_ids, row_tolerance_px=dynamic["row_tolerance_px"],)

    # Deformed image is detected, but its own IDs are not used.
    deformed_triangles, deformed_detected_annotated, deformed_binary = detect_triangles(deformed_img, min_area=dynamic["min_area"], max_area=dynamic["max_area"], assign_ids=False, use_grid_ids=False, row_tolerance_px=dynamic["row_tolerance_px"],)

    print(f"Reference triangles detected: {len(reference_triangles)}")
    print(f"Deformed triangles detected:  {len(deformed_triangles)}")

    show_image(reference_binary, "Reference binary mask", show=show)
    show_image(deformed_binary, "Deformed binary mask", show=show)

    show_image(reference_annotated, "Reference annotated with stable IDs", show=show)
    show_image(deformed_detected_annotated, "Deformed detected triangles, no IDs", show=show)

    # Final matching from deformed to reference IDs.
    matches = match_deformed_to_reference(reference_triangles, deformed_triangles, max_distance=dynamic["matching_max_distance"],)

    print_detection_summary(reference_triangles, deformed_triangles, matches)

    # Strict dataset acceptance rules.
    if len(reference_triangles) == 0:
        raise RuntimeError(
            "Triangle detection failed: no reference triangles were detected. "
            "This sample must be rejected."
        )

    if len(deformed_triangles) == 0:
        raise RuntimeError(
            "Triangle detection failed: no deformed triangles were detected. "
            "This sample must be rejected."
        )

    if len(matches) != len(reference_triangles):
        raise RuntimeError(
            "Triangle matching failed. "
            f"Reference triangles detected: {len(reference_triangles)}. "
            f"Deformed triangles detected: {len(deformed_triangles)}. "
            f"Matched triangles: {len(matches)}. "
            "This sample must be rejected."
        )

    df = pd.DataFrame(matches)

    output_columns = [
        "ID",
        "cx_ref",
        "cy_ref",
        "cx_def",
        "cy_def",
        "dx_px",
        "dy_px",
        "rotation_change_deg",
    ]

    if len(df) > 0:
        df = df[output_columns]

        df = df.rename(
            columns={
                "cx_ref": "cx_original",
                "cy_ref": "cy_original",
                "cx_def": "cx_new",
                "cy_def": "cy_new",
                "dx_px": "dx",
                "dy_px": "dy",
                "rotation_change_deg": "rotation_deg",
            }
        )

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        if show:
            print(df)
    else:
        print("No matches found.")
        print("Try increasing matching_max_distance.")

    deformed_matched_ids = draw_matched_deformed_ids(deformed_img, matches)
    displacement_img = draw_displacement_arrows(reference_img, matches)

    side_by_side = make_side_by_side(reference_annotated, deformed_matched_ids)

    show_image(deformed_matched_ids, "Deformed image with inherited reference IDs", show=show)
    show_image(displacement_img, "Displacement arrows", show=show)
    show_image(side_by_side, "Reference IDs vs deformed inherited IDs", figsize=(14, 8), show=show)

    csv_path = output_dir / "triangle_matrix.csv"
    json_path = output_dir / "triangle_analysis.json"

    reference_annotated_path = output_dir / "reference_annotated_stable_ids.png"
    deformed_matched_path = output_dir / "deformed_inherited_reference_ids.png"
    deformed_detected_path = output_dir / "deformed_detected_no_ids.png"
    reference_binary_path = output_dir / "reference_binary_mask.png"
    deformed_binary_path = output_dir / "deformed_binary_mask.png"
    displacement_path = output_dir / "displacement_arrows.png"
    side_by_side_path = output_dir / "side_by_side_reference_vs_deformed.png"

    triangle_matrix_full_path = output_dir / "triangle_matrix_full.npy"
    X_displacements_path = output_dir / "X_displacements.npy"
    metadata_path = output_dir / "triangle_detection_metadata.json"

    triangle_matrix_full, X_displacements = build_matrices_from_matches(matches)

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(matches), f, indent=4)

    np.save(triangle_matrix_full_path, triangle_matrix_full)
    np.save(X_displacements_path, X_displacements)

    cv2.imwrite(str(reference_annotated_path), reference_annotated)
    cv2.imwrite(str(deformed_matched_path), deformed_matched_ids)
    cv2.imwrite(str(deformed_detected_path), deformed_detected_annotated)
    cv2.imwrite(str(reference_binary_path), reference_binary)
    cv2.imwrite(str(deformed_binary_path), deformed_binary)
    cv2.imwrite(str(displacement_path), displacement_img)
    cv2.imwrite(str(side_by_side_path), side_by_side)

    metadata = {
        "reference_path": str(reference_path),
        "deformed_path": str(deformed_path),
        "output_dir": str(output_dir),
        "min_area": min_area,
        "max_area": max_area,
        "matching_max_distance": matching_max_distance,
        "use_grid_ids": use_grid_ids,
        "row_tolerance_px": row_tolerance_px,
        "reference_triangles_detected": len(reference_triangles),
        "deformed_triangles_detected": len(deformed_triangles),
        "matched_triangles": len(matches),
        "success": True,
        "X_displacements_path": str(X_displacements_path),
        "triangle_matrix_full_path": str(triangle_matrix_full_path),
        "X_displacements_columns": [
            "x_ref_px",
            "y_ref_px",
            "dx_px",
            "dy_px",
            "rotation_deg",
        ],
        "triangle_matrix_full_columns": [
            "ID",
            "row",
            "col",
            "cx_ref",
            "cy_ref",
            "cx_def",
            "cy_def",
            "dx_px",
            "dy_px",
            "displacement_px",
            "angle_ref_deg",
            "angle_def_deg",
            "rotation_change_deg",
            "area_ref_px2",
            "area_def_px2",
            "matching_distance_px",
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Saved outputs to:")
    print(output_dir)
    print()
    print("Files:")
    print(csv_path)
    print(json_path)
    print(reference_annotated_path)
    print(deformed_matched_path)
    print(deformed_detected_path)
    print(reference_binary_path)
    print(deformed_binary_path)
    print(displacement_path)
    print(side_by_side_path)
    print(triangle_matrix_full_path)
    print(X_displacements_path)
    print(metadata_path)

    return {
        "success": True,
        "reference_triangles": reference_triangles,
        "deformed_triangles": deformed_triangles,
        "matches": matches,
        "dataframe": df,
        "triangle_matrix_full": triangle_matrix_full,
        "X_displacements": X_displacements,
        "output_dir": output_dir,
        "triangle_matrix_full_path": triangle_matrix_full_path,
        "X_displacements_path": X_displacements_path,
        "metadata_path": metadata_path,
    }


if __name__ == "__main__":
    run_triangle_detection(show=False)