#code voor testversie Sweder, als de driehoeken niet worden gevonden: ga naar line 394 en verander de waardes, volgende code is main_run_real_visual_pipeline.py
from __future__ import annotations

import json
from pathlib import Path
from itertools import permutations
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.aruco_warp import warp_deformed_to_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image


def make_json_safe(obj: Any) -> Any:
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


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def clamp_rotation_to_clockwise_0_60(angle_deg: float) -> float:
    candidates = []

    for k in range(-3, 4):
        candidate = angle_deg + 120.0 * k
        candidate_wrapped = wrap_angle_deg(candidate)
        candidates.append(candidate_wrapped)

    valid = [c for c in candidates if 0.0 <= c <= 60.0]

    if len(valid) > 0:
        return min(valid, key=lambda c: abs(c - angle_deg))

    def distance_to_allowed_range(c: float) -> float:
        if c < 0.0:
            return abs(c)
        if c > 60.0:
            return abs(c - 60.0)
        return 0.0

    closest = min(candidates, key=distance_to_allowed_range)

    return float(np.clip(closest, 0.0, 60.0))


def optimal_rotation_from_vertices(
    ref_vertices: np.ndarray,
    def_vertices: np.ndarray,
) -> tuple[float, float]:
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


def triangle_angle(vertices: np.ndarray) -> float:
    pts = np.array(vertices, dtype=np.float32)

    edges = [
        (pts[0], pts[1]),
        (pts[1], pts[2]),
        (pts[2], pts[0]),
    ]

    edge_info = []

    for p1, p2 in edges:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        angle_wrapped = wrap_angle_deg(angle)

        horizontal_score = min(abs(angle_wrapped), abs(abs(angle_wrapped) - 180.0))
        edge_info.append((horizontal_score, p1, p2))

    if len(edge_info) == 0:
        return 0.0

    _, p1, p2 = min(edge_info, key=lambda item: item[0])

    if p2[0] < p1[0]:
        p1, p2 = p2, p1

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return float(np.degrees(np.arctan2(dy, dx)))


def segment_orange_triangles(
    image: np.ndarray,
    lower_orange: tuple[int, int, int] = (5, 100, 80),
    upper_orange: tuple[int, int, int] = (22, 255, 255),
    kernel_size: int = 5,
    min_blob_area_px: float = 1000.0,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array(lower_orange, dtype=np.uint8)
    upper = np.array(upper_orange, dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_main = np.array(lower_orange, dtype=np.uint8)
    upper_main = np.array(upper_orange, dtype=np.uint8)
    mask_main = cv2.inRange(hsv, lower_main, upper_main)
    lower_pale = np.array([0, 0, 115], dtype=np.uint8)
    upper_pale = np.array([179, 80, 255], dtype=np.uint8)
    mask_pale = cv2.inRange(hsv, lower_pale, upper_pale)
    mask = cv2.bitwise_or(mask_main, mask_pale)
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # Smooth small pixel noise before morphology.
    mask = cv2.medianBlur(mask, 5)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # First close small holes/gaps inside the orange triangle.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Then remove small isolated noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cleaned = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_blob_area_px:
            cv2.drawContours(cleaned, [contour], -1, 255, thickness=-1)

    return cleaned

def assign_ids_by_grid(triangles, row_tolerance_px=5):
    print(f"DEBUG: assign_ids_by_grid is used with row_tolerance_px={row_tolerance_px}")
    if len(triangles) == 0:
        return triangles

    # Sort all triangles from top to bottom
    sorted_triangles = sorted(triangles, key=lambda t: t["centroid"][1])

    rows = []

    for tri in sorted_triangles:
        cy = tri["centroid"][1]

        placed = False

        for row in rows:
            row_mean_y = np.mean([r["centroid"][1] for r in row])

            if abs(cy - row_mean_y) <= row_tolerance_px:
                row.append(tri)
                placed = True
                break

        if not placed:
            rows.append([tri])

    # Sort rows from top to bottom
    rows = sorted(rows, key=lambda row: np.mean([t["centroid"][1] for t in row]))

    triangle_id = 0
    output = []

    for row in rows:
        # Sort each row from left to right
        row_sorted = sorted(row, key=lambda t: t["centroid"][0])

        for tri in row_sorted:
            tri["id"] = triangle_id
            output.append(tri)
            triangle_id += 1

    return output

def draw_detected_triangles(
    image: np.ndarray,
    triangles: list[dict[str, Any]],
    draw_ids: bool = True,
) -> np.ndarray:
    annotated = image.copy()

    for tri in triangles:
        cx, cy = tri["centroid"]
        vertices = np.array(tri["vertices"], dtype=np.int32)

        cv2.drawContours(annotated, [vertices], -1, (0, 255, 0), 2)
        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        if draw_ids and tri.get("id") is not None:
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


def detect_orange_triangles(
    image: np.ndarray,
    min_area: float,
    max_area: float,
    assign_ids: bool,
    row_tolerance_px: float,
    lower_orange: tuple[int, int, int] = (0, 60, 40),
    upper_orange: tuple[int, int, int] = (30, 255, 255),
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    binary = segment_orange_triangles(
        image=image,
        lower_orange=lower_orange,
        upper_orange=upper_orange,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    triangles: list[dict[str, Any]] = []

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

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        angle_deg = triangle_angle(vertices)

        triangles.append(
            {
                "id": None,
                "row": None,
                "col": None,
                "centroid": (float(cx), float(cy)),
                "angle_deg": float(angle_deg),
                "area_px2": float(area),
                "contour": contour,
                "vertices": vertices,
                "triangle_fit": float(triangle_fit),
                "solidity": float(solidity),
            }
        )

    if assign_ids:
        triangles = assign_ids_by_grid(
            triangles,
            row_tolerance_px=33,
        )

    annotated = draw_detected_triangles(
        image=image,
        triangles=triangles,
        draw_ids=assign_ids,
    )

    return triangles, annotated, binary


def estimate_dynamic_thresholds(
    reference_triangles: list[dict[str, Any]],
) -> dict[str, float]:
    if len(reference_triangles) == 0:
        return {
            "median_area": 0.0,
            "median_spacing": 0.0,
            "min_area": 900.0, #dit moet aangepast worden als er niet genoeg driehoeken worden gedetecteerd of teveel (lager als je te weinig detecteerd) (hoger als je teveel detecteerd )
            "max_area": 1_000_000.0,
            "row_tolerance_px": 100.0, #dit moet aangepast worden naargelang afstand van de driehoeken in de afbeelding
            "matching_max_distance": 30.0,
        }

    areas = np.array([tri["area_px2"] for tri in reference_triangles], dtype=float)
    centroids = np.array([tri["centroid"] for tri in reference_triangles], dtype=float)

    median_area = float(np.median(areas))

    if len(reference_triangles) >= 2:
        tree = cKDTree(centroids)
        distances, _ = tree.query(centroids, k=2)
        nearest_distances = distances[:, 1]
        median_spacing = float(np.median(nearest_distances))
    else:
        median_spacing = 60.0

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


def match_deformed_to_reference(
    reference_triangles: list[dict[str, Any]],
    deformed_triangles: list[dict[str, Any]],
    max_distance: float,
) -> list[dict[str, Any]]:
    if len(reference_triangles) == 0 or len(deformed_triangles) == 0:
        return []

    ref_centroids = np.array(
        [tri["centroid"] for tri in reference_triangles],
        dtype=np.float32,
    )

    def_centroids = np.array(
        [tri["centroid"] for tri in deformed_triangles],
        dtype=np.float32,
    )

    tree = cKDTree(def_centroids)

    matches = []
    used_deformed_indices = set()

    for ref_index, ref_tri in enumerate(reference_triangles):
        distance, def_index = tree.query(ref_centroids[ref_index])

        if distance > max_distance:
            continue

        if int(def_index) in used_deformed_indices:
            continue

        used_deformed_indices.add(int(def_index))

        def_tri = deformed_triangles[int(def_index)]

        cx_ref, cy_ref = ref_tri["centroid"]
        cx_def, cy_def = def_tri["centroid"]

        dx = cx_def - cx_ref
        dy = cy_def - cy_ref

        displacement = float(np.sqrt(dx**2 + dy**2))

        rotation_change_deg, rotation_fit_error_px = optimal_rotation_from_vertices(
            ref_tri["vertices"],
            def_tri["vertices"],
        )

        matches.append(
            {
                "ID": ref_tri["id"],
                "row": ref_tri["row"],
                "col": ref_tri["col"],
                "cx_ref": float(cx_ref),
                "cy_ref": float(cy_ref),
                "cx_def": float(cx_def),
                "cy_def": float(cy_def),
                "dx_px": float(dx),
                "dy_px": float(dy),
                "displacement_px": displacement,
                "angle_ref_deg": float(ref_tri["angle_deg"]),
                "angle_def_deg": float(def_tri["angle_deg"]),
                "rotation_change_deg": float(rotation_change_deg),
                "rotation_fit_error_px": float(rotation_fit_error_px),
                "area_ref_px2": float(ref_tri["area_px2"]),
                "area_def_px2": float(def_tri["area_px2"]),
                "matching_distance_px": float(distance),
                "reference_triangle_fit": ref_tri.get("triangle_fit"),
                "deformed_triangle_fit": def_tri.get("triangle_fit"),
                "reference_solidity": ref_tri.get("solidity"),
                "deformed_solidity": def_tri.get("solidity"),
                "deformed_internal_index": int(def_index),
            }
        )

    return sorted(matches, key=lambda m: m["ID"])


def draw_matched_deformed_ids(
    image: np.ndarray,
    matches: list[dict[str, Any]],
) -> np.ndarray:
    annotated = image.copy()

    for match in matches:
        cx = match["cx_def"]
        cy = match["cy_def"]

        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.putText(
            annotated,
            str(match["ID"]),
            (int(cx) + 5, int(cy) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_displacement_arrows(
    image: np.ndarray,
    matches: list[dict[str, Any]],
    scale: float = 1.0,
) -> np.ndarray:
    annotated = image.copy()

    for match in matches:
        start = (
            int(round(match["cx_ref"])),
            int(round(match["cy_ref"])),
        )

        end = (
            int(round(match["cx_ref"] + scale * match["dx_px"])),
            int(round(match["cy_ref"] + scale * match["dy_px"])),
        )

        cv2.arrowedLine(
            annotated,
            start,
            end,
            (0, 0, 255),
            2,
            tipLength=0.25,
        )

        cv2.circle(annotated, start, 3, (255, 0, 0), -1)

    return annotated


def make_side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    h1, _ = img1.shape[:2]
    h2, _ = img2.shape[:2]

    target_h = max(h1, h2)

    def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h, _ = img.shape[:2]

        if h == target_h:
            return img

        pad_bottom = target_h - h

        return cv2.copyMakeBorder(
            img,
            0,
            pad_bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return np.hstack(
        [
            pad_to_height(img1, target_h),
            pad_to_height(img2, target_h),
        ]
    )


def build_matrices_from_matches(
    matches: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
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


def run_real_visual_pipeline(
    reference_image_path: str | Path,
    deformed_image_path: str | Path,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "real_visual_test",
    required_ids: list[int] | None = None,
    aruco_type: str = "DICT_4X4_100",
    expected_n_triangles: int | None = None,
    lower_orange: tuple[int, int, int] = (0, 60, 40),
    upper_orange: tuple[int, int, int] = (30, 255, 255),
) -> dict[str, Any]:
    if required_ids is None:
        required_ids = [1, 2, 3]

    reference_image_path = Path(reference_image_path)
    deformed_image_path = Path(deformed_image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aruco_output_dir = output_dir / "aruco_warp"
    detection_output_dir = output_dir / "triangle_detection"
    detection_output_dir.mkdir(parents=True, exist_ok=True)

    checks: dict[str, Any] = {
        "reference_image_path": str(reference_image_path),
        "deformed_image_path": str(deformed_image_path),
        "required_aruco_ids": required_ids,
        "aruco_type": aruco_type,
        "expected_n_triangles": expected_n_triangles,
        "lower_orange_hsv": lower_orange,
        "upper_orange_hsv": upper_orange,
    }

    aruco_result = warp_deformed_to_reference(
        reference_image_path=reference_image_path,
        deformed_image_path=deformed_image_path,
        output_dir=aruco_output_dir,
        aruco_type=aruco_type,
        required_ids=required_ids,
        require_all_markers=True,
    )

    checks["aruco_success"] = bool(aruco_result["success"])
    checks["aruco_used_ids"] = aruco_result["used_ids"]
    checks["aruco_reference_detected_ids"] = aruco_result["reference_detected_ids"]
    checks["aruco_deformed_detected_ids"] = aruco_result["deformed_detected_ids"]

    reference_img = load_image(reference_image_path)
    aligned_deformed_img = load_image(aruco_result["aligned_deformed_image_path"])

    reference_rough, _, reference_binary_rough = detect_orange_triangles(
        image=reference_img,
        min_area=10,
        max_area=1_000_000,
        assign_ids=False,
        row_tolerance_px=700.0,
        lower_orange=lower_orange,
        upper_orange=upper_orange,
    )

    dynamic = estimate_dynamic_thresholds(reference_rough)

    reference_triangles, reference_annotated, reference_binary = detect_orange_triangles(
        image=reference_img,
        min_area=dynamic["min_area"],
        max_area=dynamic["max_area"],
        assign_ids=True,
        row_tolerance_px=dynamic["row_tolerance_px"],
        lower_orange=lower_orange,
        upper_orange=upper_orange,
    )

    deformed_triangles, deformed_detected_annotated, deformed_binary = detect_orange_triangles(
        image=aligned_deformed_img,
        min_area=dynamic["min_area"],
        max_area=dynamic["max_area"],
        assign_ids=False,
        row_tolerance_px=dynamic["row_tolerance_px"],
        lower_orange=lower_orange,
        upper_orange=upper_orange,
    )

    matches = match_deformed_to_reference(
        reference_triangles=reference_triangles,
        deformed_triangles=deformed_triangles,
        max_distance=dynamic["matching_max_distance"],
    )

    triangle_matrix_full, X_displacements = build_matrices_from_matches(matches)

    df = pd.DataFrame(matches)

    if len(df) > 0:
        csv_columns = [
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
            "rotation_change_deg",
            "matching_distance_px",
            "area_ref_px2",
            "area_def_px2",
            "reference_triangle_fit",
            "deformed_triangle_fit",
            "reference_solidity",
            "deformed_solidity",
        ]
        df = df[csv_columns]

    reference_annotated_path = detection_output_dir / "reference_annotated_stable_ids.png"
    deformed_detected_path = detection_output_dir / "deformed_detected_no_ids.png"
    deformed_matched_path = detection_output_dir / "deformed_inherited_reference_ids.png"
    reference_binary_path = detection_output_dir / "reference_orange_binary_mask.png"
    deformed_binary_path = detection_output_dir / "deformed_orange_binary_mask.png"
    displacement_path = detection_output_dir / "displacement_arrows.png"
    side_by_side_path = detection_output_dir / "side_by_side_reference_vs_deformed.png"
    csv_path = detection_output_dir / "triangle_matches.csv"
    json_path = detection_output_dir / "triangle_matches.json"
    triangle_matrix_full_path = detection_output_dir / "triangle_matrix_full.npy"
    X_displacements_path = detection_output_dir / "X_displacements.npy"
    checks_path = detection_output_dir / "visual_check_report.json"

    deformed_matched_ids = draw_matched_deformed_ids(aligned_deformed_img, matches)
    displacement_img = draw_displacement_arrows(reference_img, matches)
    side_by_side = make_side_by_side(reference_annotated, deformed_matched_ids)

    cv2.imwrite(str(reference_annotated_path), reference_annotated)
    cv2.imwrite(str(deformed_detected_path), deformed_detected_annotated)
    cv2.imwrite(str(deformed_matched_path), deformed_matched_ids)
    cv2.imwrite(str(reference_binary_path), reference_binary)
    cv2.imwrite(str(deformed_binary_path), deformed_binary)
    cv2.imwrite(str(displacement_path), displacement_img)
    cv2.imwrite(str(side_by_side_path), side_by_side)

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(matches), file, indent=4)

    np.save(triangle_matrix_full_path, triangle_matrix_full)
    np.save(X_displacements_path, X_displacements)

    n_ref = len(reference_triangles)
    n_def = len(deformed_triangles)
    n_match = len(matches)

    match_ratio = float(n_match / n_ref) if n_ref > 0 else 0.0

    if n_match > 0:
        matching_distances = np.array(
            [m["matching_distance_px"] for m in matches],
            dtype=float,
        )
        displacements = np.array(
            [m["displacement_px"] for m in matches],
            dtype=float,
        )

        mean_matching_distance = float(np.mean(matching_distances))
        max_matching_distance = float(np.max(matching_distances))
        mean_displacement = float(np.mean(displacements))
        max_displacement = float(np.max(displacements))
    else:
        mean_matching_distance = None
        max_matching_distance = None
        mean_displacement = None
        max_displacement = None

    if expected_n_triangles is None:
        ready_for_neural_network = n_ref > 0 and n_match == n_ref
    else:
        ready_for_neural_network = (
            n_ref == expected_n_triangles
            and n_match == expected_n_triangles
        )

    checks.update(
        {
            "dynamic_thresholds": dynamic,
            "reference_triangles_detected": n_ref,
            "deformed_triangles_detected": n_def,
            "matched_triangles": n_match,
            "match_ratio": match_ratio,
            "mean_matching_distance_px": mean_matching_distance,
            "max_matching_distance_px": max_matching_distance,
            "mean_displacement_px": mean_displacement,
            "max_displacement_px": max_displacement,
            "ready_for_neural_network": bool(ready_for_neural_network),
            "outputs": {
                "reference_annotated": str(reference_annotated_path),
                "deformed_detected": str(deformed_detected_path),
                "deformed_matched": str(deformed_matched_path),
                "reference_binary": str(reference_binary_path),
                "deformed_binary": str(deformed_binary_path),
                "displacement_arrows": str(displacement_path),
                "side_by_side": str(side_by_side_path),
                "triangle_matches_csv": str(csv_path),
                "triangle_matches_json": str(json_path),
                "triangle_matrix_full": str(triangle_matrix_full_path),
                "X_displacements": str(X_displacements_path),
            },
        }
    )

    with open(checks_path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(checks), file, indent=4)

    print()
    print("REAL VISUAL PIPELINE SUMMARY")
    print("----------------------------")
    print(f"Reference triangles detected: {n_ref}")
    print(f"Deformed triangles detected:  {n_def}")
    print(f"Matched triangles:            {n_match}")
    print(f"Match ratio:                  {match_ratio:.3f}")
    print(f"Ready for neural network:     {ready_for_neural_network}")
    print()
    print(f"Saved outputs to: {detection_output_dir}")
    print(f"X_displacements:  {X_displacements_path}")
    print(f"Check report:     {checks_path}")

    return {
        "success": True,
        "ready_for_neural_network": ready_for_neural_network,
        "aruco_result": aruco_result,
        "checks": checks,
        "reference_triangles": reference_triangles,
        "deformed_triangles": deformed_triangles,
        "matches": matches,
        "dataframe": df,
        "triangle_matrix_full": triangle_matrix_full,
        "X_displacements": X_displacements,
        "output_dir": detection_output_dir,
        "X_displacements_path": X_displacements_path,
        "triangle_matrix_full_path": triangle_matrix_full_path,
        "checks_path": checks_path,
    }