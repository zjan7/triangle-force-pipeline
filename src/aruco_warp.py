#(pipe)this code is used to create the warp next is triangle_detection.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def detect_aruco(image: np.ndarray, aruco_type: str = "DICT_4X4_100",) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
   
    if aruco_type not in ARUCO_DICT:
        raise ValueError(f"Unknown ArUco type: {aruco_type}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters,)

    return corners, ids, rejected


def build_marker_corner_dict(corners: list[np.ndarray], ids: np.ndarray | None,) -> dict[int, np.ndarray]:
    
    marker_dict: dict[int, np.ndarray] = {}

    if ids is None:
        return marker_dict

    ids_flat = ids.flatten()

    for marker_corners, marker_id in zip(corners, ids_flat):
        points = marker_corners.reshape((4, 2)).astype(np.float32)
        marker_dict[int(marker_id)] = points

    return marker_dict


def get_matched_points(
    reference_dict: dict[int, np.ndarray],
    deformed_dict: dict[int, np.ndarray],
    required_ids: list[int] | None = None,
    require_all: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
   
    if required_ids is None:
        used_ids = sorted(set(reference_dict.keys()) & set(deformed_dict.keys()))
    else:
        used_ids = [
            marker_id
            for marker_id in required_ids
            if marker_id in reference_dict and marker_id in deformed_dict
        ]

    if require_all and required_ids is not None:
        missing_reference = [
            marker_id for marker_id in required_ids if marker_id not in reference_dict
        ]
        missing_deformed = [
            marker_id for marker_id in required_ids if marker_id not in deformed_dict
        ]

        if missing_reference or missing_deformed:
            raise RuntimeError(
                "Required ArUco markers were not detected in both images. "
                f"Missing in reference: {missing_reference}. "
                f"Missing in deformed: {missing_deformed}."
            )

    if len(used_ids) == 0:
        raise RuntimeError("No common ArUco marker IDs found in both images.")

    reference_points = []
    deformed_points = []

    for marker_id in used_ids:
        reference_marker_points = reference_dict[marker_id]
        deformed_marker_points = deformed_dict[marker_id]

        for reference_point, deformed_point in zip(
            reference_marker_points,
            deformed_marker_points,
        ):
            reference_points.append(reference_point)
            deformed_points.append(deformed_point)

    reference_points_array = np.array(reference_points, dtype=np.float32)
    deformed_points_array = np.array(deformed_points, dtype=np.float32)

    return reference_points_array, deformed_points_array, used_ids


def draw_detected_markers(image: np.ndarray, corners: list[np.ndarray], ids: np.ndarray | None,) -> np.ndarray:
   
    output = image.copy()

    if ids is None:
        return output

    ids_flat = ids.flatten()

    for marker_corners, marker_id in zip(corners, ids_flat):
        points = marker_corners.reshape((4, 2)).astype(int)

        for index in range(4):
            point_1 = tuple(points[index])
            point_2 = tuple(points[(index + 1) % 4])
            cv2.line(output, point_1, point_2, (0, 255, 0), 2)

        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))

        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)

        cv2.putText(output, f"ID {int(marker_id)}", (int(points[0][0]), int(points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,)

    return output


def warp_deformed_to_reference(reference_image_path: str | Path, deformed_image_path: str | Path, output_dir: str | Path = PROJECT_ROOT / "outputs" / "aruco_warp_test", aruco_type: str = "DICT_4X4_100", required_ids: list[int] | None = None, require_all_markers: bool = True,) -> dict[str, Any]:
    
    if required_ids is None:
        required_ids = [1, 2, 3]

    reference_image_path = Path(reference_image_path)
    deformed_image_path = Path(deformed_image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_image = cv2.imread(str(reference_image_path))
    deformed_image = cv2.imread(str(deformed_image_path))

    if reference_image is None:
        raise FileNotFoundError(f"Could not load reference image: {reference_image_path}")

    if deformed_image is None:
        raise FileNotFoundError(f"Could not load deformed image: {deformed_image_path}")

    reference_corners, reference_ids, _ = detect_aruco(reference_image, aruco_type)
    deformed_corners, deformed_ids, _ = detect_aruco(deformed_image, aruco_type)

    reference_id_list = [] if reference_ids is None else reference_ids.flatten().tolist()
    deformed_id_list = [] if deformed_ids is None else deformed_ids.flatten().tolist()

    print("Reference ArUco IDs:", reference_id_list)
    print("Deformed ArUco IDs :", deformed_id_list)

    if reference_ids is None or deformed_ids is None:
        raise RuntimeError("ArUco markers were not detected in one or both images.")

    reference_debug = draw_detected_markers(reference_image, reference_corners, reference_ids,)
    deformed_debug = draw_detected_markers(deformed_image, deformed_corners, deformed_ids,)

    reference_detected_path = output_dir / "reference_detected.png"
    deformed_detected_path = output_dir / "deformed_detected.png"

    cv2.imwrite(str(reference_detected_path), reference_debug)
    cv2.imwrite(str(deformed_detected_path), deformed_debug)

    reference_dict = build_marker_corner_dict(reference_corners, reference_ids)
    deformed_dict = build_marker_corner_dict(deformed_corners, deformed_ids)

    reference_points, deformed_points, used_ids = get_matched_points(reference_dict, deformed_dict, required_ids=required_ids, require_all=require_all_markers,)

    print("Used ArUco IDs:", used_ids)
    print("Number of matched ArUco corner points:", len(reference_points))

    if len(reference_points) < 4:
        raise RuntimeError("Not enough matched points to compute homography.")

    homography, mask = cv2.findHomography(deformed_points, reference_points, cv2.RANSAC,)

    if homography is None:
        raise RuntimeError("Homography estimation failed.")

    reference_height, reference_width = reference_image.shape[:2]

    aligned_deformed = cv2.warpPerspective(deformed_image, homography, (reference_width, reference_height),)

    overlay = cv2.addWeighted(reference_image, 0.5, aligned_deformed, 0.5,0,)

    aligned_deformed_path = output_dir / "aligned_deformed.png"
    overlay_path = output_dir / "overlay_reference_vs_aligned.png"
    homography_path = output_dir / "homography.npy"
    metadata_path = output_dir / "aruco_warp_metadata.json"

    cv2.imwrite(str(aligned_deformed_path), aligned_deformed)
    cv2.imwrite(str(overlay_path), overlay)
    np.save(homography_path, homography)

    metadata = {
        "reference_image": str(reference_image_path),
        "deformed_image": str(deformed_image_path),
        "aruco_type": aruco_type,
        "required_ids": required_ids,
        "require_all_markers": require_all_markers,
        "reference_detected_ids": reference_id_list,
        "deformed_detected_ids": deformed_id_list,
        "used_ids": used_ids,
        "number_of_matched_corner_points": int(len(reference_points)),
        "reference_detected_image": str(reference_detected_path),
        "deformed_detected_image": str(deformed_detected_path),
        "aligned_deformed_image": str(aligned_deformed_path),
        "overlay_image": str(overlay_path),
        "homography": str(homography_path),
        "success": True,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Saved reference marker debug image: {reference_detected_path}")
    print(f"Saved deformed marker debug image: {deformed_detected_path}")
    print(f"Saved aligned deformed image: {aligned_deformed_path}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved homography: {homography_path}")
    print(f"Saved metadata: {metadata_path}")

    return {
        "success": True,
        "aligned_deformed_image_path": aligned_deformed_path,
        "overlay_path": overlay_path,
        "homography_path": homography_path,
        "reference_detected_path": reference_detected_path,
        "deformed_detected_path": deformed_detected_path,
        "metadata_path": metadata_path,
        "homography": homography,
        "used_ids": used_ids,
        "reference_detected_ids": reference_id_list,
        "deformed_detected_ids": deformed_id_list,
        "metadata": metadata,
    }


if __name__ == "__main__":
    reference_path = PROJECT_ROOT / "outputs" / "reference" / "reference_zero_force_with_aruco.png"
    deformed_path = PROJECT_ROOT / "outputs" / "deformation_test" / "deformed_with_aruco.png"

    warp_deformed_to_reference(reference_image_path=reference_path, deformed_image_path=deformed_path, output_dir=PROJECT_ROOT / "outputs" / "aruco_warp_test", required_ids=[1, 2, 3], require_all_markers=True,)