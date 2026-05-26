from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent

reference_path = PROJECT_ROOT / "outputs" / "reference" / "reference_zero_force_with_aruco.png"
deformed_path = PROJECT_ROOT / "outputs" / "deformed" / "deformed_with_aruco.png"
aligned_path = PROJECT_ROOT / "outputs" / "aruco_warp_test" / "aligned_deformed.png"

output_dir = PROJECT_ROOT / "outputs" / "aruco_alignment_debug"
output_dir.mkdir(parents=True, exist_ok=True)


def get_aruco_dictionary():
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = get_aruco_dictionary()

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None:
        return {}

    ids = ids.flatten()
    result = {}

    for marker_id, marker_corners in zip(ids, corners):
        result[int(marker_id)] = marker_corners.reshape(4, 2)

    return result


def marker_centers(markers):
    return {
        marker_id: corners.mean(axis=0)
        for marker_id, corners in markers.items()
    }


def print_marker_comparison(ref_markers, aligned_markers):
    common_ids = sorted(set(ref_markers.keys()) & set(aligned_markers.keys()))

    print()
    print("=" * 80)
    print("ARUCO MARKER ALIGNMENT CHECK")
    print("=" * 80)

    if len(common_ids) == 0:
        print("No common markers found.")
        return

    residuals = []

    ref_centers = marker_centers(ref_markers)
    aligned_centers = marker_centers(aligned_markers)

    for marker_id in common_ids:
        ref_c = ref_centers[marker_id]
        ali_c = aligned_centers[marker_id]
        diff = ali_c - ref_c
        dist = np.linalg.norm(diff)
        residuals.append(dist)

        print(
            f"Marker {marker_id}: "
            f"dx={diff[0]: .3f} px, dy={diff[1]: .3f} px, distance={dist:.3f} px"
        )

    residuals = np.array(residuals)
    print()
    print(f"Mean marker-center alignment error: {residuals.mean():.3f} px")
    print(f"Max marker-center alignment error:  {residuals.max():.3f} px")


def draw_markers(image, markers, color):
    out = image.copy()
    for marker_id, corners in markers.items():
        pts = corners.astype(int)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        c = pts.mean(axis=0).astype(int)
        cv2.putText(
            out,
            str(marker_id),
            tuple(c),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def main():
    reference = cv2.imread(str(reference_path))
    aligned = cv2.imread(str(aligned_path))

    if reference is None:
        raise FileNotFoundError(reference_path)

    if aligned is None:
        raise FileNotFoundError(aligned_path)

    ref_markers = detect_markers(reference)
    aligned_markers = detect_markers(aligned)

    print(f"Reference markers found: {sorted(ref_markers.keys())}")
    print(f"Aligned markers found:   {sorted(aligned_markers.keys())}")

    print_marker_comparison(ref_markers, aligned_markers)

    ref_drawn = draw_markers(reference, ref_markers, (0, 255, 0))
    aligned_drawn = draw_markers(aligned, aligned_markers, (0, 0, 255))

    overlay = cv2.addWeighted(ref_drawn, 0.5, aligned_drawn, 0.5, 0)

    cv2.imwrite(str(output_dir / "reference_markers.png"), ref_drawn)
    cv2.imwrite(str(output_dir / "aligned_markers.png"), aligned_drawn)
    cv2.imwrite(str(output_dir / "marker_overlay.png"), overlay)

    print()
    print("Saved debug images to:")
    print(output_dir)


if __name__ == "__main__":
    main()