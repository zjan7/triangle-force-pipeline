#code isn't used
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


def detect_aruco(image, aruco_type="DICT_4X4_100"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    return corners, ids, rejected


def build_marker_corner_dict(corners, ids):
    marker_dict = {}

    if ids is None:
        return marker_dict

    ids = ids.flatten()

    for marker_corners, marker_id in zip(corners, ids):
        pts = marker_corners.reshape((4, 2)).astype(np.float32)
        marker_dict[int(marker_id)] = pts

    return marker_dict


def get_matched_points(ref_dict, def_dict, required_ids=None):
    if required_ids is None:
        common_ids = sorted(set(ref_dict.keys()) & set(def_dict.keys()))
    else:
        common_ids = [i for i in required_ids if i in ref_dict and i in def_dict]

    if len(common_ids) < 1:
        raise ValueError("No common ArUco marker IDs found in both images.")

    ref_points = []
    def_points = []

    for marker_id in common_ids:
        ref_pts = ref_dict[marker_id]   # shape (4,2)
        def_pts = def_dict[marker_id]   # shape (4,2)

        for rp, dp in zip(ref_pts, def_pts):
            ref_points.append(rp)
            def_points.append(dp)

    ref_points = np.array(ref_points, dtype=np.float32)
    def_points = np.array(def_points, dtype=np.float32)

    return ref_points, def_points, common_ids


def draw_detected_markers(image, corners, ids):
    output = image.copy()

    if ids is None:
        return output

    ids = ids.flatten()

    for marker_corners, marker_id in zip(corners, ids):
        pts = marker_corners.reshape((4, 2)).astype(int)

        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(output, p1, p2, (0, 255, 0), 2)

        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)

        cv2.putText(output, f"ID {marker_id}", (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return output


def main():
    reference_path = r"C:\Users\zibej\Downloads\ref2.png" #hier moet je een afbeelding aanroepen uit je eigen laptop maar dit is de referentie
    deformed_path = r"C:\Users\zibej\Downloads\5.png"
    aruco_type = "DICT_4X4_100"
    required_ids = [1, 2, 3] #these are the ones we used

    reference_img = cv2.imread(reference_path)
    deformed_img = cv2.imread(deformed_path)

    if reference_img is None:
        raise FileNotFoundError(f"Could not load reference image: {reference_path}")

    if deformed_img is None:
        raise FileNotFoundError(f"Could not load deformed image: {deformed_path}")

    ref_corners, ref_ids, _ = detect_aruco(reference_img, aruco_type)
    def_corners, def_ids, _ = detect_aruco(deformed_img, aruco_type)

    print("Reference IDs:", None if ref_ids is None else ref_ids.flatten().tolist())
    print("Deformed IDs :", None if def_ids is None else def_ids.flatten().tolist())

    if ref_ids is None or def_ids is None:
        raise RuntimeError("ArUco markers not detected in one or both images.")

    ref_debug = draw_detected_markers(reference_img, ref_corners, ref_ids)
    def_debug = draw_detected_markers(deformed_img, def_corners, def_ids)

    cv2.imwrite("reference_detected.png", ref_debug)
    cv2.imwrite("deformed_detected.png", def_debug)

    ref_dict = build_marker_corner_dict(ref_corners, ref_ids)
    def_dict = build_marker_corner_dict(def_corners, def_ids)

    ref_points, def_points, used_ids = get_matched_points(ref_dict, def_dict, required_ids=required_ids)
    print("Used marker IDs:", used_ids)
    print("Number of matched points:", len(ref_points))

    if len(ref_points) < 4:
        raise RuntimeError("Not enough matched points to compute homography.")

    H, mask = cv2.findHomography(def_points, ref_points, cv2.RANSAC)

    if H is None:
        raise RuntimeError("Homography estimation failed.")
    print("Homography:\n", H)
    ref_h, ref_w = reference_img.shape[:2]

    warped_img = cv2.warpPerspective(deformed_img, H, (ref_w, ref_h))

    cv2.imwrite("warped_to_reference8.png", warped_img)
    overlay = cv2.addWeighted(reference_img, 0.5, warped_img, 0.5, 0)
    cv2.imwrite("overlay_reference_vs_warped.png", overlay)
    cv2.imshow("Reference", reference_img)
    cv2.imshow("Deformed", deformed_img)
    cv2.imshow("Warped to Reference", warped_img)
    cv2.imshow("Overlay", overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    