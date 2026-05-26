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
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

def detect_aruco_markers(image, aruco_type="DICT_4X4_100"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    if hasattr(cv2.aruco, "DetectorParameters"): #dit staat erin omdat verschillende versies van opencv andere gebruiken
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    return corners, ids, rejected

def draw_detections(image, corners, ids): #kan je zien of die de juiste detecteert
    output = image.copy()

    if ids is None or len(ids) == 0:
        print("No ArUco markers detected.")
        return output

    ids = ids.flatten()

    for marker_corners, marker_id in zip(corners, ids):
        pts = marker_corners.reshape((4, 2)).astype(int)

        top_left = tuple(pts[0])
        top_right = tuple(pts[1])
        bottom_right = tuple(pts[2])
        bottom_left = tuple(pts[3])
        cv2.line(output, top_left, top_right, (0, 255, 0), 2)     #omlijning in het groen dan zie je of hij juist doet
        cv2.line(output, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(output, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(output, bottom_left, top_left, (0, 255, 0), 2)

        center_x = int(np.mean(pts[:, 0]))     # hier heb ik de centerpoints nog bij gezet is handig
        center_y = int(np.mean(pts[:, 1]))
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(output, f"ID: {marker_id}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)     # dit is de marker ID om nogeens te controleren

        print(f"Detected marker ID {marker_id} at center ({center_x}, {center_y})")

    return output


def main():
    image_path = r"C:\Users\zibej\Downloads\grid1.png" #dit moet nog bij de output gebeuren ipv uit mijn bibliotheek. Moeten ook altijd referentie afbeelding gegevens  opslaan en posities van aruco 
    aruco_type = "DICT_4X4_100"     #dit is de dictionair die ik gebruik er bestaan er nog een paar die heb ik bovenaan gezet
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        return

    corners, ids, rejected = detect_aruco_markers(image, aruco_type)

    if ids is not None:
        print("Detected IDs:", ids.flatten().tolist())
    else:
        print("No markers detected.")

    output = draw_detections(image, corners, ids)

    cv2.imshow("Detected ArUco Markers", output)
    cv2.imwrite("detected_aruco_output.png", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()