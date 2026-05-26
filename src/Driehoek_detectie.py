#deze code wordt niet gebruikt
#hey, dit is de driehoekdetectiecode. Het detecteert eerst de driehoeken van de referentie en van de final en dan zoekt die matches uiteindelijk zie je hoeveel die heeft kunnen matchen. De IDs zijn alternerend weergegeven val linksboven omdat dit makkelijker leesbaar is maar zijn uiteindelijk enkel relevant om de matches te controleren en te geven
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
import json
#dit is een heilige library 

reference_path = r"C:\Users\zibej\Downloads\ref2.png" #dit is de referentie afbeelding die moet altijd hetzelfde zijn maar kan ook worden aangepast in de ref2.py code 
deformed_path  = r"C:\Users\zibej\Downloads\8.png" #dit is het resultaat van de aruco warped die noemt: Warping_deformed_reference.py

output_dir = Path(r"C:\Users\zibej\Downloads\triangle_outputs") #alles wordt opgeslagen in een output folder 
output_dir.mkdir(parents=True, exist_ok=True)

#1 cm is 37.8 pixels handig om te weten voor het model dan kunnen we hiermee spelen en in model 1.73 cm uit elkaar dus 65.4 ongeveer pixels
min_area = 40#hiermee negeren we te kleine dingen
max_area = 2000 #hiermee negeren we te grote dingen dit is belangrijk want eerst nam die alle Aruco markers ook mee beetje vreemd

matching_max_distance = 60 #dit is voor het matchen en is om te zeggen dat de centroid van ref en nieuw niet meer dan 60 pixels uit elkaar mogen zitten, heb ik een beetje mee zitten spelen en dit werkte, je kan zelf andere waardes proberen ofc

use_grid_ids = True #zie het assign-deel maar basically verdelen we het in rijen en kolommen
row_tolerance_px = 30 #y-waardes met binnen 30 pixels van elkaar zien als 1 rij. Weeral veel getest dit werkte goed je kan het nog aanpassen

def load_image(path): #dit is wel logisch denk ik maar het laad de afbeelding in OpenCV
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return img


def show_image(img, title="", figsize=(8, 8)):
    plt.figure(figsize=figsize)

    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #OpenCV werkt met BGR dus hier vormen we hem nog om naar RGB

    plt.title(title)
    plt.axis("off")
    plt.show()


def wrap_angle_deg(angle): #hoeken altijd in een range van -180 tot 180 graden dit is gewoon voor hele grote rotaties maar ik heb geen idee hoe groot die rotaties zijn tbh
    return (angle + 180) % 360 - 180


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
    # Treat tiny numerical noise near 0 or 360 as zero
    if d_angle <= zero_tolerance_deg:
        d_angle = 0.0

    if d_angle >= 360 - zero_tolerance_deg:
        d_angle = 0.0

    return d_angle

def assign_ids_by_grid(triangles, row_tolerance_px=30):

    if len(triangles) == 0:
        return triangles

    triangles_sorted_y = sorted(triangles, key=lambda t: t["centroid"][1]) #sorteren alle y-waarden in rijen
    rows = []

    for tri in triangles_sorted_y:
        cx, cy = tri["centroid"]
        placed = False

        for row in rows:
            row_y_values = [item["centroid"][1] for item in row]
            row_mean_y = np.mean(row_y_values) #

            if abs(cy - row_mean_y) <= row_tolerance_px: #eerste vergelijken of het in de rij past 
                row.append(tri) #als het past gaat die de rij in anders maakt die een andere rij
                placed = True
                break

        if not placed:
            rows.append([tri])

    rows = sorted(rows, key=lambda row: np.mean([item["centroid"][1] for item in row]))  #binnen elke rij van boven naar onder gesorteerd 
    output = []
    current_id = 0

    for row_index, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda t: t["centroid"][0])  #binnen elke rij van links naar rechts gesorteerd 

        for col_index, tri in enumerate(row_sorted):  #hier geef ik elk driehoekje een ID, rij en kolom
            tri["row"] = row_index
            tri["col"] = col_index
            tri["id"] = current_id

            output.append(tri)
            current_id += 1

    return output


def draw_detected_triangles(image, triangles, draw_ids=True): #teken de driehoeken
    annotated = image.copy()

    for tri in triangles:
        cx, cy = tri["centroid"]
        vertices = np.array(tri["vertices"], dtype=np.int32)

        cv2.drawContours(annotated, [vertices], -1, (0, 255, 0), 2)   #hiermee de groene contour, mag elke kleur zijn is gewoon om zelf te checken

        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)  #middelpunt met rode bol

        if draw_ids and tri["id"] is not None:
            cv2.putText(annotated, str(tri["id"]), (int(cx) + 5, int(cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    return annotated


def detect_triangles(image, min_area=40, max_area=2000, assign_ids=True, use_grid_ids=True, row_tolerance_px=30):  #dit is mega belangrijk hiermee detecteren we de driehoeken

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #contrast voor detectie
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  #hiermee haal je kleine noise weg is Gaussian Blur
    binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #grey naar zwart wit, witte driehoek op zwarte achtergrond
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)  #OpenCV gaat lokaal zelf ene threshold kiezen
    binary = cv2.bitwise_or(binary_otsu, binary_adaptive)  #combineert beiden manieren als het een driehoek detecteert gaat het bijgehouden worden is gewoon basic OpenCV
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  #dit is gewoon een opencv manier om alle kleine onzuiverheden recht te trekken

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #alle randen van de witte objecten 

    triangles = []

    for contour in contours:  
        area = cv2.contourArea(contour) #zoek de oppervlakken van alle gevonden contouren

        if area < min_area or area > max_area:  #te klein en te groot weg
            continue

        if len(contour) < 5: #kleine en slechte contours weg
            continue

        hull = cv2.convexHull(contour) #dit is ook een opencv truc om de driehoeken te vinden 
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / hull_area
        if solidity < 0.65: # Reject very irregular/non-convex blobs
            continue

        enclosing_area, enclosing_triangle = cv2.minEnclosingTriangle(hull) 
        if enclosing_triangle is None:
            continue

        if enclosing_area <= 0:
            continue

        triangle_fit = area / enclosing_area
        if triangle_fit < 0.55: # als dit te klein is gaan er veel genegeerd worden en als dit te groot is gaan veel niet driehoeken worden doorgelaten (rare methode maar werkt beter dan de  if len(approx) == 3: wat ik vorige keren probeerde)
            continue

        vertices = enclosing_triangle.reshape(3, 2)

        M = cv2.moments(contour)

        if M["m00"] == 0: 
            continue

        cx = M["m10"] / M["m00"] #hiermee vinden we de centroids van de gevonden driehoeken 
        cy = M["m01"] / M["m00"]

        angle_deg = triangle_angle(vertices)
#slaan we alle info op van de driehoeken 
        triangles.append({"id": None, "row": None, "col": None, "centroid": (cx, cy), "angle_deg": angle_deg, "area_px2": area, "contour": contour, "vertices": vertices, "triangle_fit": triangle_fit, "solidity": solidity})

    if assign_ids: #hier voegen we IDs toe aan gedetecteerde driehoeken 
        if use_grid_ids:
            triangles = assign_ids_by_grid(triangles, row_tolerance_px=row_tolerance_px)
        else:
            triangles = assign_ids_simple_sort(triangles)

    annotated = draw_detected_triangles(image, triangles, draw_ids=assign_ids)

    return triangles, annotated, binary


def match_deformed_to_reference(reference_triangles, deformed_triangles, max_distance=60): #dus hier gaan we de deformed triangles matchen aan de referentie driehoeken en zo krijgen ze hun IDs

    if len(reference_triangles) == 0:
        return []

    if len(deformed_triangles) == 0:
        return []

    ref_centroids = np.array([tri["centroid"] for tri in reference_triangles], dtype=np.float32) #zoeken naar centroid arrays van referentie en deformed

    def_centroids = np.array([tri["centroid"] for tri in deformed_triangles], dtype=np.float32)

    tree = cKDTree(def_centroids) #hiermee snel zoeken welke het dichtste bij is (echt cool kende ik ook nie, iemand deed het op een blog en het werkt echt goed)

    matches = []
    used_deformed_indices = set()

    for ref_index, ref_tri in enumerate(reference_triangles):
        distance, def_index = tree.query(ref_centroids[ref_index]) #dichtsbijzijnde deformed triangle bij referentie 

        if distance > max_distance: #reject het als het te ver weg is
            continue

        if def_index in used_deformed_indices: #elke deformed driehoek slechts  keer
            continue

        used_deformed_indices.add(def_index)

        def_tri = deformed_triangles[def_index]

        cx_ref, cy_ref = ref_tri["centroid"] 
        cx_def, cy_def = def_tri["centroid"]

        dx = cx_def - cx_ref #hiermee zoeken we de verplaatsingen van referentie naar deformed
        dy = cy_def - cy_ref

        displacement = np.sqrt(dx**2 + dy**2) #dit is de totale, kan er altijd uit maar dacht dat het nog leuk is om te zien

        d_angle = counterclockwise_angle_difference(ref_tri["angle_deg"], def_tri["angle_deg"]) #rotatie berekening tussen deformed en referentie

        matches.append({
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
            "matching_distance_px": distance,

            "reference_triangle_fit": ref_tri.get("triangle_fit", None),
            "deformed_triangle_fit": def_tri.get("triangle_fit", None),
            "reference_solidity": ref_tri.get("solidity", None),
            "deformed_solidity": def_tri.get("solidity", None),

            "deformed_internal_index": def_index
        })

    matches = sorted(matches, key=lambda m: m["ID"])

    return matches


def draw_matched_deformed_ids(image, matches): #teken de IDs op de gematchte deformed driehoeken
    annotated = image.copy()

    for m in matches:
        cx = m["cx_def"]
        cy = m["cy_def"]

        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.putText(annotated, str(m["ID"]), (int(cx) + 5, int(cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

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

        return cv2.copyMakeBorder(img, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

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
reference_img = load_image(reference_path)
deformed_img = load_image(deformed_path)

show_image(reference_img, "Reference image")
show_image(deformed_img, "Deformed / warped image")

# Reference image gets stable IDs
reference_triangles, reference_annotated, reference_binary = detect_triangles(reference_img, min_area=min_area, max_area=max_area, assign_ids=True, use_grid_ids=use_grid_ids, row_tolerance_px=row_tolerance_px)

# Deformed image is detected, but its own IDs are not used
deformed_triangles, deformed_detected_annotated, deformed_binary = detect_triangles(deformed_img, min_area=min_area, max_area=max_area, assign_ids=False, use_grid_ids=False, row_tolerance_px=row_tolerance_px)

print(f"Reference triangles detected: {len(reference_triangles)}")
print(f"Deformed triangles detected:  {len(deformed_triangles)}")

show_image(reference_binary, "Reference binary mask")
show_image(deformed_binary, "Deformed binary mask")

show_image(reference_annotated, "Reference annotated with stable IDs")
show_image(deformed_detected_annotated, "Deformed detected triangles, no IDs")

matches = match_deformed_to_reference(reference_triangles, deformed_triangles, max_distance=matching_max_distance) #finaal matchen van deformed naar reference IDs

print_detection_summary(reference_triangles, deformed_triangles, matches)

df = pd.DataFrame(matches)

output_columns = [
    "ID",
    "cx_ref",
    "cy_ref",
    "cx_def",
    "cy_def",
    "dx_px",
    "dy_px",
    "rotation_change_deg"
]

if len(df) > 0:
    df = df[output_columns]

    df = df.rename(columns={
        "cx_ref": "cx_original",
        "cy_ref": "cy_original",
        "cx_def": "cx_new",
        "cy_def": "cy_new",
        "dx_px": "dx",
        "dy_px": "dy",
        "rotation_change_deg": "rotation_deg"
    })

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    display(df)
else:
    print("No matches found.")
    print("Try increasing matching_max_distance.")

deformed_matched_ids = draw_matched_deformed_ids(deformed_img, matches)

#displacement_img = draw_displacement_arrows(reference_img, matches)

side_by_side = make_side_by_side(reference_annotated, deformed_matched_ids)

show_image(deformed_matched_ids, "Deformed image with inherited reference IDs")
#show_image(displacement_img, "Displacement arrows")
show_image(side_by_side, "Reference IDs vs deformed inherited IDs", figsize=(14, 8))



csv_path = output_dir / "triangle_matrix.csv" #resultaten opslaan
json_path = output_dir / "triangle_analysis.json"

reference_annotated_path = output_dir / "reference_annotated_stable_ids.png"
deformed_matched_path = output_dir / "deformed_inherited_reference_ids.png"
deformed_detected_path = output_dir / "deformed_detected_no_ids.png"
reference_binary_path = output_dir / "reference_binary_mask.png"
deformed_binary_path = output_dir / "deformed_binary_mask.png"
displacement_path = output_dir / "displacement_arrows.png"
side_by_side_path = output_dir / "side_by_side_reference_vs_deformed.png"

df.to_csv(csv_path, index=False)

with open(json_path, "w") as f:
    json.dump(matches, f, indent=4)

cv2.imwrite(str(reference_annotated_path), reference_annotated)
cv2.imwrite(str(deformed_matched_path), deformed_matched_ids)
cv2.imwrite(str(deformed_detected_path), deformed_detected_annotated)
cv2.imwrite(str(reference_binary_path), reference_binary)
cv2.imwrite(str(deformed_binary_path), deformed_binary)
cv2.imwrite(str(displacement_path), displacement_img)
cv2.imwrite(str(side_by_side_path), side_by_side)

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