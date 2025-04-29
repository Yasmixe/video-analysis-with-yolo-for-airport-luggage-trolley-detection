import cv2
import json
import numpy as np

# Charger le mask
mask = cv2.imread(r'C:\Users\yasmi\Documents\detection\test\mask1.jpg', cv2.IMREAD_GRAYSCALE)

# Binariser le mask
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Trouver les contours
contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Liste pour stocker toutes les annotations
annotations = []

# Parcourir chaque contour
for contour in contours:
    contour = contour.squeeze()  # Supprime les dimensions inutiles

    if len(contour.shape) != 2: 
        continue

    all_points_x = contour[:, 0].tolist()  # liste des x
    all_points_y = contour[:, 1].tolist()  # liste des y

    annotation = {
        "shape_attributes": {
            "name": "polygon",
            "all_points_x": all_points_x,
            "all_points_y": all_points_y
        },
        "region_attributes": {}
    }

    annotations.append(annotation)

# Maintenant, on peut sauver les annotations en JSON
output_data = {
    "annotations": annotations
}

with open("contours_annotations.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Annotations JSON créées avec succès !")
