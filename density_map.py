import os
import json
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

image_folder = "C:/Users/yasmi/Documents/detection/Airport-Trolley-aeroportalgerV1/train/images/"
# Dossier où tu enregistres les annotations
annotation_folder = "C:/Users/yasmi/Documents/detection/Airport-Trolley-aeroportalgerV1/train/annotations/"
density_folder = "C:/Users/yasmi/Documents/detection/Airport-Trolley-aeroportalgerV1/train/density/"
os.makedirs(density_folder, exist_ok=True)

images = os.listdir(image_folder)

for img_name in images:
    # Charger l'image pour obtenir sa taille
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Créer une carte vide
    density = np.zeros((h, w), dtype=np.float32)

    # Charger les annotations (points)
    json_path = os.path.join(annotation_folder, img_name.replace(".jpg", ".json"))
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            points = json.load(f)

        for point in points:
            x, y = point
            if x >= w or y >= h:
                continue  # sécurite

            density[y, x] += 1  # 1 chariot ajouté à cet endroit

    # Appliquer un flou Gaussien pour simuler la densité
    density = gaussian_filter(density, sigma=15)  # <-- tu peux ajuster sigma (15 est OK pour commencer)

    # Sauvegarder la density map
    density_path = os.path.join(density_folder, img_name.replace(".jpg", ".npy"))
    np.save(density_path, density)

    # Facultatif : voir ce que ça donne
    plt.imshow(density, cmap='jet')
    plt.title(f"Density map for {img_name}")
    plt.axis('off')
    plt.show()
