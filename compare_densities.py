import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Charger le modèle YOLO
model = YOLO(r'C:\Users\yasmi\Documents\detection\runs\detect\pile4\weights\best.pt')

def get_density(image_path):
    # Prédiction YOLO
    results = model.predict(image_path, conf=0.2)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    image = cv2.imread(image_path)

    if len(boxes) == 0:
        print(f"Aucune pile détectée dans {image_path}")
        return None, None

    # Extraire la première ROI détectée
    x1, y1, x2, y2 = map(int, boxes[0])
    roi = image[y1:y2, x1:x2]

    # Densité par seuillage
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 1, 255, cv2.THRESH_BINARY)
    total_pixels = binary_roi.size
    non_zero_pixels = np.count_nonzero(binary_roi)
    density = (non_zero_pixels / total_pixels) * 100

    return density, roi

# Charger et analyser les deux images
density1, roi1 = get_density("image2.jpg")
density2, roi2 = get_density("image.png")

if density1 is not None and density2 is not None:
    print(f"Densité avant : {density1:.2f}%")
    print(f"Densité après : {density2:.2f}%")

    if density2 < density1 - 5:  # seuil de 5%
        print("La pile de chariots s’est probablement VIDÉE.")
    elif density2 > density1 + 5:
        print("La pile s’est REMPLIE.")
    else:
        print("⏸Pas de changement significatif.")
    
    # Affichage
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB))
    plt.title(f"Avant - Densité: {density1:.2f}%")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
    plt.title(f"Après - Densité: {density2:.2f}%")
    plt.axis('off')
    plt.show()
