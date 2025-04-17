from ultralytics import YOLO
import cv2

# Charger votre modèle personnalisé (remplacez par votre fichier .pt)
model = YOLO("yolov8n-seg.pt")  # ou "yolov8n.pt" pour un modèle de base
results = model.predict("data_4.jpg", conf=0.2)  # conf = seuil de confiance

# Afficher les résultats
results[0].show()