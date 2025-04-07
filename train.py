from ultralytics import YOLO

# Charger le modèle YOLOv8n (nano, rapide et léger)
import torch
print(torch.cuda.is_available())  # doit retourner True

model = YOLO('yolov8n.pt')

# Entraînement du modèle sur ton dataset personnalisé
results = model.train(
    data='data.yaml',  # chemin vers ton fichier yaml
    epochs=50,                    # nombre d'époques
    imgsz=640,                    # taille des images
    batch=8,                      # taille de batch
    name='yolov8_chariotsv2',       # nom du dossier de sauvegarde
    device='cpu'                 # ou 'cpu' si tu n'as pas de GPU
)
