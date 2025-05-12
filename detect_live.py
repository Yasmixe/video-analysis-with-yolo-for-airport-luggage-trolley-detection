from ultralytics import YOLO

# Charger ton modèle entraîné
model = YOLO("bestyolov8m.pt")

# Lancer la prédiction avec affichage vidéo
model.predict(
    source="video1.mp4",        
    show=True,       
    conf=0.1        
)
