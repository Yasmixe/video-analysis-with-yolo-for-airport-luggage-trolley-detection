import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime

# Charger le modèle
model = YOLO(r'C:\Users\yasmi\Documents\detection\runs\detect\pile4\weights\best.pt')

# Ouvrir la vidéo (remplace par le chemin ou une webcam avec 0)
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Frame rate de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval_minutes = 0.2
frame_interval = int(fps * 60 * frame_interval_minutes)

# Pour comparaison
last_density = None

# CSV pour logs
with open("densite_log.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Densité", "État"])

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prendre une frame toutes les 3-4 minutes
        if frame_idx % frame_interval == 0:
            # Enregistrer temporairement
            temp_path = "frame_temp.jpg"
            cv2.imwrite(temp_path, frame)

            # Prédiction YOLO
            results = model.predict(temp_path, conf=0.2)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                total = binary.size
                non_zero = np.count_nonzero(binary)
                density = (non_zero / total) * 100

                # Détection du changement
                if last_density is not None:
                    diff = density - last_density
                    if diff < -5:
                        state = "Pile vidée"
                    elif diff > 5:
                        state = "Pile remplie"
                    else:
                        state = "Stable"
                else:
                    state = "Premier point"
                
                last_density = density

                # Enregistrer dans CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, f"{density:.2f}", state])
                print(f"{timestamp} | Densité: {density:.2f}% | {state}")
        
        frame_idx += 1

cap.release()
