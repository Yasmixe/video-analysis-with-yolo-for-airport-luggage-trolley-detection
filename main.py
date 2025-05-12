import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
import tkinter
matplotlib.use('Qt5Agg')  # Requires PyQt5

model = YOLO(r'C:\Users\yasmi\Documents\detection\runs\detect\new2\weights\best.pt')
results = model.predict("data_2.jpg", conf=0.2)  # conf = seuil de confiance

# Afficher les résultats
results[0].show()