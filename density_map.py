import cv2

img_path = 'C:/Users/yasmi/Documents/detection/Airport-Trolley-aeroportalgerV1/train/images/68428914-chariot-à-bagages-de-l-aéroport-avec-des-valises-sur-fond-blanc-rendu-3d'  # ou ton chemin mis à jour

img = cv2.imread(img_path)
if img is None:
    print("L'image n'a pas pu être chargée, vérifie le chemin.")
else:
    print(f"Image chargée avec succès ! Dimensions : {img.shape}")
