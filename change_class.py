import os
import json

# Dossier contenant tous tes fichiers .txt
txt_folder = r"C:\Users\yasmi\Documents\detection\augmented\val"  # Mets ici ton chemin
output_json = r"C:\Users\yasmi\Documents\detection\augmented\val\merged_labels.json"


# Liste pour stocker toutes les annotations
merged_annotations = []

# Parcours tous les fichiers .txt
for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        txt_path = os.path.join(txt_folder, filename)

        with open(txt_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Lire x_center, y_center, width, height (en float)
                x_center, y_center, width, height = map(float, parts[1:])

                # Toujours mettre class_id à 0
                annotation = {
                    "filename": os.path.splitext(filename)[0] + ".jpg",  # on suppose que c'est une image .jpg
                    "class_id": 0,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                }

                merged_annotations.append(annotation)

# Écrire toutes les annotations dans un seul fichier JSON
with open(output_json, "w") as json_file:
    json.dump(merged_annotations, json_file, indent=4)

print(f"Fusion terminée ! {len(merged_annotations)} annotations sauvées dans {output_json}")
