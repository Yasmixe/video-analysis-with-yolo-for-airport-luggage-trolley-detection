import os
import glob

folder_path = r"C:\Users\yasmi\Documents\detection\augmented\val"

txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

for file_path in txt_files:
    try:
        os.remove(file_path)
        print(f"Supprimé : {file_path}")
    except Exception as e:
        print(f"Erreur pour {file_path} : {e}")
