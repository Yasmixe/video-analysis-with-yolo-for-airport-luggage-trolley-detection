import os
folder_path = r'C:\Users\yasmi\Documents\detection\augmented' 

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith('.webp'):
            file_path = os.path.join(root, file)
            print(f"Suppression de : {file_path}")
            os.remove(file_path)

print("Suppression terminée.")
