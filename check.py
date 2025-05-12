import os

labels_dir = r"C:\Users\yasmi\Documents\detection\Airport-Trolley-aeroportalgerV1\valid\labels"
images_dir = r"C:\Users\yasmi\Documents\detection\Airport-Trolley-aeroportalgerV1\valid\images"  # adapte le chemin !

for txt_file in os.listdir(labels_dir):
    if txt_file.endswith(".txt"):
        image_file = txt_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_file)

        if not os.path.exists(image_path):
            print(f"⚠️ Image manquante pour : {txt_file}")

        with open(os.path.join(labels_dir, txt_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"⚠️ Mauvais format dans {txt_file} : {line}")
