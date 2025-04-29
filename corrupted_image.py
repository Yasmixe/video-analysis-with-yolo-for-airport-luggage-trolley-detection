import os
from PIL import Image

def check_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    img.verify()  # Vérifie l'intégrité de l'image
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image found: {filename}")

check_images("C:\Users\yasmi\Documents\detection\Airport-Trolley-aeroportalgerV1\train\images")
