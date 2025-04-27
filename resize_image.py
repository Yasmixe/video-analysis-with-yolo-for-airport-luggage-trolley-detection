import cv2
import os

# Path to the folder containing the original images
folder_path = r"C:\Users\yasmi\Documents\detection\test"

# Path to the folder where resized images will be saved
output_folder_path = r"C:\Users\yasmi\Documents\detection\test"

# Desired output image size
output_size = (256, 256)

# Loop over all the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):

        # Load the original image
        image_path = os.path.join(folder_path, filename)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Failed to read image: {image_path}")
            continue

        # Resize the image
        resized_image = cv2.resize(original_image, (852,710))

        # Save the resized image in the output folder
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, resized_image)


