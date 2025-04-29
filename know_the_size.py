import cv2
import os

# set the path to the folder containing the images
folder_path = r"C:\Users\yasmi\Documents\detection\test"

# loop over each file in the folder
for file_name in os.listdir(folder_path):
    # check if the file is an image 
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        # read the image using OpenCV
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        # print the shape of the image
        print(f'{file_name}: {image.shape}')