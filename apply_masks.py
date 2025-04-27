import cv2
import os
import numpy as np

# Path to the image and mask
image_path = r"C:\Users\yasmi\Documents\detection\test\image1.jpg"
mask_path = r"C:\Users\yasmi\Documents\detection\test\mask1.jpg"
output_folder_path = r"C:\Users\yasmi\Documents\detection\test"

# Load the original image
original_image = cv2.imread(image_path)

# Load the mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Ensure mask is binary
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Resize mask to match image size (if necessary)
if mask.shape[:2] != original_image.shape[:2]:
    mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Apply morphological operations to improve the mask
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.dilate(mask, kernel, iterations=1)

# Apply the mask to the original image
masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

# Unsharp Masking for better sharpening
gaussian = cv2.GaussianBlur(masked_image, (9,9), 10.0)
sharpened_image = cv2.addWeighted(masked_image, 1.5, gaussian, -0.5, 0)

# Save the preprocessed image
filename = os.path.basename(image_path)
preprocessed_path = os.path.join(output_folder_path, "preprocessed_" + filename)
cv2.imwrite(preprocessed_path, sharpened_image)

print("Preprocessing completed!")
