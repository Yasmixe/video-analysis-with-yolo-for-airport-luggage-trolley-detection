import cv2
import json
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Dossier où sont tes images
image_folder = "C:/Users/yasmi/Documents/detection/Airport-Trolley-aeroportalgerV1/train/images/"
# Dossier où tu enregistres les annotations
annotation_folder = "C:/Users/yasmi/Documents/detection/Airport-Trolley-aeroportalgerV1/train/annotations/"

os.makedirs(annotation_folder, exist_ok=True)
images = os.listdir(image_folder)

for img_name in images:
    points = []
    img_path = os.path.join(image_folder, img_name)
    img = mpimg.imread(img_path)

    # -- Get the image size
    height, width = img.shape[0], img.shape[1]

    # -- Set figure size according to image size (important!)
    dpi = 100
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    ax.set_title(f"Click on each chariot - {img_name}")
    ax.axis('off')  # Enlève les axes pour plus joli

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            points.append((x, y))
            ax.plot(x, y, 'ro')  # plot point
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Save annotations
    annotation_path = os.path.join(annotation_folder, img_name.replace(".jpg", ".json"))
    with open(annotation_path, "w") as f:
        json.dump(points, f)