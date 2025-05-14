from ultralytics import YOLO

if __name__ == '__main__':
    # Charger le modèle
    model = YOLO('yolo11n-obb.pt')
    model = YOLO(r'C:\Users\yasmi\Documents\dash\best_model_last_images.pt')
    model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Entraîner le modèle
    results = model.train(
        data=r"C:\Users\yasmi\Documents\dash\MyProject.v4i.yolov8-obb\data.yaml",
        epochs=80,
        imgsz=512,
        batch=2,                      # taille de batch
        name='yolov11-obb2',
        device = 0,
        workers=0
    )
