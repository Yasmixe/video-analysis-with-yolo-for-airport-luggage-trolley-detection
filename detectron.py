if __name__ == '__main__':

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.engine import DefaultTrainer
    import cv2
    import numpy as np
    from norfair import Detection, Tracker, video, draw_tracked_objects
    from detectron2.data.datasets import register_coco_instances
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    register_coco_instances("my_dataset_train", {}, r"C:\Users\yasmi\Documents\dash\data\annotations\_annotations_train_coco.json", r"C:\Users\yasmi\Documents\dash\data\train")
    register_coco_instances("my_dataset_val", {}, r"C:\Users\yasmi\Documents\dash\data\annotations\_annotations_valid_coco.json", r"C:\Users\yasmi\Documents\dash\data\valid")
                                                                                  
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")


    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000  # Nombre d'itérations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)


