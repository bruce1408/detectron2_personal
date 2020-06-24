# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import MetadataCatalog, DatasetCatalog
# import detectron2
from detectron2.data.datasets import register_coco_instances
# from detectron2.utils.logger import setup_logger
# setup_logger()


register_coco_instances("train_val", {}, "/home/bruce/bigVolumn/Datasets/deepfashion/deepfashion2_validation.json", "/home/bruce/bigVolumn/Datasets/deepfashion/image")
mydata_metadata = MetadataCatalog.get("train_val")
dataset_dicts = DatasetCatalog.get("train_val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_val",)
cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = (500)
# cfg.SOLVER.STEPS = (1000, 1500)
# cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
# cfg.TEST.EVAL_PERIOD = 500
# print(cfg.OUTPUT_DIR)
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# print('cfg is\n', cfg)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

# prediction
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set the testing threshold for this model
cfg.DATASETS.TEST = ("train_val",)
predictor = DefaultPredictor(cfg)

im = cv2.imread("/home/bruce/bigVolumn/Datasets/deepfashion/image/000072.jpg")
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('rr', v.get_image()[:, :, ::-1])
cv2.waitKey(0)