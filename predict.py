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
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

setup_logger()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_val",)
# cfg.DATASETS.TEST = ()
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#
# cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.SOLVER.BASE_LR = 0.001
# cfg.SOLVER.WARMUP_ITERS = 1000
# cfg.SOLVER.MAX_ITER = (500)
# # cfg.SOLVER.STEPS = (1000, 1500)
# # cfg.SOLVER.GAMMA = 0.05
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
# cfg.TEST.EVAL_PERIOD = 500


MetadataCatalog.get("train_val").thing_classes = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
                                                  'long_sleeved_outwear',
                                                  'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
                                                  'long_sleeved_dress', 'vest_dress', 'sling_dress']
fruits_nuts_metadata = MetadataCatalog.get("train_val")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set the testing threshold for this model
cfg.DATASETS.TEST = ("train_val",)
predictor = DefaultPredictor(cfg)

im = cv2.imread("/home/bruce/bigVolumn/Datasets/deepfashion/image/000002.jpg")
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('rr', v.get_image()[:, :, ::-1])
cv2.waitKey(0)
