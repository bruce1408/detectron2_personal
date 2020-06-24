# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os
import numpy as np
import cv2
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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


register_coco_instances("train_val", {}, "/raid/chenx/validation/deepfashion2_validation.json",
                        "/raid/chenx/validation/image")
mydata_metadata = MetadataCatalog.get("train_val")
dataset_dicts = DatasetCatalog.get("train_val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_val",)
cfg.DATASETS.TEST = ()
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "/raid/chenx/validation/model_final_f6e8b1.pkl"
# train model
cfg.SOLVER.WEIGHT_DECAY = 0.0001  # 权重衰减
cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
cfg.SOLVER.IMS_PER_BATCH = 12
cfg.SOLVER.MOMENTUM = 0.9  # 优化器动能
cfg.SOLVER.BASE_LR = 0.001  # 初始学习率
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 10000  # 最大迭代次数
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05  # 学习率衰减倍数
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
cfg.TEST.EVAL_PERIOD = 500  # 迭代指定次数进行评估
print(cfg.OUTPUT_DIR)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# print('cfg is\n', cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# prediction
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set the testing threshold for this model
# cfg.DATASETS.TEST = ("train_val",)
# predictor = DefaultPredictor(cfg)
#
# im = cv2.imread("/home/bruce/bigVolumn/Datasets/deepfashion/image/000072.jpg")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('rr', v.get_image()[:, :, ::-1])
# cv2.waitKey(0)