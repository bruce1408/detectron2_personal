# Setup detectron2 logger
# import detectron2
from detectron2.data.datasets import register_coco_instances
# from detectron2.utils.logger import setup_logger
# setup_logger()



# register_coco_instances("deepfashion_train", {}, "/content/DeepFashion2/deepfashion2_train.json",
# "/content/DeepFashion2/train/image")

register_coco_instances("train_val", {},
                        "/home/bruce/bigVolumn/Datasets/deepfashion/deepfashion2_validation.json",
                        "/home/bruce/bigVolumn/Datasets/deepfashion/image")
