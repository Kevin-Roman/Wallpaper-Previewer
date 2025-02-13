from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CLASSES = 2
FC_DIM = 2048  # 512 for resnet18, for rest it is 2048

WEIGHTS_ENCODER = Path("./wall_segmentation/weights/encoder_decoder_ppm_encoder.pth")
WEIGHTS_DECODER = Path("./wall_segmentation/weights/encoder_decoder_ppm_decoder.pth")
