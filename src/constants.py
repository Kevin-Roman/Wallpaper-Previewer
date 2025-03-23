from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

# General
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Room Layout Estimation
ARC_LENGTH_DEVIATION_TOLERANCE = 0.03
# FCN Layout Augmentation
FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)
FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)
FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_WEIGHTS = Path(
    "./weights/room_layout_estimation/fcn_augmentation.ckpt"
)

# Wall Segmentation
# Encoder-Decoder with PPM
ENCODER_DECODER_PPM_WALL_SEGMENTATION_WEIGHTS = namedtuple(
    "Weights", ["encoder", "decoder"]
)(
    encoder=Path("./weights/wall_segmentation/encoder_decoder_ppm_encoder.pth"),
    decoder=Path("./weights/wall_segmentation/encoder_decoder_ppm_decoder.pth"),
)
