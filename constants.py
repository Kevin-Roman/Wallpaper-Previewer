from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

# General
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_PATH = Path("./temp/")
BLENDER_SCENE_PATH = Path("./src/rendering/render_wall_with_material.blend")

# Room Layout Estimation
ARC_LENGTH_DEVIATION_TOLERANCE = 0.06
# FCN Layout Augmentation (LSUN-Room)
FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)
FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)
FCN_AUGMENTATION_ROOM_LAYOUT_ESTIMATION_WEIGHTS = Path(
    "./weights/room_layout_estimation/model_retrained.ckpt"
)

# Wall Segmentation
# Encoder-Decoder with PPM (WallSegmentation)
ENCODER_DECODER_PPM_WALL_SEGMENTATION_WEIGHTS = namedtuple(
    "Weights", ["encoder", "decoder"]
)(
    encoder=Path("./weights/wall_segmentation/best_encoder_epoch_19.pth"),
    decoder=Path("./weights/wall_segmentation/best_decoder_epoch_19.pth"),
)

# Illumination Estimation
# StyleLight
STYLEGAN2_ADA_FFHQ_ILLUMINATION_ESTIMATION_WEIGHTS = Path(
    "./weights/illumination_estimation/network-snapshot-002000.pkl"
)
