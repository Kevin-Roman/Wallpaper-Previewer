import math
from pathlib import Path
from typing import Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torch import nn
from torchtyping import TensorType
from torchvision import models

from constants import TORCH_DEVICE

from .constants import MODEL_IMAGE_SIZE
from .utils import ProcessImage


class Predictor:
    """Predictor for layout estimation."""

    def __init__(self, weight_path: Path, backbone: str = "resnet101") -> None:
        # pylint: disable=E1120
        self.model = LayoutSeg.load_from_checkpoint(
            checkpoint_path=weight_path, backbone=backbone
        )
        self.model.freeze()
        self.model.to(TORCH_DEVICE)

    @torch.no_grad()
    def feed(self, image_pil: PILImage.Image) -> np.ndarray:
        """Feed image to the model and return the multi-class label mask."""
        image_tensor = ProcessImage.parse(MODEL_IMAGE_SIZE, image_pil)
        _, outputs = self.model(image_tensor.unsqueeze(0).to(TORCH_DEVICE))
        return outputs.permute(1, 2, 0).cpu().numpy().squeeze(-1)


class LayoutSeg(pl.LightningModule):
    """Layout segmentation model."""

    def __init__(
        self,
        lr: float = 1e-4,
        backbone: str = "resnet101",
        l1_factor: float = 0.2,
        l2_factor: float = 0.0,
        edge_factor: float = 0.2,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.model = ResPlanarSeg(pretrained=True, backbone=backbone)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.edge_factor = edge_factor
        self.save_hyperparameters()

    def forward(self, inputs: torch.Tensor):
        scores = self.model(inputs)
        _, outputs = torch.max(scores, 1)
        return scores, outputs


class ResPlanarSeg(nn.Module):

    def __init__(self, pretrained: bool = True, backbone: str = "resnet101") -> None:
        super().__init__()
        Backbone: Type[models.ResNet] = getattr(models, backbone)
        self.resnet = Backbone(pretrained=pretrained)
        self.planar_seg = PlanarSegHead(
            bottleneck_channels=37, in_features=self.resnet.fc.in_features
        )

    def forward(self, x: TensorType[3, 320, 320]) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.resnet.conv1(x)  # 64 x 160 x 160
        x = self.resnet.bn1(x)
        e1 = self.resnet.relu(x)
        e2 = self.resnet.maxpool(e1)  # 64 x 80 x 80
        e3 = self.resnet.layer1(e2)  # 256 x 80 x 80
        e4 = self.resnet.layer2(e3)  # 512 x 40 x 40
        e5 = self.resnet.layer3(e4)  # 1024  x 20 x 20
        e6 = self.resnet.layer4(e5)  # 2048 x 10 x 10
        e7 = self.resnet.maxpool(e6)  # 2048 x 5 x 5

        return self.planar_seg(e7, e6, e5)


class PlanarSegHead(nn.Module):
    def __init__(
        self, bottleneck_channels: int, in_features: int = 2048, num_classes: int = 5
    ) -> None:
        super().__init__()
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm2d(in_features)
        self.fc_conv = nn.Conv2d(
            in_features, in_features, kernel_size=1, stride=1, bias=False
        )

        self.clf1 = nn.Conv2d(
            in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False
        )
        self.clf2 = nn.Conv2d(
            in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False
        )
        self.clf3 = nn.Conv2d(
            in_features // 2, bottleneck_channels, kernel_size=1, stride=1, bias=False
        )

        self.dec1 = self.transposed_conv(
            bottleneck_channels, bottleneck_channels, stride=2
        )
        self.dec2 = self.transposed_conv(
            bottleneck_channels, bottleneck_channels, stride=2
        )
        self.dec3 = self.transposed_conv(
            bottleneck_channels, bottleneck_channels, stride=16
        )

        self.fc_stage2 = nn.Conv2d(
            bottleneck_channels, num_classes, kernel_size=1, stride=1, bias=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *features) -> torch.Tensor:
        """Forward pass of the model."""
        e7, e6, e5 = features

        x = self.drop1(e7)
        x = self.fc_conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop2(x)

        c = self.clf1(x)  # 5 x 5 x 5
        d6 = self.dec1(c)  # 5 x 10 x 10

        d6_b = self.clf2(e6)  # 5 x 10 x 10
        d5 = self.dec2(d6_b + d6)  # 5 x 20 x 20

        d5_b = self.clf3(e5)  # 5 x 20 x 20
        d0 = self.dec3(d5_b + d5)  # 5 x 320 x 320

        d = self.fc_stage2(d0)
        return d

    @staticmethod
    def transposed_conv(
        in_channels: int, out_channels: int, stride=2
    ) -> nn.ConvTranspose2d:
        """Transposed conv with same padding."""
        kernel_size, padding = {
            2: (4, 1),
            4: (8, 2),
            16: (32, 8),
        }[stride]
        layer = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        return layer
