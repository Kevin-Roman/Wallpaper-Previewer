"""Core module for layout estimation."""

import pytorch_lightning as pl
import torch
from torchtyping import TensorType

from .model import ResPlanarSeg

MODEL_IMAGE_SIZE = 320


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scores = self.model(inputs)
        _, outputs = torch.max(scores, 1)
        return scores, outputs


def label_as_rgb_visual(
    x: TensorType["N", "H", "W"] | TensorType["N", 1, "H", "W"]  # noqa: F821
) -> TensorType["N", "C", "H", "W"]:  # noqa: F821
    """Make segment tensor into a coloured image"""

    colours_rgb = [
        [0.9764706, 0.27058825, 0.3647059],
        [1.0, 0.8980392, 0.6666667],
        [0.5647059, 0.80784315, 0.70980394],
        [0.31764707, 0.31764707, 0.46666667],
        [0.94509804, 0.96862745, 0.8235294],
    ]

    if x.dim() == 4:
        x = x.squeeze(1)
    assert x.dim() == 3

    n, h, w = x.size()
    palette = torch.tensor(colours_rgb).to(x.device)
    labels = torch.arange(x.max() + 1).to(x.device)

    canvas = torch.zeros(n, h, w, 3).to(x.device)
    for color, lbl_id in zip(palette, labels):
        if canvas[x == lbl_id].size(0):
            canvas[x == lbl_id] = color

    return canvas.permute(0, 3, 1, 2)
