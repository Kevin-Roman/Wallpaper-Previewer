"""Main module for layout estimation."""

from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np
import PIL
import torch

from . import core, sequence

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT = Path("./layout_estimation/weights/model_retrained.ckpt")


class LayoutSegmentationLabels(IntEnum):
    WALL_CENTER = 0
    WALL_LEFT = 1
    WALL_RIGHT = 2
    FLOOR = 3
    CEILING = 4

    @classmethod
    def walls(
        cls,
    ) -> tuple[
        "LayoutSegmentationLabels",
        "LayoutSegmentationLabels",
        "LayoutSegmentationLabels",
    ]:
        return cls.WALL_CENTER, cls.WALL_LEFT, cls.WALL_RIGHT


class Predictor:
    """Predictor for layout estimation."""

    def __init__(self, weight_path: Path) -> None:
        # pylint: disable=E1120
        self.model = core.LayoutSeg.load_from_checkpoint(
            checkpoint_path=weight_path, backbone="resnet101"
        )
        self.model.freeze()
        self.model.to(DEVICE)

    @torch.no_grad()
    def feed(self, image: torch.Tensor) -> np.ndarray:
        """Feed image to the model and return the multi-class label mask."""
        _, outputs = self.model(image.unsqueeze(0).to(DEVICE))
        return outputs.permute(1, 2, 0).cpu().numpy().squeeze(-1)

    @torch.no_grad()
    def feed_and_blend(self, image: torch.Tensor, alpha: int = 0.4) -> np.ndarray:
        """Feed image to the model and return the blended output."""
        _, outputs = self.model(image.unsqueeze(0).to(DEVICE))
        label = core.label_as_rgb_visual(outputs.cpu()).squeeze(0)
        blend_output = (image / 2 + 0.5) * (1 - alpha) + (label * alpha)
        return blend_output.permute(1, 2, 0).numpy()


class LayoutEstimator:
    def __init__(self) -> None:
        self.weight_path = WEIGHT
        self.predictor = Predictor(self.weight_path)

    def estimate_layout_coloured(
        self,
        image_path: Path,
        model_image_size: int,
        cat_visual: bool,
        output_folder: Path,
    ) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)

        image, shape = sequence.Image.parse(model_image_size, image_path)

        label = cv2.resize(self.predictor.feed_and_blend(image, alpha=1.0), shape)
        image = cv2.resize((image / 2 + 0.5).permute(1, 2, 0).numpy(), shape)
        if cat_visual:
            output = np.concatenate([image, label], axis=1)
        else:
            output = label
        output_path = output_folder / image_path.name

        cv2.imwrite(str(output_path), (output[..., ::-1] * 255).astype(np.uint8))

    def estimate_walls(
        self,
        image_path: Path,
        model_image_size: int,
    ) -> list[np.ndarray]:
        image, shape = sequence.Image.parse(model_image_size, image_path)

        label_mask = cv2.resize(
            self.predictor.feed(image), shape, interpolation=PIL.Image.NEAREST
        )
        walls_masks = [
            np.isin(label_mask, wall_side)
            for wall_side in LayoutSegmentationLabels.walls()
        ]

        return walls_masks

    @staticmethod
    def find_wall_corners(mask: np.ndarray):
        """Find the corners of the wall in the mask.

        Assumes that the wall is the largest closed quadrilateral in the mask."""
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            assert "No wall found in the image."

        print(f"contours: {len(contours)}")
        contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx) != 4:
            assert "Wall is not a quadrilateral."

        corners = approx.reshape(-1, 2)

        if len(corners) != 4:
            assert "Wall is not a quadrilateral."

        return corners


if __name__ == "__main__":
    # estimate_layout_coloured(
    #     Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg"),
    #     core.MODEL_IMAGE_SIZE,
    #     True,
    #     Path("./output"),
    # )
    layout_estimator = LayoutEstimator()
    walls_masks = layout_estimator.estimate_walls(
        Path("./data/1a98599d3f7d168f2cf53e64ad1dd5c6e95e1b64.jpg"),
        core.MODEL_IMAGE_SIZE,
    )

    cv2.imwrite(
        "./output/wall_layout_segmentation_mask.png",
        walls_masks[0].astype(np.uint8) * 255,
    )
