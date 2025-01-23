import pathlib

import cv2
import numpy as np
import torch

from . import core, sequence

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self, weight_path: str):
        self.model = core.LayoutSeg.load_from_checkpoint(
            weight_path, backbone="resnet101"
        )
        self.model.freeze()
        self.model.to(DEVICE)

    @torch.no_grad()
    def feed(self, image: torch.Tensor, alpha: int = 0.4) -> np.ndarray:
        _, outputs = self.model(image.unsqueeze(0).to(DEVICE))
        label = core.label_as_rgb_visual(outputs.cpu()).squeeze(0)
        blend_output = (image / 2 + 0.5) * (1 - alpha) + (label * alpha)
        return blend_output.permute(1, 2, 0).numpy()


def estimate_layout_save_image(path, weight, image_size, cat_visual, output_folder):
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    predictor = Predictor(weight_path=weight)
    images = sequence.ImageFolder(image_size, path)

    for image, shape, path in images:
        label = cv2.resize(predictor.feed(image, alpha=1.0), shape)
        image = cv2.resize((image / 2 + 0.5).permute(1, 2, 0).numpy(), shape)
        if cat_visual:
            output = np.concatenate([image, label], axis=1)
        else:
            output = label
        output_path = output_folder / path.name
        cv2.imwrite(str(output_path), (output[..., ::-1] * 255).astype(np.uint8))


# def image(path, weight, image_size):
#     logger.info("Press `q` to exit the sequence inference.")
#     predictor = Predictor(weight_path=weight)
#     images = sequence.ImageFolder(image_size, path)

#     for image, shape, _ in images:
#         output = cv2.resize(predictor.feed(image), shape)
#         cv2.imshow("layout", output[..., ::-1])
#         if cv2.waitKey(0) & 0xFF == ord("q"):
#             break


# def video(device, path, weight, image_size):
#     predictor = Predictor(weight_path=weight)
#     stream = sequence.VideoStream(image_size, path, device)

#     for image in stream:
#         output = cv2.resize(predictor.feed(image), stream.origin_size)
#         cv2.imshow("layout", output[:, :, ::-1])
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    # video(0, None, "./layout_estimation/checkpoint/model_retrained.ckpt", 320)
    estimate_layout_save_image(
        "./data/0a578e8af1642d0c1e715aaa04478858ac0aab01.jpg",
        "./layout_estimation/weights/model_retrained.ckpt",
        320,
        False,
        "./output/",
    )
