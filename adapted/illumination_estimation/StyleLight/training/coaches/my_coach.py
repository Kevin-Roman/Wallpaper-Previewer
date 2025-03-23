import numpy as np
import PIL.Image
import torch
from PIL import ImageDraw
from tqdm import tqdm

from external.PTI.configs import global_config, hyperparameters
from external.PTI.utils.log_utils import log_images_from_w

from ...skylibs.demo_crop import crop2pano
from ...skylibs.envmaps.environmentmap import EnvironmentMap
from .base_coach import BaseCoach

name = "name"


class MyCoach(BaseCoach):
    def __init__(self, source_image: np.ndarray) -> None:
        super().__init__(data_loader=None, use_wandb=False)
        self.source_image = source_image

    def train(self) -> np.ndarray | None:
        use_ball_holder = True
        is_128x256 = False

        env = EnvironmentMap(256, "latlong")
        image_crop2pano = crop2pano(env, self.source_image)
        image = 2 * (image_crop2pano / 255.0) - 1

        image = torch.tensor(
            image.transpose([2, 0, 1]), device=global_config.device
        )  # /255.0     ################################### 0-1
        image = image.unsqueeze(0).to(torch.float32)

        self.restart_training()

        mask_fname = "./adapted/illumination_estimation/StyleLight/crop60_256x512.jpg"

        mask_pil = PIL.Image.open(mask_fname).convert("RGB")

        if is_128x256:
            mask_pil = mask_pil.resize((256, 128), PIL.Image.LANCZOS)
        else:
            mask_pil = mask_pil.resize((512, 256), PIL.Image.LANCZOS)

        mask_pil_sum_c = np.sum(mask_pil, axis=2)
        mask_pil_sum_c_row = np.sum(mask_pil_sum_c, axis=1)
        mask_pil_sum_c_col = np.sum(mask_pil_sum_c, axis=0)
        row_min = np.argwhere(mask_pil_sum_c_row).min() + 10  # 128
        row_max = np.argwhere(mask_pil_sum_c_row).max() - 5

        col_min = np.argwhere(mask_pil_sum_c_col).min() + 10  # 256
        col_max = np.argwhere(mask_pil_sum_c_col).max() - 10

        img1 = ImageDraw.Draw(mask_pil)
        img1.rectangle(
            [(col_min, row_min), (col_max, row_max)],
            fill=(255, 0, 0),
            outline="red",
        )

        bbox = [row_min, row_max, col_min, col_max]

        w_pivot = self.calc_inversions(image, name, bbox)
        w_pivot = w_pivot.to(global_config.device)
        log_images_counter = 0
        real_images_batch = image.to(global_config.device)
        real_images_batch = real_images_batch[
            :, :, bbox[0] : bbox[1], bbox[2] : bbox[3]
        ]

        for i in tqdm(range(hyperparameters.max_pti_steps)):  # max_pti_steps = 350
            generated_images = self.forward(w_pivot)
            generated_images = torch.clip(generated_images, -1, 1)
            generated_images = generated_images[
                :, :, bbox[0] : bbox[1], bbox[2] : bbox[3]
            ]

            loss, l2_loss_val, loss_lpips = self.calc_loss(
                generated_images,
                real_images_batch,
                name,
                self.G,
                use_ball_holder,
                w_pivot,
            )

            self.optimizer.zero_grad()

            if (
                loss_lpips <= hyperparameters.LPIPS_value_threshold
            ):  # LPIPS_value_threshold = 0.06
                break

            loss.backward()
            self.optimizer.step()

            # locality_regularization_interval = 1
            # training_step = 1
            use_ball_holder = (
                global_config.training_step
                % hyperparameters.locality_regularization_interval
                == 0
            )

            # image_rec_result_log_snapshot = 100
            if (
                self.use_wandb
                and log_images_counter % global_config.image_rec_result_log_snapshot
                == 0
            ):
                log_images_from_w([w_pivot], self.G, [name])

            global_config.training_step += 1
            log_images_counter += 1

        self.image_counter += 1

        generated_images = self.forward(w_pivot)
        gamma = 2.4
        limited = True

        if limited:
            generated_images_singlemap = torch.mean(
                generated_images, dim=1, keepdim=True
            )
            r_percentile = torch.quantile(generated_images_singlemap, 0.999)
            light_mask = (generated_images_singlemap > r_percentile) * 1.0
            hdr = torch.clip(generated_images * (1 - light_mask), -1, 1) + torch.clip(
                generated_images * light_mask, -1, 2
            )

        else:
            hdr = torch.clip(generated_images, -1, 1)

        full = (hdr + 1) / 2
        inv_tone = True
        if inv_tone:
            full_inv_tonemap = torch.pow(full / 5, gamma)
            img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        else:
            img_hdr_np = full.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

        return img_hdr_np
