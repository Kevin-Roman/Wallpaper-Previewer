import os
from pathlib import Path
from random import choice
from string import ascii_uppercase

import numpy as np
from PIL import Image as PILImage

from external.PTI.configs import global_config
from hdrio import imsave

from .training.coaches.my_coach import MyCoach


def generate_hdr_panorama(image: PILImage.Image):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = global_config.cuda_visible_devices

    global_config.run_name = "".join(choice(ascii_uppercase) for _ in range(12))
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    coach = MyCoach(np.array(image))

    return coach.train()


if __name__ == "__main__":
    hdr_panorama = generate_hdr_panorama(
        PILImage.open(Path("./data/0a578e8af1642d0c1e715aaa04478858ac0aab01.jpg"))
    )

    if hdr_panorama is not None:
        print(hdr_panorama.dtype)
        imsave("./temp/test.exr", hdr_panorama)
