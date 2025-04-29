import numpy as np

from envmap import rotation_matrix

from .envmaps.environmentmap import EnvironmentMap

DEGREES = 60


def crop2pano(env: EnvironmentMap, image: np.ndarray) -> np.ndarray:
    dcm = rotation_matrix(
        azimuth=0,
        # elevation=np.pi/8,
        elevation=0,
        # roll=np.pi/12)
        roll=0,
    )

    h, w, _ = image.shape

    masked_pano = env.Fov2MaskedPano(
        image,
        vfov=DEGREES,  # DEGREES
        rotation_matrix=dcm,
        ar=4.0 / 3.0,
        # resolution=(640, 480),
        resolution=(w, h),
        # resolution=(512, 256),
        projection="perspective",
        mode="normal",
    )

    masked_pano = masked_pano.astype("uint8")
    return masked_pano
