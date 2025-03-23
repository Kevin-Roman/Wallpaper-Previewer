import math

import cv2
import numpy as np

from .skylibs.envmaps.environmentmap import EnvironmentMap


def warp_hdr_panorama(hdr_panorama: np.ndarray, beta: float = -0.6) -> np.ndarray:
    env = EnvironmentMap(hdr_panorama, "latlong")

    h, w, c = env.data.shape

    warp_image = np.zeros_like(env.data)

    # compute
    x = np.zeros((h, w))
    y = np.zeros((h, w))
    z = np.zeros((h, w))
    for ii in range(h):
        for jj in range(w):
            # direction of original coord
            x[ii, jj] = np.sin(ii * 1.0 / h * math.pi) * np.cos(
                jj * 2.0 / w * math.pi
            )  # +beta-beta
            y[ii, jj] = np.sin(ii * 1.0 / h * math.pi) * np.sin(jj * 2.0 / w * math.pi)
            z[ii, jj] = np.cos(ii * 1.0 / h * math.pi)

    a = x**2 + y**2 + z**2
    b = 2 * x * beta
    c = beta**2 - 1

    t = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # # intersection
    x_ = x * t + beta  # -beta
    y_ = y * t
    z_ = z * t

    # # norm
    mag = np.sqrt(x_**2 + y_**2 + z_**2)
    x_norm = x_ / mag
    y_norm = y_ / mag
    z_norm = z_ / mag

    # angle (i.j)--> (x_norm, y_norm, z_norm)

    theta = np.arccos(z_norm)
    phi = np.arctan2(y_norm, x_norm)  # -math.pi/2

    theta_index = (theta / math.pi * h).astype(int)
    phi_index = (phi / (2 * math.pi) * w).astype(int)

    # remove black pixels
    remove_black_pixels = True
    if remove_black_pixels:
        img_hdr = env.data[:214, :, :]

        img_hdr = cv2.resize(img_hdr, (512, 256))
    else:
        img_hdr = env.data

    for ii in range(h):
        for jj in range(w):
            warp_image[ii, jj, :] = img_hdr[
                theta_index[ii, jj] - 1, phi_index[ii, jj] - 1, :
            ]

    return warp_image.astype(np.float32)
