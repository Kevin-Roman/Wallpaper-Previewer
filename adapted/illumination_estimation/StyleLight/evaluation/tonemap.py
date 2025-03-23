from pathlib import Path

import numpy as np

from external.skylibs.hdrio import imsave

from .util import PanoramaHandler, TonemapHDR


def tonemap_hdr_panorama(hdr_panorama: np.ndarray, temp_save_path: Path) -> None:
    tone = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    handle = PanoramaHandler()

    hdr_panorama_path = str(temp_save_path.resolve())
    imsave(hdr_panorama_path, hdr_panorama)
    exr = handle.read_hdr(hdr_panorama_path)
    img, _ = tone(exr, clip=False)
    return img
