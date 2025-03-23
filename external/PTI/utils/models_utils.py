import functools
import os
import pickle
import sys

import torch
from ..configs import global_config, paths_config
from constants import STYLEGAN2_ADA_FFHQ_ILLUMINATION_ESTIMATION_WEIGHTS

sys.path.append(os.path.join("external", "PTI"))


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f"{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt"
    with open(new_G_path, "rb") as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G():
    with open(STYLEGAN2_ADA_FFHQ_ILLUMINATION_ESTIMATION_WEIGHTS, "rb") as f:
        old_G = pickle.load(f)["G_ema"].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G
