import torch
import pickle
import numpy as np
from settings import Config

import sys
sys.path.append("stylegan2-ada-pytorch")


class Generator:
    def __init__(self, pickle_path=Config.generation.model_path, device=Config.generation.device):
        self.device = device
        with open(pickle_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(device)
            self.G.eval()

    def truncate_w(self, w, truncation_psi=1):
        """
        Performs linear interpolation between a given w and the average w vector
        truncation_psi=1 means no truncation
        """
        w_avg = self.G.mapping.w_avg
        w = truncation_psi*(w - w_avg) + w_avg
        return w

    def get_z(self, seed):
        """
        Generates latent vector z from a random seed
        """
        z = np.random.RandomState(seed).randn(1, self.G.z_dim)
        return z

    def get_w(self, z, truncation_psi=1):
        """
        Generates w vector using latent vector z
        """
        z = torch.tensor(z).to(self.device)
        with torch.no_grad():
            w = self.G.mapping(z, None)
            w = self.truncate_w(w, truncation_psi)
        return w

    def get_img(self, w):
        """
        Generates image using latent vector w
        """
        with torch.no_grad():
            img = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        return img