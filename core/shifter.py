import os
from settings import Config
from utils import load_numpy


class Shifter:
    def __init__(self, vectors_dir=Config.shifting.vectors_path, ext=Config.shifting.extension):
        self.fnames = [file for file in os.listdir(vectors_dir) if file.endswith(ext)]
        self.vectors = {}
        for file in self.fnames:
            path = os.path.join(vectors_dir, file)
            name = file.replace(ext, '')
            vec = load_numpy(path, Config.generation.device)
            vec = vec.unsqueeze(0)
            self.vectors[name] = vec

    def __call__(self, w, direction, amount):
        """
        Shifts latent vector w in the given direction
        :param w: input vector
        :param direction: name of a key in vectors dictionary
        :param amount: scale factor for direction
        :return: shifted vector
        """
        w = w + self.vectors[direction] * amount
        return w