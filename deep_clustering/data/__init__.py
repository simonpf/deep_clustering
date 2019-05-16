import glob
import os
import torch
from torch.utils.data import Dataset
import scipy as sp
import numpy as np
from PIL import Image

class Shapes(Dataset):
    def __init__(self, data_path):
        self.files = glob.glob(os.path.join(data_path, "img_*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx])
        im = np.array(im, dtype = np.float32)
        im = np.sum(im[:, :, :3], axis = -1)
        if not im.max() == 0.0:
            im /= im.max()
        return im[np.newaxis, :, :]
