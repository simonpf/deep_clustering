import os
from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

import torch
import torch.optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from deep_clustering.models.cvae import Cvae, loss_function
from deep_clustering.data import Shapes
from deep_clustering.models.training import train_network

model = Cvae(1, 3, [128, 128])
data = Shapes("/home/simon/src/deep_clustering/data")
dl = DataLoader(data)
optimizer = torch.optim.Adam(model.parameters())
criterion = loss_function
output_path = "cvae"

train_network(data, model, optimizer, criterion, output_path, 2)
x = next(iter(dl))
