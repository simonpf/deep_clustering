import os
from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")
    ip.magic("%env OMP_NUM_THREADS=2")

import torch
import torch.optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from deep_clustering.models.cae import Cae, loss_function
from deep_clustering.data import Shapes
from deep_clustering.models.training import train_network

model = Cae(1, 10, [(32, 1), (32, 1), (64, 1), (64, 1)])
data = Shapes("/home/simonpf/src/deep_clustering/data")
dl = DataLoader(data)
optimizer = torch.optim.Adam(model.parameters())
criterion = loss_function
output_path = "vae"

train_network(data, model, optimizer, criterion, output_path, 1)
i = iter(dl)
x = next(i)
x = next(i)
x = next(i)
x = next(i)
x = next(i)
x = next(i)
xr = model(x)[0].detach()

#optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
#train_network(data, model, optimizer, criterion, output_path, 5)
def save_recs():
    x = next(i)
    xr = model(x)[0].detach()
    plt.matshow(x[0, 0, :, :])
    f = plt.gcf()
    f.savefig("x.png")
    plt.matshow(xr[0, 0, :, :])
    f = plt.gcf()
    f.savefig("xr.png")
