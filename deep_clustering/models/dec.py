import torch
import torch.nn as nn
from deep_clustering.models.cae import loss_function


class Dec(nn.Module):
    def __init__(self, cae, kmeans):
        super(Dec, self).__init__()
        self.cae = cae
        self.kmeans = kmeans

    def forward(self, x):
        z = self.cae.encode(x)
        xr = self.cae.decode(z)
        return (xr, z)

    def loss(self, x, xr, z):

        z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.kmeans.latent_dimensions)
        return loss_function(x, xr) + self.kmeans.loss(z)

