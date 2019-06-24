import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_clustering.models.modules import Encoder, Decoder

class Cae(nn.Module):
    def __init__(self, channels, latent_dim, arch = [(32, 2), (32, 2)]):
        super(Cae, self).__init__()

        #
        # Encoder
        #

        self.encoders = []
        in_channels = channels
        for i, (c, d) in enumerate(arch):
            self.encoders += [Encoder(in_channels, c, d)]
            setattr(self, "encoder_" + str(i), self.encoders[-1])
            in_channels = c

        #
        # Latent stuff
        #

        self.latent_mu = nn.Conv2d( in_channels, latent_dim, kernel_size = 1)
        self.latent_logvar = nn.Conv2d(in_channels, latent_dim, kernel_size = 1)

        self.latent_z = nn.Conv2d(latent_dim, in_channels, kernel_size = 1)

        #
        # Decoder
        #

        self.decoders = []
        for i, (c, d) in enumerate(arch[::-1]):
            self.decoders += [Decoder(in_channels, c, d)]
            setattr(self, "decoder_" + str(i), self.decoders[-1])
            in_channels = c

        self.channel_reduction = nn.Conv2d(in_channels, channels, 3, padding = 1)

    def encode(self, x):
        for m in self.encoders:
            x = m(x)
        return self.latent_mu(x)

    def decode(self, mu):
        z = self.latent_z(mu)
        for m in self.decoders:
            z = m(z)
        z = self.channel_reduction(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu = self.encode(x)
        return (self.decode(mu),)

def loss_function(x, x_r):
    return F.binary_cross_entropy(x_r, x, reduction='sum')
