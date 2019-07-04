import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_clustering.models.modules import Model, Encoder, Decoder

class Cae(Model):
    def __init__(self, name, channels, latent_dim, arch = [(32, 2), (32, 2)]):
        super(Cae, self).__init__(name)

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
        self._train_level = 0

    def _apply_noise(self, x, p = 0.1, noise_type = None):
        if self._training:
            r = torch.rand(x.size())
            mask = r > p
            return r * x
        else:
            return x


    def encode(self, x):
        for i, m in enumerate(self.encoders):
            if i == self._train_level:
                x = self._apply_noise(x, 0.5)
                x = m(x)
                return x
            else:
                x = m(x)
        return self.latent_mu(x)

    def decode(self, mu):
        n = len(self.encoders)
        if self._train_level >= n:
            z = self.latent_z(mu)
        else:
            z = mu
        for m in self.decoders[max(n - self._train_level - 1, 0):]:
            z = m(z)
        z = self.channel_reduction(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu = self.encode(x)
        return (self.decode(mu),)

    def visualize_filter(self, j, n_steps = 100, step_length = 0.1):
        x = torch.randn((1, 1, 64, 64), requires_grad = True)
        x = torch.zeros((1, 1, 64, 64), requires_grad = True)
        
        for i in range(n_steps):
            y = self.encode(x)
            n = y.size()[-1]
            c = torch.sum(y[0, j, n // 2, n // 2])
            c.backward()
            print(c)
            x.data = x.data + step_length * x.grad
        return x

        





def loss_function(x, x_r):
    return F.binary_cross_entropy(x_r, x, reduction='sum')
