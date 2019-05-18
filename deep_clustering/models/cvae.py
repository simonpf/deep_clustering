import torch
import torch.nn as nn
import torch.nn.functional as F

class Cvae(nn.Module):
    def __init__(self, channels, latent_dim, arch = [32, 64]):
        super(Cvae, self).__init__()

        #
        # Encoder
        #

        encoder_modules = []
        in_channels = channels
        for c in arch:
            encoder_modules += [nn.Conv2d(in_channels, c, 3, stride = 1)]
            encoder_modules += [nn.BatchNorm2d(c)]
            encoder_modules += [nn.ReLU()]
            encoder_modules += [nn.Conv2d(c, c, 3, stride = 2)]
            encoder_modules += [nn.BatchNorm2d(c)]
            encoder_modules += [nn.ReLU()]
            in_channels = c
        self.encoder = nn.Sequential(*encoder_modules)

        #
        # Latent stuff
        #

        self.latent_mu = nn.Conv2d( in_channels, latent_dim, kernel_size = 3)
        self.latent_logvar = nn.Conv2d(in_channels, latent_dim, kernel_size = 3)

        #
        # Decoder
        #

        decoder_modules = []
        decoder_modules += [nn.Conv2d(latent_dim, in_channels, 3)]
        for c in arch[-2::-1]:
            decoder_modules += [nn.ConvTranspose2d(in_channels, c, 3, stride = 2, output_padding = 1)]
            encoder_modules += [nn.BatchNorm2d(c)]
            decoder_modules += [nn.ReLU()]
            decoder_modules += [nn.ConvTranspose2d(c, c, 3)]
            decoder_modules += [nn.BatchNorm2d(c)]
            decoder_modules += [nn.ReLU()]
            in_channels = c
        decoder_modules += [nn.ConvTranspose2d(c, channels, 3, stride = 2, output_padding = 1)]
        encoder_modules += [nn.BatchNorm2d(channels)]
        decoder_modules += [nn.ReLU()]
        decoder_modules += [nn.ConvTranspose2d(channels, channels, 3)]
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x):
        x = self.encoder(x)
        return self.latent_mu(x), self.latent_logvar(x)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


