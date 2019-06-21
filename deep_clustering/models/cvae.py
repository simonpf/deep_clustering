import torch
import torch.nn as nn
import torch.nn.functional as F

# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
#         super(Encoder, self).__init__()

#         self.modules = []

#         self.modules += [nn.Conv2d(in_channels, out_channels, kernel_width, padding = 1)]
#         setattr(self, "conv2d_" + str(0), self.modules[-1])
#         self.modules += [nn.BatchNorm2d(out_channels)]
#         setattr(self, "batch_norm_" + str(0), self.modules[-1])
#         self.modules += [nn.ReLU()]

#         for i in range(1, depth):
#             self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
#             setattr(self, "conv2d_" + str(i), self.modules[-1])
#             self.modules += [nn.BatchNorm2d(out_channels)]
#             setattr(self, "batch_norm_" + str(i), self.modules[-1])
#             self.modules += [nn.ReLU()]

#     def forward(self, x):
#         for m in self.modules:
#             x = m(x)

#         x, indices = F.max_pool2d(x, kernel_size = 2, stride = 2, return_indices = True)
#         return x, indices

# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
#         super(Decoder, self).__init__()

#         self.modules = []

#         self.modules += [nn.Conv2d(in_channels, out_channels, kernel_width, padding = 1)]
#         setattr(self, "conv2d_" + str(0), self.modules[-1])
#         self.modules += [nn.BatchNorm2d(out_channels)]
#         setattr(self, "batch_norm_" + str(0), self.modules[-1])
#         self.modules += [nn.ReLU()]

#         for i in range(1, depth):
#             self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
#             setattr(self, "conv2d_" + str(i), self.modules[-1])
#             self.modules += [nn.BatchNorm2d(out_channels)]
#             setattr(self, "batch_norm_" + str(i), self.modules[-1])
#             self.modules += [nn.ReLU()]

#     def forward(self, x, indices, dim):
#         x = F.max_unpool2d(x, indices, kernel_size = 2, stride = 2, output_size = dim)
#         for m in self.modules:
#             x = m(x)
#         return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
        super(Encoder, self).__init__()

        self.modules = []

        self.modules += [nn.Conv2d(in_channels, out_channels, kernel_width, padding = 1)]
        setattr(self, "conv2d_" + str(0), self.modules[-1])
        self.modules += [nn.BatchNorm2d(out_channels)]
        setattr(self, "batch_norm_" + str(0), self.modules[-1])
        self.modules += [nn.ReLU()]

        for i in range(1, depth):
            self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
            setattr(self, "conv2d_" + str(i), self.modules[-1])
            self.modules += [nn.BatchNorm2d(out_channels)]
            setattr(self, "batch_norm_" + str(i), self.modules[-1])
            self.modules += [nn.ReLU()]

        self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1, stride = 2)]

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
        super(Decoder, self).__init__()

        self.modules = []

        self.modules += [nn.ConvTranspose2d(in_channels, out_channels, kernel_width, padding = 1,
                                            output_padding = 1, stride = 2)]
        setattr(self, "conv2d_" + str(0), self.modules[-1])
        self.modules += [nn.BatchNorm2d(out_channels)]
        setattr(self, "batch_norm_" + str(0), self.modules[-1])
        self.modules += [nn.ReLU()]

        for i in range(1, depth):
            self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
            setattr(self, "conv2d_" + str(i), self.modules[-1])
            self.modules += [nn.BatchNorm2d(out_channels)]
            setattr(self, "batch_norm_" + str(i), self.modules[-1])
            self.modules += [nn.ReLU()]

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

class Cvae(nn.Module):
    def __init__(self, channels, latent_dim, arch = [(32, 2), (32, 2)]):
        super(Cvae, self).__init__()

        #
        # Encoder
        #

        self.encoders = []
        in_channels = channels
        for i, (c, d) in enumerate(arch):
            self.encoders += [Encoder(in_channels, c, d)]
            setattr(self, "endcoder_" + str(i), self.encoders[-1])
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
        indices = []
        dims = []
        for m in self.encoders:
            dims += [x.size()]
            x = m(x)
        return self.latent_mu(x), self.latent_logvar(x)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.latent_z(z)
        for m in self.decoders:
            z = m(z)
        z = self.channel_reduction(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


