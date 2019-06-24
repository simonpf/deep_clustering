import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
        super(Encoder, self).__init__()

        self.modules = []

        self.modules += [nn.Conv2d(in_channels, out_channels, kernel_width, padding = 1)]
        self.modules += [nn.BatchNorm2d(out_channels)]
        self.modules += [nn.ReLU()]

        for i in range(1, depth):
            self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
            self.modules += [nn.BatchNorm2d(out_channels)]
            self.modules += [nn.ReLU()]

        self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width,
                                   padding = 1, stride = 2)]

        for i, m in enumerate(self.modules):
            setattr(self, "module_" + str(i), m)

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
        super(Decoder, self).__init__()

        self.modules = []

        self.modules += [nn.ConvTranspose2d(in_channels, out_channels, kernel_width,
                                            padding = 1, output_padding = 1, stride = 2)]
        self.modules += [nn.BatchNorm2d(out_channels)]
        self.modules += [nn.ReLU()]

        for i in range(1, depth):
            self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
            self.modules += [nn.BatchNorm2d(out_channels)]
            self.modules += [nn.ReLU()]

        for i, m in enumerate(self.modules):
            setattr(self, "module_" + str(i), m)

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x
