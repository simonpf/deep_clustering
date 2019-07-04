import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self, name, path = None):
        super(Model, self).__init__()

        self.name = name
        if path is None:
            try:
                self.path.os.environ["MODEL_PATH"]
            except:
                self.path = "."
        else:
            self.path = path

        output_path = os.path.join(self.path, self.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path


    def train(self,
              data_set,
              optimizer,
              criterion,
              n_epochs,
              dataset_callback):

        cuda = torch.cuda.is_available()

        log_file = os.path.join(self.output_path, "training_log.txt")
        if os.path.isfile(log_file):
            log_file = open(log_file, mode = "r+")
        else:
            log_file = open(log_file, mode = "w")


        for i in range(n_epochs):

            if not dataset_callback is None:
                dataset_callback(data_set)
            data_loader = DataLoader(data_set, batch_size = 64)


            epoch_loss = 0.0

            for j, x in enumerate(data_loader):

                if cuda:
                    x.cuda()

                y = self(x)
                optimizer.zero_grad()
                loss = criterion(x, *y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().float()

                print("Epoch {0}, batch {1}: {2}".format(i, j, loss.detach().float()))


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
