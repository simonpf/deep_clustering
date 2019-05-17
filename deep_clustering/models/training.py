import torch
import torch.cuda
from torch.utils.data import DataLoader
import os
import glob
import numpy as np

def load_most_recent(model, output_path):
    model_files = glob.glob(os.path.join(output_path, "model_*.pt"))
    if len(model_files) == 0:
        print("No model found in {}.".format(output_path))
        return None
    indices = np.array([int(os.path.basename(m).split("_")[1].split(".")[0]) for m in model_files])
    print(indices)
    i = np.argmax(indices)

    print("Loading most recent model {}.".format(model_files[i]))
    model.load_state_dict(torch.load(model_files[i]))
    model.eval()


def train_network(data_set,
                  model,
                  optimizer,
                  criterion,
                  output_path,
                  n_epochs ,
                  dataset_callback = None):


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cuda = torch.cuda.is_available()

    log_file = os.path.join(output_path, "training_log.txt")
    if os.path.isfile(log_file):
        log_file = open(log_file, mode = "r+")
    else:
        log_file = open(log_file, mode = "w")


    for i in range(n_epochs):

        if not dataset_callback is None:
            dataset_callback(data_set)
        data_loader = DataLoader(data_set, batch_size = 32)


        epoch_loss = 0.0

        for j, x in enumerate(data_loader):

            if cuda:
                x.cuda()

            x_r, mu, logvar = model(x)
            dn = (x.size()[-1] - x_r.size()[-1]) // 2
            print(x_r.size(), x.size(), dn)
            optimizer.zero_grad()
            loss = criterion(x_r, x[:, :, dn : -dn, dn : -dn], mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().float()

            print("Epoch {0}, batch {1}: {2}".format(i, j, loss.detach().float()))
