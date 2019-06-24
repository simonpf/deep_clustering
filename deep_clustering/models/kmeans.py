import sklearn
import sklearn.cluster
import torch
import torch.nn as nn
import torch.nn.functional as F

class KMeans(nn.Module):
    def __init__(self, k, latent_dimensions):
        super(KMeans, self).__init__()
        self.k = k
        self.latent_dimensions = latent_dimensions
        self.centroids = nn.Parameter(torch.tensor((k, latent_dimensions),
                                                    dtype = torch.float))

    def initialize(self, x):
        self.kmeans = sklearn.cluster.KMeans(self.k).fit(x.detach().numpy())
        self.centroids.data = torch.tensor(self.kmeans.cluster_centers_)

    def loss(self, x):
        x = x.view(-1, self.latent_dimensions, 1)

        # Compute q: eq. (1)
        d = torch.transpose(self.centroids, 0, 1).view(1, self.latent_dimensions, self.k)
        d = d - x
        d = torch.sum(d ** 2, 1)
        q = 1.0 / (1.0 + d)
        q = q / torch.sum(q, 1, True)

        f = torch.sum(q, 0, True)
        p = q ** 2 / f
        p = p / torch.sum(p, 1, True)

        l = torch.sum(p * torch.log(p / q), (0, 1))
        return l
