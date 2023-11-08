import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0, first_layer: bool = False):
        super(SineLayer, self).__init__()
        self.w0 = w0
        self.n = in_dim
        self.c = 6
        self.linear = nn.Linear(in_dim, out_dim)

        self.first_layer = first_layer

        self.init_weights()

    def init_weights(self):
        if self.first_layer:
            nn.init.uniform_(self.linear.weight, -1. / self.n, 1. / self.n)
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(6. / self.n) / self.w0, np.sqrt(6. / self.n) / self.w0)

    def forward(self, x):
        return torch.sin(self.linear(x) * self.w0)


class SirenModel(nn.Module):
    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(SirenModel, self).__init__()
        self.name = 'SIREN'
        self.net = nn.Sequential(SineLayer(in_dim, hidden_dim, w0, True),
                                 SineLayer(hidden_dim, hidden_dim, w0),
                                 SineLayer(hidden_dim, hidden_dim, w0),
                                 SineLayer(hidden_dim, out_dim, w0))

    def forward(self, x):
        return self.net(x)
