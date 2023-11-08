import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DARTSLayer(nn.Module):
    def __init__(self, num_inp, num_out):
        super(DARTSLayer, self).__init__()

        self.act_list = [F.relu, torch.tanh, torch.sin]
        self.num_act = len(self.act_list)
        self.num_inp = num_inp
        self.num_out = num_out

        self.num_of_softmax_weights = self.num_act * self.num_out
        self.activation_weights = nn.Parameter(torch.ones(self.num_out, self.num_act))
        self.linear = nn.Linear(num_inp, self.num_out)

        self.act_tau = 1.0

    def forward(self, x):
        self.act_tau = self.act_tau * .99
        linear = self.linear(x)

        temp = []
        for index, act in enumerate(self.act_list):
            temp.append(act(linear))

        activated_linear = torch.stack(temp, dim=2)

        act_logits = F.gumbel_softmax(self.activation_weights, dim=1, tau=self.act_tau, hard=False)

        activation_of_linear_outputs = activated_linear * act_logits

        activation_of_linear_outputs = torch.sum(activation_of_linear_outputs, dim=2)

        return activation_of_linear_outputs


class DARTS(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, w0=30):
        super(DARTS, self).__init__()
        self.name = 'DARTS'
        self.net = nn.Sequential(
            DARTSLayer(in_dim, hidden_dim),
            DARTSLayer(hidden_dim, hidden_dim),
            DARTSLayer(hidden_dim, hidden_dim),
            DARTSLayer(hidden_dim, hidden_dim),
            DARTSLayer(hidden_dim, out_dim)
        )

        with torch.no_grad():
            self.net[0].linear.weight.uniform_(-1. / in_dim, 1. / in_dim)
            self.net[1].linear.weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                               np.sqrt(6. / hidden_dim) / w0)
            self.net[2].linear.weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                               np.sqrt(6. / hidden_dim) / w0)
            self.net[3].linear.weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                               np.sqrt(6. / hidden_dim) / w0)
            self.net[4].linear.weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                               np.sqrt(6. / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)
