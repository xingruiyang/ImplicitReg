import json
import numpy as np
import torch
import torch.nn as nn


class ImplicitNet(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 use_tanh=True,
                 clamp_dist=-1):
        super(ImplicitNet, self).__init__()

        dims = [latent_dim+3] + hidden_dims + [1]
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            setattr(self, "lin_"+str(layer), lin)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.act_fn = nn.LeakyReLU(negative_slope=0.01)
        self.clamp_dist = clamp_dist
        self.clamp = clamp_dist > 0

    def forward(self, inputs):
        x = inputs
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin_" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act_fn(x)
        x = self.tanh(x) if self.use_tanh else x
        # return x*self.clamp_dist if self.clamp else x
        # return torch.clamp(x, -self.clamp_dist, self.clamp_dist) if self.clamp else x
        return x * self.clamp_dist
