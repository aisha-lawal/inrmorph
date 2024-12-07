import torch
import torch.nn as nn
import numpy as np


class Siren(nn.Module):

    def __init__(self, layers, omega, time_features):

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.time_features = time_features

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            with torch.no_grad():
                if i == 0:

                    self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                    self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                else:
                    self.layers.append(nn.Linear(layers[i] + self.time_features, layers[i + 1]))
                    self.layers[-1].weight.uniform_(
                        -np.sqrt(6 / layers[i]) / self.omega,
                        np.sqrt(6 / layers[i]) / self.omega,
                    )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords, time):
        coords = torch.sin(self.omega * self.layers[0](coords))

        for layer in self.layers[1:-1]:
            coords = torch.cat([coords, time], dim=-1)
            coords = torch.sin(self.omega * layer(coords))
        coords = torch.cat([coords, time], dim=-1)
        return self.layers[-1](coords)
