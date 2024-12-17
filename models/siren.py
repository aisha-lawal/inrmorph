import torch.nn as nn
import torch
import numpy as np


class Siren(nn.Module):
    def __init__(self, layers, omega_0, time_features):

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega_0 = omega_0
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
                        -np.sqrt(6 / layers[i]) / self.omega_0,
                        np.sqrt(6 / layers[i]) / self.omega_0,
                    )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords, time):
        coords = torch.sin(self.omega_0 * self.layers[0](coords))
        for layer in self.layers[1:-1]:
            coords = torch.cat([coords, time], dim=-1)
            coords = torch.sin(self.omega_0 * layer(coords))
        coords = torch.cat([coords, time], dim=-1)
        return self.layers[-1](coords)

#time not in final layer
# class Siren(nn.Module):
#
#     def __init__(self, layers, omega_0, time_features):
#
#         super(Siren, self).__init__()
#         # except last two layers, one not concatenated with time and the other is output layer.
#         self.n_layers = len(layers) - 2
#         self.omega_0 = omega_0
#         self.time_features = time_features
#         # Make the layers
#         self.layers = []
#         with torch.no_grad():
#             for i in range(self.n_layers):
#                 if i == 0:
#
#                     self.layers.append(nn.Linear(layers[i], layers[i + 1]))
#                     self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
#                 else:
#                     self.layers.append(nn.Linear(layers[i] + self.time_features, layers[i + 1]))
#                     self.layers[-1].weight.uniform_(
#                         -np.sqrt(6 / layers[i]) / self.omega_0,
#                         np.sqrt(6 / layers[i]) / self.omega_0)
#
#             # for final layer, final layer shouldn't be conatenated with time_features
#             self.layers.append(nn.Linear(layers[self.n_layers], layers[self.n_layers + 1]))
#             self.layers[-1].weight.uniform_(
#                 -np.sqrt(6 / layers[self.n_layers]) / self.omega_0,
#                 np.sqrt(6 / layers[self.n_layers]) / self.omega_0)
#
#         self.layers = nn.Sequential(*self.layers)
#
#     def forward(self, coords, time):
#         coords = torch.sin(self.omega_0 * self.layers[0](coords))
#
#         for layer in self.layers[1:-1]:
#             coords = torch.cat([coords, time], dim=-1)
#             coords = torch.sin(self.omega_0 * layer(coords))
#         return self.layers[-1](coords)

