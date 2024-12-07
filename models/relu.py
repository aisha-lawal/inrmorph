import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self, layers, time_features):
        super(ReLU, self).__init__()
        self.time_features = time_features
        self.layers = []

        for i in range(len(layers) - 1):
            if i == 0:
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            else:
                self.layers.append(nn.Linear(layers[i] + self.time_features, layers[i + 1]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords, time):

        for layer in self.layers[:-1]:
            coords = torch.nn.functional.relu(layer(coords))
            coords = torch.cat([coords, time], dim=-1)

        return self.layers[-1](coords)
