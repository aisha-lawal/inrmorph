import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30,
                 is_first=False, is_last=False,
                 init_method='sine', init_gain=1, fbs=None, hbs=None):
        super().__init__()
        self.omega = omega
        self.is_last = is_last  ## no activation
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # init bias
        init_bias_cond(self.linear, fbs, is_first)

    def forward(self, input):
        wx_b = self.linear(input)
        if not self.is_last:
            return finer_activation(wx_b, self.omega)  # no activation for last layer
        return wx_b  # is_last==True


class Finer(nn.Module):
    def __init__(self, in_features=3, out_features=3, hidden_layers=5, hidden_features=256,
                 first_omega=30, hidden_omega=30,
                 init_method='sine', init_gain=1, fbs=None, hbs=None, time_features=64):
        super().__init__()
        self.time_features = time_features
        self.net = []

        # for first layer
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True,
                                   omega=first_omega,
                                   init_method=init_method, init_gain=init_gain, fbs=fbs))

        for i in range(hidden_layers):  # for hidden layers
            current_in_features = hidden_features + (time_features * (i + 1))
            next_out_features = hidden_features + (time_features * (i + 1))
            self.net.append(FinerLayer(current_in_features, next_out_features,
                                       omega=hidden_omega,
                                       init_method=init_method, init_gain=init_gain, hbs=hbs))

        # for output layer
        self.net.append(FinerLayer(next_out_features + time_features, out_features, is_last=True,
                                   omega=hidden_omega,
                                   init_method=init_method, init_gain=init_gain, hbs=hbs))

        self.net = nn.ModuleList(self.net)

    def forward(self, coords, time):
        # for first layer
        x = self.net[0](coords)
        # for hidden layers
        for i, layer in enumerate(self.net[1:-1]):
            x = torch.cat([x, time], dim=-1)
            x = layer(x)

        # last layer
        x = torch.cat([x, time], dim=-1)
        x = self.net[-1](x)
        return x


############### WEIGHTS INITIALIZATION ############################
# weights are initilized same as SIREN
def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)


def init_weights(m, omega=1, c=1, is_first=False):  # Default: Pytorch initialization
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first:  # 1/infeatures for first layer
            bound = 1 / fan_in
        else:
            bound = np.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)


############### BIAS INITIALIZATION ############################
# bias are initialized as a uniform distribution between -k and k
def init_bias(m, k):
    if hasattr(m, 'bias'):
        init.uniform_(m.bias, -k, k)


def init_bias_cond(linear, fbs=None, is_first=True):
    if is_first and fbs != None:
        init_bias(linear, fbs)


############### FINER ACTIVATION FUNCTION ############################
# according to the paper, the activation function is sin(omega * alpha(x) * x), where alpha(x) = |x| + 1
def generate_alpha(x):
    with torch.no_grad():
        return torch.abs(x) + 1


def finer_activation(x, omega=1):
    return torch.sin(omega * generate_alpha(x) * x)
