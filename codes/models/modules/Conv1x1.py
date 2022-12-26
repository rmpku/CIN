import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, rev):
        w_shape = self.w_shape
        if not rev:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, rev=False):
        weight = self.get_weight(rev)
        if not rev:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z
