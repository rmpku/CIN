import torch.nn as nn


class Identity(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, encoded, cover_img=None):
        out = encoded.clone()
        return out
