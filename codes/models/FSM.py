import torch.nn as nn
import torch

#
class FSM(nn.Module):
    def __init__(self, opt):
        super(FSM, self).__init__()
        #
        self.split1_img = opt['network']['InvBlock']['split1_img']
        self.strength_factor = opt['noise']['StrengthFactor']['S']

    def forward(self, encoded_img, cover_down=None, rev=False):
        #
        if not rev:
            # 
            msg = encoded_img[:, self.split1_img:, :, :]
            out = cover_down + self.strength_factor * msg
            return out
        else:            
            # msg copy
            out = torch.cat((encoded_img, encoded_img), dim=1)
            return out

