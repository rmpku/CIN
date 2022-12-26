import torch.nn as nn
from models.modules.InvArch import InvArch
from models.modules.Conv1x1 import InvertibleConv1x1



class IM(nn.Module):
    def __init__(self, opt):
        super(IM, self).__init__()
        #
        self.operations = nn.ModuleList()
        #
        for _ in range(opt['network']['InvBlock']['block_num']):
            #
            if opt['network']['InvBlock']['downscaling']['use_conv1x1']:
                a = InvertibleConv1x1(opt['network']['InvBlock']['split1_img'] + \
                    opt['network']['InvBlock']['split2_repeat'])
                self.operations.append(a)
            #
            b = InvArch(opt['network']['InvBlock']['split1_img'], opt['network']['InvBlock']['split2_repeat'])
            self.operations.append(b)

    def forward(self, x, rev=False):
        if not rev:
            #
            for op in self.operations:
                x = op.forward(x, rev)
            #
        else:
            for op in reversed(self.operations):
                x = op.forward(x, rev)
        return x


    