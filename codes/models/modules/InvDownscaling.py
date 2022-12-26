from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F




#
class InvDownscaling(nn.Module):
    def __init__(self, opt):
        super(InvDownscaling, self).__init__()
        self.down_opt = opt['network']['InvBlock']['downscaling']
        #
        self.in_nc = self.down_opt['in_nc']
        self.operations = nn.ModuleList()
        #
        if self.down_opt['type'] == 'haar':
            for i in range(self.down_opt['down_num']):
                b = HaarDownsampling(self.down_opt['current_cn'])
                self.operations.append(b)
                self.down_opt['current_cn'] = int(self.in_nc / self.down_opt['scale'] / self.down_opt['scale'])

    def forward(self, x, rev=False):
        if not rev:
            for op in self.operations:
                x = op.forward(x, rev)
        else:
            for op in reversed(self.operations):
                x = op.forward(x, rev)
        return x



class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

 


