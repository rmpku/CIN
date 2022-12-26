import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np




#
class NSM(nn.Module):
    def __init__(self, opt, device):
        super(NSM, self).__init__()
        # 
        self.opt = opt
        self.device = device
        #
        self.resnet = models.resnet50(pretrained=False).to(device)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2).to(device)
        self.dct = DCT().to(device)

    def forward(self, noised_img):
        #
        with torch.no_grad():
            dct_blocks = self.dct(noised_img)
            out = self.resnet(dct_blocks)

        return out


class DctBasic(nn.Module):
    def __init__(self):
        super(DctBasic, self).__init__()

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
                                  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
                                  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
                                  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv
    
    def dct(self, image):
        # coff for dct and idct
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image.shape[2] // 8
        image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image_dct

    def yuv_dct(self, image):
        # clamp and convert from [-1,1] to [0,255]
        image = (image.clamp(-1, 1) + 1) * 255 / 2

        # pad the image so that we can do dct on 8x8 blocks
        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

        # convert to yuv
        image_yuv = self.rgb2yuv(image)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        # apply dct
        image_dct = self.dct(image_yuv)

        return image_dct


class DCT(DctBasic):
    def __init__(self):
        super(DCT, self).__init__()

    def forward(self, image):
        # [-1,1] to [0,255], rgb2yuv, dct
        image_dct = self.yuv_dct(image)

        # normliz
        out = nn.functional.normalize(image_dct)

        return out.clamp(-1, 1)



def discim_accurary(out, label, opt):

    pred = torch.argmax(out, dim=1)
    num_correct = (pred == label).sum().item()
    if len(pred) < opt['train']['batch_size']:
        accuracy = num_correct / len(pred)
    else:
        accuracy = num_correct / (opt['train']['batch_size'])
    return accuracy * 100

