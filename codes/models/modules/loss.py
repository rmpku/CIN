import torch
import torch.nn as nn
import torchvision
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp




# -------------Reconstruction  img & msg  loss-----------------------------
class ReconstructionImgLoss(nn.Module):
    def __init__(self, opt, eps=1e-6):
        super(ReconstructionImgLoss, self).__init__()
        self.losstype = opt['loss']['type']['TypeRecImg']
        self.eps = eps
        self.N = opt['network']['input']['in_img_nc'] * opt["network"]['H'] * opt["network"]['W']

    def forward(self, true_img, fake_img):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((fake_img - true_img)**2, (1, 2, 3))) / self.N 
        elif self.losstype == 'l1':
            diff = fake_img - true_img
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3))) / self.N 
        else:
            print("reconstruction loss type error!")
            return 0


class ReconstructionMsgLoss(nn.Module):
    def __init__(self, opt):
        super(ReconstructionMsgLoss, self).__init__()
        self.losstype = opt['loss']['type']['TyptRecMsg']
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, messages, decoded_messages): 
        if self.losstype == 'mse':
            return self.mse_loss(messages, decoded_messages)
        elif self.losstype == 'bce':
            return self.bce_loss(messages, decoded_messages)
        elif self.losstype == 'bce_logits':
            return self.bce_logits_loss(messages, decoded_messages)
        else:
            print("ReconstructionMsgLoss loss type error!")
            return 0



# ------------------Encoded watermarked-image loss-------------------
class EncodedLoss(nn.Module):
    def __init__(self, opt, eps=1e-6):
        super(EncodedLoss, self).__init__()
        self.losstype = opt['loss']['type']['TyprEncoded']
        self.eps = eps
        self.N = opt['network']['input']['in_img_nc'] * opt["network"]['H'] * opt["network"]['W']

    def forward(self, true_img, watermarking_img):
        if self.losstype == 'l2':
            loss_mask = 1
            return torch.mean(torch.sum((loss_mask * watermarking_img - true_img)**2, (1, 2, 3))) / self.N
        elif self.losstype == 'l1':
            diff = watermarking_img - true_img
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3))) / self.N 
        else:
            print("EncodedLoss loss type error!")
            return 0

