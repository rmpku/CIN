import torch
import torch.nn as nn
from models.IM import IM
from models.Noise_pool import Noise_pool
from models.FSM import FSM
from models.DEM import DEM
from models.modules.InvDownscaling import InvDownscaling
from models.NIAM.NIAM import NIAM
from models.NSM_module import NSM


# ---------------------------------------------------------------------------------
class CINBasic(nn.Module):
    def __init__(self, opt, device):
        super(CINBasic, self).__init__()
        # 
        self.opt = opt
        self.device = device
        self.h, self.w = opt['network']['H'], opt['network']['W']
        self.msg_length = opt['network']['message_length']
        #
        self.invertible_model = IM(opt).to(device)
        self.cs_model = FSM(opt).to(device)
        self.noise_model = Noise_pool(opt, device).to(device)
        self.fusion_model= DEM(opt).to(device)
        if opt['network']['InvBlock']['downscaling']['use_down']:
            self.invDown = InvDownscaling(opt).to(device)
        self.decoder2 = NIAM(self.h, self.w, self.msg_length).to(device)
        self.nsm_model = NSM(opt, self.device)

    def encoder(self, image, msg):
        # down                                             #[128]
        cover_down = self.invDown(image)                #[64]
        # fusion
        fusion = self.fusion_model(cover_down, msg, self.invDown)          #[64]
        # inv_forward
        inv_encoded = self.invertible_model(fusion)    #[64]
        # cs
        cs = self.cs_model(inv_encoded, cover_down)
        # up to out
        watermarking_img = self.invDown(cs, rev=True).clamp(-1, 1)   #[128]

        return watermarking_img

    def noise_pool(self, watermarking_img, image, noise_choice):
        # noise
        noised_img = self.noise_model(watermarking_img, image, noise_choice)    #[128]
        
        return noised_img

    def nsm(self, noised_img):
        return torch.round(torch.mean((torch.argmax(self.nsm_model(noised_img.clone().detach().clamp(-1,1)), dim=1)).float()))

    def train_val_decoder(self, noised_img, noise_choice):
        #
        if noise_choice == 'Jpeg' or noise_choice == 'JpegTest':
            # decoder1
            msg_fake_1 = None
            img_fake = torch.zeros_like(noised_img).cuda()
            # niam  
            msg_fake_2 = self.decoder2(noised_img).clamp(-1, 1)
        else:
            # decoder1                                           
            down = self.invDown(noised_img)            #[64]
            cs_rev = self.cs_model(down, rev=True)
            inv_back = self.invertible_model(cs_rev, rev=True)   #[64]
            img_fake, msg_fake_1 = self.fusion_model(inv_back, None, self.invDown, rev=True)   #[64]
            img_fake = self.invDown(img_fake, rev=True)   #[128]
            img_fake = img_fake.clamp(-1, 1)
            # niam
            msg_fake_2 = None
        #
        msg_nsm = None
    
        return img_fake, msg_fake_1, msg_fake_2, msg_nsm

    def test_decoder(self, noised_img, pre_noise):
        if pre_noise == 1:
            # decoder1
            msg_fake_1 = None
            img_fake = torch.zeros_like(noised_img).cuda()
            # decoder2 
            msg_fake_2 = self.decoder2(noised_img).clamp(-1, 1)
        else:
            # decoder1                                           
            down = self.invDown(noised_img)            #[64]
            cs_rev = self.cs_model(down, rev=True)
            inv_back = self.invertible_model(cs_rev, rev=True)   #[64]
            img_fake, msg_fake_1 = self.fusion_model(inv_back, None, self.invDown, rev=True)   #[64]
            img_fake = self.invDown(img_fake, rev=True)   #[128]
            img_fake = img_fake.clamp(-1, 1)
            # decoder2
            msg_fake_2 = None
        #
        msg_nsm = msg_fake_1 if msg_fake_1 is not None else msg_fake_2

        return img_fake, msg_fake_1, msg_fake_2, msg_nsm




class CIN(CINBasic):
    def __init__(self, opt, device):
        super(CIN, self).__init__(opt, device)

    def forward(self, image, msg, noise_choice, is_train=True):
        #
        if is_train:
            watermarking_img = self.encoder(image, msg)
            noised_img = self.noise_pool(watermarking_img.clone(), image, noise_choice)
            img_fake, msg_fake_1, msg_fake_2, msg_nsm = self.train_val_decoder(noised_img, noise_choice)
        #
        else:
            watermarking_img = self.encoder(image, msg)
            noised_img = self.noise_pool(watermarking_img.clone(), image, noise_choice)
            pre_noise = self.nsm(noised_img)
            img_fake, msg_fake_1, msg_fake_2, msg_nsm = self.test_decoder(noised_img, pre_noise)

        return watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, msg_nsm 

