import torch.nn as nn
import torch.nn.functional as F




class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, opt):
        super(Resize, self).__init__()
        resize_ratio_down = opt['noise']['Resize']['p']
        self.h = opt['network']['H']
        self.w = opt['network']['W']
        self.scaled_h = int(resize_ratio_down * self.h)
        self.scaled_w = int(resize_ratio_down * self.w)
        self.interpolation_method = opt['noise']['Resize']['interpolation_method']

    def forward(self, wm_imgs, cover_img=None):
        
        #
        noised_down = F.interpolate(
                                    wm_imgs,
                                    size=(self.scaled_h, self.scaled_w),
                                    mode=self.interpolation_method
                                    )
        noised_up = F.interpolate(
                                    noised_down,
                                    size=(self.h, self.w),
                                    mode=self.interpolation_method
                                    )

        return noised_up


