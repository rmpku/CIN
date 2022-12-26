import torch
import torch.nn as nn
from models.modules.Noise_option.cropout import get_random_rectangle_inside
import numpy as np




class Crop(nn.Module):
    """
    Keep the value of only one random rectangular area and set the value of other areas to 0
    """
    def __init__(self, opt):
        super(Crop, self).__init__()
        #
        ratio = opt['noise']['Crop']['p']  # 9% of total pixel
        #
        self.height_ratio_range = np.sqrt(ratio)
        self.width_ratio_range = np.sqrt(ratio)

    def forward(self, encoded_img, cover_img=None):

        cropout_mask = torch.zeros_like(encoded_img)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=encoded_img,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        noised_img = encoded_img * cropout_mask     # + encoded_img * (1-cropout_mask)

        return  noised_img

