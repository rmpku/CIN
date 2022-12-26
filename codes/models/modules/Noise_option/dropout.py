import torch
import torch.nn as nn
import numpy as np

class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self, opt):
        super(Dropout, self).__init__()
        #
        self.p = opt['noise']['Dropout']['p']

    def forward(self, encoded_img, cover_image):
        
        # p% pixels replace with cover image
        mask = np.random.choice([0.0, 1.0], encoded_img.shape[2:], p=[self.p, 1 - self.p])
        mask_tensor = torch.tensor(mask, device=encoded_img.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(encoded_img)
        noised_image = encoded_img * mask_tensor + cover_image * (1-mask_tensor)
        
        return noised_image


