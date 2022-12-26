import torch.nn as nn
import torchvision.transforms as transforms



#
class ColorJitter(nn.Module):
    """
    
    """
    def __init__(self, opt, distortion):
        super(ColorJitter, self).__init__()
        #
        brightness   = opt['noise']['Brightness']['f']
        contrast     = opt['noise']['Contrast']['f']
        saturation   = opt['noise']['Saturation']['f']
        hue          = opt['noise']['Hue']['f']
        #
        if distortion == 'Brightness':
            self.transform = transforms.ColorJitter(brightness=brightness)
        if distortion == 'Contrast':
            self.transform = transforms.ColorJitter(contrast=contrast)
        if distortion == 'Saturation':
            self.transform = transforms.ColorJitter(saturation=saturation)
        if distortion == 'Hue':
            self.transform = transforms.ColorJitter(hue=hue)

    def forward(self, watermarked_img, cover_img=None):
        #
        watermarked_img = (watermarked_img + 1 ) / 2   # [-1, 1] -> [0, 1]
        #
        ColorJitter = self.transform(watermarked_img)
        #
        ColorJitter = (ColorJitter * 2) - 1  # [0, 1] -> [-1, 1]

        return ColorJitter


