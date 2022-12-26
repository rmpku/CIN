import torch.nn as nn
import numpy as np


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the position is random and the size is controlled by height_ratio_range and width_ratio_range.

    :param image: The image we want to crop
    :param height_ratio_range: The range of crop height ratio
    :param width_ratio_range:  The range of crop width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(height_ratio_range * image_height)
    remaining_width = int(width_ratio_range * image_width)

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Cropout(nn.Module):
    """    
    Randomly crops the image from top/bottom and left/right. 
    The amount to crop is controlled by parameters heigth_ratio_range and width_ratio_range
    """
    def __init__(self, opt):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Cropout, self).__init__()
        #
        ratio = opt['noise']['Cropout']['p']  # 0.5 means 50% retain of total watermarked pixel
        #
        self.height_ratio_range = int(np.sqrt(ratio) * 100) / 100  
        self.width_ratio_range = int(np.sqrt(ratio) * 100) / 100


    def forward(self, encoded_img, cover_img):
        
        #
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(encoded_img, self.height_ratio_range, self.width_ratio_range)
        #
        out = cover_img.clone()
        out[:,:, h_start:h_end, w_start: w_end] = encoded_img[:,:, h_start:h_end, w_start: w_end]        
        
        return out
    
    
    