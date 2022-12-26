from weakref import ref
import torch.nn as nn
import numpy as np 
from . import *


#-------------------------------------------------------------
class NIAM(nn.Module):
	'''
	Reference from MBRS Net
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(NIAM, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(self.diffusion_length ** 0.5)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet(channels, channels, blocks=1)

		self.final_layer = ConvBNRelu(channels, 1)

		self.message_layer = nn.Linear(self.diffusion_length, message_length)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)

		x = self.message_layer(x)
		return x
