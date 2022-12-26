import torch.nn as nn
import numpy as np 
import torch.nn.functional as F


# ------------------- 1. Convolutional Neural Network ------------------- #
class ConvBNRelu(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride=1):
		super(ConvBNRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class BottleneckBlock(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(BottleneckBlock, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					  stride=drop_rate, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x


class SENet(nn.Module):
	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
		super(SENet, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(out_channels, out_channels, r, drop_rate)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class SENet_decoder(nn.Module):
	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
		super(SENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate)
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


# ------------------- 2. NIAM ------------------- #
class NIAM(nn.Module):
	'''
	Reference from "https://github.com/jzyustc/MBRS"
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
