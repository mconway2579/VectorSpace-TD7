
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.nonlinearities import LeakyReLU


from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.nn.nets import ResidualNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AvgL1Norm

class TD7Encoder(nn.Module):
	def __init__(self, s_backbone_dim, action_dim, encoder_dim=256, hdim=256, activ=nn.ELU):
		super(TD7Encoder, self).__init__()

		self.activ = activ()

		# state encoder
		self.zs_mlp = nn.Sequential(
			nn.Linear(s_backbone_dim, hdim),
			activ(),
			nn.Linear(hdim, hdim),
			activ(),
			nn.Linear(hdim, encoder_dim)
		)

		# state-action encoder (encoder_dim + action_dim input)
		self.zsa_mlp = nn.Sequential(
			nn.Linear(encoder_dim + action_dim, hdim),
			activ(),
			nn.Linear(hdim, hdim),
			activ(),
			nn.Linear(hdim, encoder_dim)
		)

	def zs(self, backbone_state):
		zs = self.zs_mlp(backbone_state)
		zs = AvgL1Norm(zs)
		return zs

	def za(self, action):
		return action

	def zsa(self, zs, a):
		context = torch.cat([zs, a], 1)
		zsa = self.zsa_mlp(context)
		return zsa
	
	
class AdditionEncoder(nn.Module):
	def __init__(self, s_backbone_dim, action_dim, encoder_dim=256, hdim=256, activ=nn.ELU):
		super(AdditionEncoder, self).__init__()

		self.activ = activ()

		# state encoder
		self.zs_mlp = nn.Sequential(
			nn.Linear(s_backbone_dim, hdim),
			activ(),
			nn.Linear(hdim, hdim),
			activ(),
			nn.Linear(hdim, encoder_dim)
		)

		# action encoder
		self.za_mlp = nn.Sequential(
			nn.Linear(action_dim, hdim),
			activ(),
			nn.Linear(hdim, hdim),
			activ(),
			nn.Linear(hdim, encoder_dim),
			nn.Tanh()
		)

	def zs(self, backbone_state):
		x = self.zs_mlp(backbone_state)
		zs = AvgL1Norm(x)
		return zs

	def za(self, action):
		za = self.za_mlp(action)
		return za

	def zsa(self, zs, a):
		za = self.za(a)
		zsa = AvgL1Norm(zs + za)
		return zsa
	
class NFlowEncoder(nn.Module):
	def __init__(self, s_backbone_dim, action_dim, encoder_dim=256, hdim=256, activ=nn.ELU):
		super(NFlowEncoder, self).__init__()

		self.activ = activ()

		# state encoder
		self.zs_mlp = nn.Sequential(
			nn.Linear(s_backbone_dim, hdim),
			activ(),
			nn.Linear(hdim, hdim),
			activ(),
			nn.Linear(hdim, encoder_dim)
		)

        # ---- flow for z* | (zs, za) ----
		context_dim = encoder_dim + action_dim

		# register the context encoder as a submodule
		self.context_encoder = nn.Sequential(
			nn.Linear(context_dim, hdim),
			self.activ,
			nn.Linear(hdim, hdim),
			self.activ,
			nn.Linear(hdim, encoder_dim * 2)
		)
		self.base_dist = ConditionalDiagonalNormal(
			shape=[encoder_dim],
			context_encoder=self.context_encoder,
		)

		blocks = []
		self.transform = CompositeTransform(blocks)
		self.flow = Flow(self.transform, self.base_dist)


	def zs(self, backbone_state):
		zs = self.zs_mlp(backbone_state)
		zs = AvgL1Norm(zs)
		return zs

	def za(self, action):
		# za = self.activ(self.za1(action))
		# za = self.activ(self.za2(za))
		# za = AvgL1Norm(self.za3(za))
		# za = torch.tanh(self.za3(za))
		return action

	def zsa(self, zs, a):
		context = torch.cat([zs, a], dim=-1)
		zsa = self.flow.sample(1, context=context)
		zsa = zsa.squeeze(1)                            # [B, encoder_dim]
		# zsa = AvgL1Norm(zsa)
		return zsa

class MLPDecoder(nn.Module):
	def __init__(self, state_dim, encoder_dim=256, hdim=256, activ=F.elu):
		super(MLPDecoder, self).__init__()

		self.activ = activ()

		self.s1 = nn.Linear(encoder_dim, hdim)
		self.s2 = nn.Linear(hdim, hdim)
		self.s3 = nn.Linear(hdim, state_dim[0])

	def forward(self, zs):
		s = self.activ(self.s1(zs))
		s = self.activ(self.s2(s))
		s = self.s3(s)
		return s


class CNNBackBone(nn.Module):
	def __init__(self, encoder_dim=256, image_shape=(3, 84, 84)):
		super(CNNBackBone, self).__init__()

		self.image_shape = image_shape
		channels, height, width = image_shape

		# SAC-style convolutional layers (4 layers, 3x3 kernels, 32 channels)
		self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

		# Calculate the flattened size after convolutions
		def conv_output_size(size, kernel_size, stride, padding):
			return (size + 2 * padding - kernel_size) // stride + 1

		h = conv_output_size(height, 3, 2, 1)  # First layer: stride 2
		h = conv_output_size(h, 3, 1, 1)       # Remaining layers: stride 1
		h = conv_output_size(h, 3, 1, 1)
		h = conv_output_size(h, 3, 1, 1)

		w = conv_output_size(width, 3, 2, 1)
		w = conv_output_size(w, 3, 1, 1)
		w = conv_output_size(w, 3, 1, 1)
		w = conv_output_size(w, 3, 1, 1)

		self.flat_size = 32 * h * w

		# FC layer to encoder dimension
		self.fc1 = nn.Linear(self.flat_size, encoder_dim)
		self.layer_norm = nn.LayerNorm(encoder_dim)

	def forward(self, x):
		# x shape: [batch, channels, height, width]
		x = F.relu(self.conv1(x), inplace=True)
		x = F.relu(self.conv2(x), inplace=True)
		x = F.relu(self.conv3(x), inplace=True)
		x = F.relu(self.conv4(x), inplace=True)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.layer_norm(x)
		x = torch.tanh(x)
		return x


class CNNDecoder(nn.Module):
	def __init__(self, encoder_dim=256, image_shape=(3, 84, 84)):
		super(CNNDecoder, self).__init__()

		self.image_shape = image_shape
		channels, height, width = image_shape

		# Calculate the size after convolutions (reverse of encoder)
		def conv_output_size(size, kernel_size, stride, padding):
			return (size + 2 * padding - kernel_size) // stride + 1

		# Forward pass through encoder convolutions (matching new 4-layer encoder)
		h1 = conv_output_size(height, 3, 2, 1)  # First layer: stride 2
		h2 = conv_output_size(h1, 3, 1, 1)      # Remaining layers: stride 1
		h3 = conv_output_size(h2, 3, 1, 1)
		h4 = conv_output_size(h3, 3, 1, 1)

		w1 = conv_output_size(width, 3, 2, 1)
		w2 = conv_output_size(w1, 3, 1, 1)
		w3 = conv_output_size(w2, 3, 1, 1)
		w4 = conv_output_size(w3, 3, 1, 1)

		self.flat_size = 32 * h4 * w4
		self.h = h4
		self.w = w4

		# Fully connected layer from encoder dimension
		self.fc = nn.Linear(encoder_dim, self.flat_size)

		# Calculate output_padding needed for exact inversion
		# For ConvTranspose2d: output_size = (input_size - 1) * stride - 2*padding + kernel_size + output_padding
		def calc_output_padding(target_size, input_size, kernel_size, stride, padding):
			return target_size - ((input_size - 1) * stride - 2 * padding + kernel_size)

		out_pad1_h = calc_output_padding(h3, h4, 3, 1, 1)
		out_pad1_w = calc_output_padding(w3, w4, 3, 1, 1)
		out_pad2_h = calc_output_padding(h2, h3, 3, 1, 1)
		out_pad2_w = calc_output_padding(w2, w3, 3, 1, 1)
		out_pad3_h = calc_output_padding(h1, h2, 3, 1, 1)
		out_pad3_w = calc_output_padding(w1, w2, 3, 1, 1)
		out_pad4_h = calc_output_padding(height, h1, 3, 2, 1)
		out_pad4_w = calc_output_padding(width, w1, 3, 2, 1)

		# Transposed convolutional layers (reverse of encoder, 4 layers with 32 channels)
		self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=(out_pad1_h, out_pad1_w))
		self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=(out_pad2_h, out_pad2_w))
		self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=(out_pad3_h, out_pad3_w))
		self.deconv4 = nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=(out_pad4_h, out_pad4_w))

	def forward(self, x):
		# x shape: [batch, encoder_dim]
		x = self.fc(x)
		x = x.view(x.size(0), 32, self.h, self.w)  # Unflatten
		x = F.relu(self.deconv1(x), inplace=True)
		x = F.relu(self.deconv2(x), inplace=True)
		x = F.relu(self.deconv3(x), inplace=True)
		x = F.sigmoid(self.deconv4(x))  # Sigmoid on final layer for [0,1] output
		return x



