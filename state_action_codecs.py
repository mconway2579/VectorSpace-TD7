
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
	def __init__(self, state_dim, action_dim, encoder_dim=256, hdim=256, activ=F.elu):
		super(TD7Encoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, encoder_dim)
		
		# state-action encoder
		self.zsa1 = nn.Linear(encoder_dim + action_dim, hdim)
		# self.zsa1 = nn.Linear(2*encoder_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, encoder_dim)

		self.za1 = nn.Linear(action_dim, encoder_dim)
	
	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		# zs = self.zs3(zs)
		return zs

	def za(self, action):
		# return self.za1(action)
		return action
		

	def zsa(self, zs, za):
		zsa = self.activ(self.zsa1(torch.cat([zs, za], 1)))
		zsa = self.activ(self.zsa2(zsa))
		# zsa = AvgL1Norm(self.zsa3(zsa))
		zsa = self.zsa3(zsa)
		return zsa
	
	
class AdditionEncoder(nn.Module):
	def __init__(self, state_dim, action_dim, encoder_dim=256, hdim=256, activ=F.elu):
		super(AdditionEncoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, encoder_dim)
		
		# action encoder
		self.za1 = nn.Linear(action_dim, hdim)
		self.za2 = nn.Linear(hdim, hdim)
		self.za3 = nn.Linear(hdim, encoder_dim)
	

	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		# zs = self.zs3(zs)
		return zs
	
	def za(self, action):
		za = self.activ(self.za1(action))
		za = self.activ(self.za2(za))
		# za = AvgL1Norm(self.za3(za))
		za = torch.tanh(self.za3(za))
		return za

	def zsa(self, zs, za):
		zsa = AvgL1Norm(zs + za)
		# zsa = zs + za
		return zsa
	
class NFlowEncoder(nn.Module):
	def __init__(self, state_dim, action_dim, encoder_dim=256, hdim=256, activ=F.elu):
		super(NFlowEncoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, encoder_dim)
		
		# action encoder
		self.za1 = nn.Linear(action_dim, hdim)
		self.za2 = nn.Linear(hdim, hdim)
		self.za3 = nn.Linear(hdim, encoder_dim)

        # ---- flow for z* | (zs, za) ----
		context_dim = 2 * encoder_dim

		# register the context encoder as a submodule
		self.context_encoder = nn.Linear(context_dim, encoder_dim * 2)

		self.base_dist = ConditionalDiagonalNormal(
			shape=[encoder_dim],
			context_encoder=self.context_encoder,
		)

		# num_blocks = 3
		# blocks = []
		# for _ in range(num_blocks):
		# 	blocks.append(LULinear(features=encoder_dim))
		# 	blocks.append(LeakyReLU())
		blocks = []

		self.transform = CompositeTransform(blocks)
		self.flow = Flow(self.transform, self.base_dist)


	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		# zs = self.zs3(zs)
		return zs
	
	def za(self, action):
		za = self.activ(self.za1(action))
		za = self.activ(self.za2(za))
		# za = AvgL1Norm(self.za3(za))
		za = torch.tanh(self.za3(za))
		return za
	
	def zsa(self, zs, za):
		context = torch.cat([zs, za], dim=-1)
		zsa = self.flow.sample(1, context=context)
		zsa = zsa.squeeze(1)                            # [B, encoder_dim]
		zsa = AvgL1Norm(zsa)
		return zsa

class MLPDecoder(nn.Module):
	def __init__(self, action_dim, encoder_dim=256, hdim=256, activ=F.elu):
		super(MLPDecoder, self).__init__()

		self.activ = activ

		# action decoder
		self.a1 = nn.Linear(encoder_dim, hdim)
		self.a2 = nn.Linear(hdim, hdim)
		self.a3 = nn.Linear(hdim, action_dim)

	def decode_action(self, za):
		a = self.activ(self.a1(za))
		a = self.activ(self.a2(a))
		a = self.a3(a)
		a = torch.tanh(a)
		return a
	
class IdentityDecoder(nn.Module):
	def __init__(self):
		super(IdentityDecoder, self).__init__()
	
	def decode_action(self, za):
		return za
