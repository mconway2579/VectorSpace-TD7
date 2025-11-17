
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform

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
		# self.zsa1 = nn.Linear(encoder_dim + action_dim, hdim)
		self.zsa1 = nn.Linear(2*encoder_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, encoder_dim)

		self.za1 = nn.Linear(action_dim, hdim)  # identity mapping
		self.za2 = nn.Linear(hdim, hdim)  # identity mapping
		self.za3 = nn.Linear(hdim, encoder_dim)  # identity mapping
	
	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs

	# def za(self, action):
	# 	return action
	def za(self, action):
		za = self.activ(self.za1(action))
		za = self.activ(self.za2(za))
		za = AvgL1Norm(self.za3(za))
		return za

	def zsa(self, zs, za):
		zsa = self.activ(self.zsa1(torch.cat([zs, za], 1)))
		zsa = self.activ(self.zsa2(zsa))
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
		return zs
	
	def za(self, action):
		za = self.activ(self.za1(action))
		za = self.activ(self.za2(za))
		za = AvgL1Norm(self.za3(za))
		return za

	def zsa(self, zs, za):
		zsa = AvgL1Norm(zs + za)
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
		flow_features = encoder_dim            # dim of the random variable (*)
		context_dim = 2 * encoder_dim          # dim of [zs, za]

		# simple 0/1 mask: half of dims transformed, half passthrough
		mask = torch.zeros(flow_features)
		mask[::2] = 1.0
		self.register_buffer("mask", mask)

		def make_transform_net(in_features, out_features):
			# small residual net: cheap but expressive enough
			return ResidualNet(
				in_features=in_features,
				out_features=out_features,
				hidden_features=hdim,     # keep this modest
				num_blocks=1,
				context_features=context_dim,
				activation=F.elu,
				dropout_probability=0.0,
				use_batch_norm=False,
			)

		coupling = AffineCouplingTransform(
			mask=self.mask,
			transform_net_create_fn=make_transform_net,
		)

		transform = CompositeTransform([coupling])  # exactly one transform
		base_dist = StandardNormal(shape=[flow_features])

		self.flow = Flow(transform, base_dist)

	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs
	
	def za(self, action):
		za = self.activ(self.za1(action))
		za = self.activ(self.za2(za))
		za = AvgL1Norm(self.za3(za))
		return za
	
	def zsa(self, zs, za):
		context = torch.cat([zs, za], dim=-1)
		zsa = self.flow.sample(1, context=context)
		zsa = zsa.squeeze(1)                            # [B, encoder_dim]
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
		return a
	
class IdentityDecoder(nn.Module):
	def __init__(self):
		super(IdentityDecoder, self).__init__()
	
	def decode_action(self, za):
		return za
