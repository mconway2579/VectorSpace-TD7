import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer


from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform

from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.nn.nets import ResidualNet
@dataclass
class Hyperparameters:
	# Generic
	batch_size: int = 256
	buffer_size: int = 1e6
	discount: float = 0.99
	target_update_rate: int = 250
	exploration_noise: float = 0.1
	
	# TD3
	target_policy_noise: float = 0.2
	noise_clip: float = 0.5
	policy_freq: int = 2
	
	# LAP
	alpha: float = 0.4
	min_priority: float = 1
	
	# TD3+BC
	lmbda: float = 0.1
	
	# Checkpointing
	max_eps_when_checkpointing: int = 20
	steps_before_checkpointing: int = 75e4 
	reset_weight: float = 0.9
	
	# Encoder Model
	zs_dim: int = 256
	enc_hdim: int = 256
	enc_activ: Callable = F.elu
	encoder_lr: float = 3e-4
	
	# Critic Model
	critic_hdim: int = 256
	critic_activ: Callable = F.elu
	critic_lr: float = 3e-4
	
	# Actor Model
	actor_hdim: int = 256
	actor_activ: Callable = F.relu
	actor_lr: float = 3e-4


def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor, self).__init__()

		self.activ = activ

		self.l0 = nn.Linear(state_dim, hdim)
		self.l1 = nn.Linear(zs_dim + hdim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)
		

	def forward(self, state, zs):
		a = AvgL1Norm(self.l0(state))
		a = torch.cat([a, zs], 1)
		a = self.activ(self.l1(a))
		a = self.activ(self.l2(a))
		return torch.tanh(self.l3(a))


class TD7Encoder(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(TD7Encoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# state-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, zs_dim)
	

	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs

	def za(self, action):
		return action

	def a(self, za):
		return za

	def zsa(self, zs, za):
		zsa = self.activ(self.zsa1(torch.cat([zs, za], 1)))
		zsa = self.activ(self.zsa2(zsa))
		zsa = self.zsa3(zsa)
		return zsa
	

class MyEncoder(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(MyEncoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# action encoder
		self.za1 = nn.Linear(action_dim, hdim)
		self.za2 = nn.Linear(hdim, hdim)
		self.za3 = nn.Linear(hdim, zs_dim)

		# action decoder
		self.a1 = nn.Linear(zs_dim, hdim)
		self.a2 = nn.Linear(hdim, hdim)
		self.a3 = nn.Linear(hdim, action_dim)
	

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

	def a(self, za):
		a = self.activ(self.a1(za))
		a = self.activ(self.a2(a))
		a = self.a3(a)
		return a


	def zsa(self, zs, za):
		zsa = AvgL1Norm(zs + za)
		return zsa
	
class NFlowEncoder(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(NFlowEncoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# action encoder
		self.za1 = nn.Linear(action_dim, hdim)
		self.za2 = nn.Linear(hdim, hdim)
		self.za3 = nn.Linear(hdim, zs_dim)

		# action decoder
		self.a1 = nn.Linear(zs_dim, hdim)
		self.a2 = nn.Linear(hdim, hdim)
		self.a3 = nn.Linear(hdim, action_dim)

		# ---- flow for z* | (zs, za) ----
		flow_features = zs_dim            # dim of the random variable (*)
		context_dim = 2 * zs_dim          # dim of [zs, za]

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
	
	def a(self, za):
		a = self.activ(self.a1(za))
		a = self.activ(self.a2(a))
		a = self.a3(a)
		return a
	
	def zsa(self, zs, za):
		context = torch.cat([zs, za], dim=-1)
		zsa = self.flow.sample(1, context=context)
		zsa = zsa.squeeze(1)                            # [B, zs_dim]
		return zsa



class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Critic, self).__init__()

		self.activ = activ
		
		self.q01 = nn.Linear(state_dim + action_dim, hdim)
		self.q1 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(state_dim + action_dim, hdim)
		self.q4 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q5 = nn.Linear(hdim, hdim)
		self.q6 = nn.Linear(hdim, 1)


	def forward(self, state, action, zsa, zs):
		sa = torch.cat([state, action], 1)
		embeddings = torch.cat([zsa, zs], 1)

		q1 = AvgL1Norm(self.q01(sa))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		q1 = self.q3(q1)

		q2 = AvgL1Norm(self.q02(sa))
		q2 = torch.cat([q2, embeddings], 1)
		q2 = self.activ(self.q4(q2))
		q2 = self.activ(self.q5(q2))
		q2 = self.q6(q2)
		return torch.cat([q1, q2], 1)


class Agent(object):
	def __init__(self, state_dim, action_dim, max_action, args, hp=Hyperparameters()): 
		# Changing hyperparameters example: hp=Hyperparameters(batch_size=128)
		self.args = args
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		try:
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			elif torch.backends.mps.is_available():
				self.device = torch.device("mps")
			else:
				self.device = torch.device("cpu")
		except Exception:
			self.device = torch.device("cpu")
		print(f"Using device: {self.device}")
		self.hp = hp

		self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		self.critic_target = copy.deepcopy(self.critic)
		self.encoder = None
		if args.encoder == "addition":
			self.encoder = MyEncoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		elif args.encoder == "nflow":
			self.encoder = NFlowEncoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		elif args.encoder == "td7":
			self.encoder = TD7Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		else:
			raise ValueError(f"Unknown encoder type: {args.encoder}")
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		self.fixed_encoder = copy.deepcopy(self.encoder)
		self.fixed_encoder_target = copy.deepcopy(self.encoder)

		self.checkpoint_actor = copy.deepcopy(self.actor)
		self.checkpoint_encoder = copy.deepcopy(self.encoder)

		self.replay_buffer = buffer.LAP(state_dim, action_dim, self.device, hp.buffer_size, hp.batch_size, 
			max_action, normalize_actions=True, prioritized=True)

		self.max_action = max_action

		self.training_steps = 0

		# Checkpointing tracked values
		self.eps_since_update = 0
		self.timesteps_since_update = 0
		self.max_eps_before_update = 1
		self.min_return = 1e8
		self.best_min_return = -1e8

		# Value clipping tracked values
		self.max = -1e8
		self.min = 1e8
		self.max_target = 0
		self.min_target = 0


	def select_action(self, state, use_checkpoint=False, use_exploration=True):
		with torch.no_grad():
			state = torch.tensor(state.reshape(1,-1), dtype=torch.float, device=self.device)

			if use_checkpoint: 
				zs = self.checkpoint_encoder.zs(state)
				action = self.checkpoint_actor(state, zs) 
			else: 
				zs = self.fixed_encoder.zs(state)
				action = self.actor(state, zs) 
			
			if use_exploration: 
				action = action + torch.randn_like(action) * self.hp.exploration_noise				

			return action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action


	def train(self):
		self.training_steps += 1

		state, action, next_state, reward, not_done = self.replay_buffer.sample()

		#########################
		# Update Encoder
		#########################
		with torch.no_grad():
			next_zs = self.encoder.zs(next_state)

		zs = self.encoder.zs(state)
		za = self.encoder.za(action)
		if self.args.encoder == "nflow":
			context = torch.cat([zs, za], dim=-1)
			log_prob = self.encoder.flow.log_prob(next_zs, context=context)
			encoder_loss = -log_prob.mean()
		elif self.args.encoder == "addition" or self.args.encoder == "td7":
			pred_zs = self.encoder.zsa(zs, za)
			encoder_loss = F.mse_loss(pred_zs, next_zs)
		else:
			raise ValueError(f"[train] Unknown encoder type: {self.args.encoder}")
		
		decoder_loss = self.encoder.a(za)
		decoder_loss = F.mse_loss(decoder_loss, action)

		total_loss = encoder_loss + decoder_loss

		self.encoder_optimizer.zero_grad()
		total_loss.backward()
		self.encoder_optimizer.step()

		#########################
		# Update Critic
		#########################
		with torch.no_grad():
			fixed_target_zs = self.fixed_encoder_target.zs(next_state)

			noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
			next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1,1)
			fixed_target_za = self.fixed_encoder_target.za(next_action)
			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, fixed_target_za)

			Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(1,keepdim=True)[0]
			Q_target = reward + not_done * self.hp.discount * Q_target.clamp(self.min_target, self.max_target)

			self.max = max(self.max, float(Q_target.max()))
			self.min = min(self.min, float(Q_target.min()))

			fixed_zs = self.fixed_encoder.zs(state)
			fixed_za = self.fixed_encoder.za(action)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, fixed_za)

		Q = self.critic(state, action, fixed_zsa, fixed_zs)
		td_loss = (Q - Q_target).abs()
		critic_loss = LAP_huber(td_loss)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		#########################
		# Update LAP
		#########################
		priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		self.replay_buffer.update_priority(priority)

		#########################
		# Update Actor
		#########################
		if self.training_steps % self.hp.policy_freq == 0:
			actor = self.actor(state, fixed_zs)
			actor_encoding = self.encoder.za(actor)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_encoding)
			Q = self.critic(state, actor, fixed_zsa, fixed_zs)

			actor_loss = -Q.mean() 

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		#########################
		# Update Iteration
		#########################
		if self.training_steps % self.hp.target_update_rate == 0:
			self.actor_target.load_state_dict(self.actor.state_dict())
			self.critic_target.load_state_dict(self.critic.state_dict())
			self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
			self.fixed_encoder.load_state_dict(self.encoder.state_dict())
			
			self.replay_buffer.reset_max_priority()

			self.max_target = self.max
			self.min_target = self.min


	# If using checkpoints: run when each episode terminates
	def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
		self.eps_since_update += 1
		self.timesteps_since_update += ep_timesteps

		self.min_return = min(self.min_return, ep_return)

		# End evaluation of current policy early
		if self.min_return < self.best_min_return:
			self.train_and_reset()

		# Update checkpoint
		elif self.eps_since_update == self.max_eps_before_update:
			self.best_min_return = self.min_return
			self.checkpoint_actor.load_state_dict(self.actor.state_dict())
			self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
			
			self.train_and_reset()


	# Batch training
	def train_and_reset(self):
		for _ in range(self.timesteps_since_update):
			if self.training_steps == self.hp.steps_before_checkpointing:
				self.best_min_return *= self.hp.reset_weight
				self.max_eps_before_update = self.hp.max_eps_when_checkpointing
			
			self.train()

		self.eps_since_update = 0
		self.timesteps_since_update = 0
		self.min_return = 1e8