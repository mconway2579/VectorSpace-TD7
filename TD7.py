import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer

logger = logging.getLogger(__name__)

from state_action_codecs import TD7Encoder, AdditionEncoder, NFlowEncoder, MLPDecoder, CNNDecoder, CNNBackBone
from utils import AvgL1Norm, LAP_huber, Hyperparameters, DummyOptimizer
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms import CompositeTransform


class ProbabilisticActor(nn.Module):
	def __init__(self, backbone_state_dim, action_dim, encoder_dim=256, hdim=256, activ=nn.ReLU):
		super(ProbabilisticActor, self).__init__()

		self.l0 = nn.Linear(backbone_state_dim, hdim)
		self.context_dim = encoder_dim + hdim
		self.final_activation = nn.Tanh()

		self.action_dist = ConditionalDiagonalNormal(
			shape=[action_dim],
			context_encoder = nn.Sequential(
				nn.Linear(self.context_dim, hdim),
				activ(),
				nn.Linear(hdim, hdim),
				activ(),
				nn.Linear(hdim, 2*action_dim),
			)
		)
		self.transform = CompositeTransform([])
		self.flow = Flow(self.transform, self.action_dist)

	def forward(self, backbone_state, zs):
		s_emb = AvgL1Norm(self.l0(backbone_state))
		context = torch.cat([s_emb, zs], dim=1)
		action = self.flow.sample(1, context=context)
		action = action.squeeze(1)
		action = self.final_activation(action)
		return action

class DeterministicActor(nn.Module):
	def __init__(self, backbone_state_dim, action_dim, encoder_dim=256, hdim=256, activ=nn.ReLU):
		super(DeterministicActor, self).__init__()


		self.l0 = nn.Linear(backbone_state_dim, hdim)
		self.context_dim = encoder_dim + hdim
		self.final_activation = nn.Tanh()

		self.action_mlp = nn.Sequential(
			nn.Linear(self.context_dim, hdim),
			activ(),
			nn.Linear(hdim, hdim),
			activ(),
			nn.Linear(hdim, action_dim),
			self.final_activation
		)

	def forward(self, backbone_state, zs):
		s_emb = AvgL1Norm(self.l0(backbone_state))
		context = torch.cat([s_emb, zs], dim=1)
		action = self.action_mlp(context)
		return action
	
class Critic(nn.Module):
	def __init__(self, backbone_state_dim, action_dim, encoder_dim=256, hdim=256, activ=nn.ELU):
		super(Critic, self).__init__()

		self.activ = activ()

		self.q01 = nn.Linear(backbone_state_dim + action_dim, hdim)
		self.q1 = nn.Linear(2*encoder_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(backbone_state_dim + action_dim, hdim)
		self.q4 = nn.Linear(2*encoder_dim + hdim, hdim)
		self.q5 = nn.Linear(hdim, hdim)
		self.q6 = nn.Linear(hdim, 1)


	def forward(self, backbone_state, action, zsa, zs):
		sa = torch.cat([backbone_state, action], 1)
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

class RewardPredictor(nn.Module):
	def __init__(self, zs_dim, action_dim, hdim=256, activ=nn.ELU):
		super(RewardPredictor, self).__init__()

		self.activ = activ()

		self.r1 = nn.Linear(zs_dim + action_dim + zs_dim, hdim)
		self.r2 = nn.Linear(hdim, hdim)
		self.r3 = nn.Linear(hdim, 1)
	def forward(self, zs, action, zsa):
		x = torch.cat([zs, action, zsa], 1)
		x = self.activ(self.r1(x))
		x = self.activ(self.r2(x))
		reward = self.r3(x)
		return reward

class Agent(object):
	def __init__(self, state_dim, action_dim, low_action_arr, high_action_arr, args, writer=None, hp=None):
		assert hp is not None, "Hyperparameters (hp) must be provided to Agent"

		# Detect if state is an image based on shape
		self.state_dim = state_dim
		if isinstance(state_dim, tuple) and len(state_dim) >= 2:
			self.is_image_state = True
		else:
			self.is_image_state = False

		self.action_dim = action_dim
		self.writer = writer

		self.loss_record_freq = 100

		self.args = args
		self.device = None
		try:
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			elif torch.backends.mps.is_available():
				self.device = torch.device("mps")
			else:
				self.device = torch.device("cpu")
		except Exception:
			self.device = torch.device("cpu")
		
		logger.info(f"Using device: {self.device}")
		self.hp = hp
		self.backbone = None
		self.backbone_dim = None
		self.backbone_optimizer = None
		if self.is_image_state:
			self.backbone_dim = self.hp.encoder_dim
			self.backbone = CNNBackBone(self.backbone_dim, state_dim).to(self.device)
			self.backbone_optimizer = torch.optim.Adam(self.backbone.parameters(), lr=hp.backbone_lr)
		else:
			self.backbone_dim = state_dim[0]
			self.backbone = nn.Identity().to(self.device)
			self.backbone_optimizer = None
		self.backbone_target = copy.deepcopy(self.backbone)

		self.encoder = None
		logger.debug(f"{state_dim=}, {action_dim=}, {self.is_image_state=}")
		if args.encoder == "addition":
			self.encoder = AdditionEncoder(self.backbone_dim, action_dim, hp.encoder_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		elif args.encoder == "nflow":
			self.encoder = NFlowEncoder(self.backbone_dim, action_dim, hp.encoder_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		elif args.encoder == "td7":
			self.encoder = TD7Encoder(self.backbone_dim, action_dim, hp.encoder_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		else:
			raise ValueError(f"Unknown encoder type: {args.encoder}")

		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		self.fixed_encoder = copy.deepcopy(self.encoder)
		self.fixed_encoder_target = copy.deepcopy(self.encoder)

		self.decoder = None
		if self.is_image_state:
			self.decoder = CNNDecoder( hp.encoder_dim, state_dim ).to(self.device)
		else:
			self.decoder = MLPDecoder(state_dim, hp.encoder_dim, hp.decoder_hdim, hp.decoder_activ).to(self.device)
		self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=hp.decoder_lr)
		self.fixed_decoder = copy.deepcopy(self.decoder)
		self.fixed_decoder_target = copy.deepcopy(self.decoder)

		self.actor = None
		if args.deterministic_actor:
			self.actor = DeterministicActor(self.backbone_dim, self.action_dim, hp.encoder_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		else:
			self.actor = ProbabilisticActor(self.backbone_dim, self.action_dim, hp.encoder_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.critic = Critic(self.backbone_dim, self.action_dim, hp.encoder_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		self.critic_target = copy.deepcopy(self.critic)

		self.reward_predictor = RewardPredictor(hp.encoder_dim, self.action_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=hp.critic_lr)

		self.checkpoint_actor = copy.deepcopy(self.actor)
		self.checkpoint_encoder = copy.deepcopy(self.encoder)
		self.checkpoint_decoder = copy.deepcopy(self.decoder)
		self.checkpoint_reward_predictor = copy.deepcopy(self.reward_predictor)
		self.backbone_checkpoint = copy.deepcopy(self.backbone)

		self.replay_buffer = buffer.LAP(state_dim, self.action_dim, self.device, low_action_arr, high_action_arr, hp.buffer_size, hp.batch_size, normalize_actions=True, prioritized=True)

		self.low_action_arr = torch.tensor(low_action_arr, device=self.device, dtype=torch.float32).unsqueeze(0)
		self.high_action_arr = torch.tensor(high_action_arr, device=self.device, dtype=torch.float32).unsqueeze(0)
		# Precompute for scaling tanh [-1,1] -> [low, high]: scaled = center + tanh * scale
		self.action_scale = (self.high_action_arr - self.low_action_arr) / 2.0
		self.action_center = (self.high_action_arr + self.low_action_arr) / 2.0

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

		logger.info("#"*20)
		logger.info("Initialized TD7 Agent")
		logger.info(f"Encoder: {self.encoder}")
		logger.info(f"Decoder: {self.decoder}")
		logger.info(f"Critic: {self.critic}")
		logger.info(f"Actor: {self.actor}")
		logger.info(f"Reward Predictor: {self.reward_predictor}")
		logger.info(f"Backbone: {self.backbone}")
		logger.info(f"{self.args.hard_updates=}")
		logger.info("#"*20)

	def select_action(self, state, use_checkpoint=False, use_exploration=True):
		# print(f"[select_action] {state.shape=}, {use_checkpoint=}, {use_exploration=}")
		logger.debug(f"select_action input: {state.shape=}")
		with torch.no_grad():
			# Handle both single and batched inputs
			# Single: (C, H, W) or (state_dim,) -> add batch dim
			# Batched: (B, C, H, W) or (B, state_dim) -> keep as is
			if self.is_image_state and len(state.shape) == 3:
				# Single image state: (C, H, W) -> (1, C, H, W)
				state = state.unsqueeze(0)
			elif self.is_image_state and len(state.shape) == 4:
				# Batched image states: (B, C, H, W) -> keep as is
				pass
			elif not self.is_image_state and len(state.shape) == 1:
				# Single vector state: (state_dim,) -> (1, state_dim)
				state = state.reshape(1, -1)
			elif not self.is_image_state and len(state.shape) == 2:
				# Batched vector states: (B, state_dim) -> keep as is
				pass

			batch_size = state.shape[0]

			backbone_state = self.backbone(state)
			logger.debug(f"backbone_state: {backbone_state.shape=}")

			a = None
			if use_checkpoint:
				zs = self.checkpoint_encoder.zs(backbone_state)
				a = self.checkpoint_actor(backbone_state, zs)
			else:
				zs = self.fixed_encoder.zs(backbone_state)
				a = self.actor(backbone_state, zs)
			logger.debug(f"actor output: {a.shape=}, {zs.shape=}")

			if use_exploration:
				a = a + torch.randn_like(a) * self.hp.exploration_noise
			a = a.clamp(-1, 1) #action should already be in [-1, 1] due to tanh

			# Clamp to [-1, 1] then scale to [low, high] per dimension
			# scaled = center + tanh * scale
			actions = (self.action_center + a * self.action_scale).cpu().data.numpy()
			logger.debug(f"final action: {actions.shape=}")
			# Flatten if single action, otherwise return batch
			return actions.flatten() if batch_size == 1 else actions

	def get_encoder_loss(self, backbone_state, env_action, backbone_next_state):
		with torch.no_grad():
			next_zs = self.encoder.zs(backbone_next_state)
		
		zs = self.encoder.zs(backbone_state)
		pred_zs = self.encoder.zsa(zs, env_action)

		prediction_loss = F.mse_loss(pred_zs, next_zs)
		encoder_loss =  (self.hp.prediction_loss_lambda * prediction_loss)
		log_loss = None
		if self.args.encoder == "nflow":
			context = torch.cat([zs, env_action], dim=-1)
			log_prob = self.encoder.flow.log_prob(next_zs, context=context)
			log_loss = -log_prob.mean()
			encoder_loss = encoder_loss + (log_loss * self.hp.log_loss_weight)
		elif self.args.encoder == "addition" or self.args.encoder == "td7":
			pass
		else:
			raise ValueError(f"[train] Unknown encoder type: {self.args.encoder}")
		if self.training_steps % self.loss_record_freq == 0 and self.writer is not None:
			step = self.training_steps
			# self.writer.add_scalar("loss/encoder/reconstruction_loss", reconstruction_loss.item(), step)
			self.writer.add_scalar("loss/encoder/prediction_loss", prediction_loss.item(), step)
			if log_loss is not None:
				self.writer.add_scalar("loss/encoder/log_loss", log_loss.item(), step)
			self.writer.add_scalar("loss/encoder/total_loss", encoder_loss.item(), step)
		return encoder_loss
	
	def get_decoder_loss(self, state, next_state, fixed_zs, fixed_zsa):
		# Use pre-computed values if provided, otherwise compute them

		state_pred = self.decoder(fixed_zs)
		next_state_pred = self.decoder(fixed_zsa)
		reconstruction_loss = F.mse_loss(state_pred, state) + F.mse_loss(next_state_pred, next_state)

		decoder_loss = reconstruction_loss

		if self.training_steps % self.loss_record_freq == 0 and self.writer is not None:
			step = self.training_steps
			self.writer.add_scalar("loss/decoder/reconstruction_loss", reconstruction_loss.item(), step)
		return decoder_loss
	
	def get_actor_loss(self, backbone_state, fixed_zs):
		# Use pre-computed value if provided, otherwise compute it
		a = self.actor(backbone_state, fixed_zs)
		fixed_zsa = self.fixed_encoder.zsa(fixed_zs, a)
		Q = self.critic(backbone_state, a, fixed_zsa, fixed_zs)

		actor_loss = -Q.mean()

		if self.training_steps % self.loss_record_freq == 0 and self.writer is not None:
			# self.loss_histories["actor_loss"].append((self.training_steps, float(actor_loss.detach().cpu().item())))
			step = self.training_steps
			self.writer.add_scalar("loss/actor/total_loss", actor_loss.item(), step)
		return actor_loss
	
	def update_lap(self, td_loss):
		priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		self.replay_buffer.update_priority(priority)

	def get_critic_loss(self, backbone_state, env_action, backbone_next_state, reward, not_done, fixed_zs, fixed_zsa, fixed_target_zs):
		# Use pre-computed values if provided, otherwise compute them
		with torch.no_grad():

			next_action = self.actor_target(backbone_next_state, fixed_target_zs) #actor_out at t+1

			noise = (torch.randn_like(next_action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
			next_action = (next_action + noise).clamp(-1,1) #noised action at t+1

			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action) #zs at t+2

			Q_target = self.critic_target(backbone_next_state, next_action, fixed_target_zsa, fixed_target_zs) #q for t+1 to t+2
			# print(f"{Q_target.shape=}")
			min_Q_target = Q_target.min(1,keepdim=True)[0]
			# print(f"{min_Q_target.shape=}")
			Q_target = reward + not_done * self.hp.discount * min_Q_target.clamp(self.min_target, self.max_target) #q for t to t+1
			# print(f"{Q_target.shape=}")
			self.max = max(self.max, float(Q_target.max()))
			self.min = min(self.min, float(Q_target.min()))

		Q = self.critic(backbone_state, env_action, fixed_zsa, fixed_zs)
		# print(f"{Q.shape=}")
		td_loss = (Q - Q_target).abs()
		# print(f"{td_loss.shape=}")
		critic_loss = LAP_huber(td_loss)
		self.update_lap(td_loss)

		# print(f"{critic_loss.shape=}")

		if self.training_steps % self.loss_record_freq == 0 and self.writer is not None:
			# self.loss_histories["critic_td_loss"].append((self.training_steps, float(td_loss.mean().detach().cpu().item())))
			# self.loss_histories["critic_loss"].append((self.training_steps, float(critic_loss.detach().cpu().item())))
			step = self.training_steps
			self.writer.add_scalar("loss/critic/td_loss", td_loss.mean().item(), step)
			self.writer.add_scalar("loss/critic/total_loss", critic_loss.item(), step)
		return critic_loss
	
	def get_reward_predictor_loss(self, env_action, reward, fixed_zs, fixed_zsa):
		predicted_reward = self.reward_predictor(fixed_zs, env_action, fixed_zsa)
		reward_loss = F.mse_loss(predicted_reward, reward)

		if self.training_steps % self.loss_record_freq == 0 and self.writer is not None:
			step = self.training_steps
			self.writer.add_scalar("loss/reward_predictor/reward_loss", reward_loss.item(), step)
		return reward_loss

	def soft_update_targets(self):
		tau = 1/self.hp.target_update_rate
		target_source = [
			(self.backbone_target, self.backbone),
			(self.actor_target, self.actor),
			(self.critic_target, self.critic),
			(self.fixed_encoder_target, self.fixed_encoder),
			(self.fixed_encoder, self.encoder),
			(self.fixed_decoder_target, self.fixed_decoder),
			(self.fixed_decoder, self.decoder),
		]
		for target, source in target_source:
			for target_param, source_param in zip(target.parameters(), source.parameters()):
				target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
	def hard_update_targets(self):
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
		self.fixed_encoder.load_state_dict(self.encoder.state_dict())
		self.fixed_decoder_target.load_state_dict(self.fixed_decoder.state_dict())
		self.fixed_decoder.load_state_dict(self.decoder.state_dict())
		self.backbone_target.load_state_dict(self.backbone.state_dict())
	def train(self):
		self.training_steps += 1

		state, env_action, next_state, reward, not_done = self.replay_buffer.sample()

		#########################
		# Update Encoder + Backbone
		#########################
		if self.backbone_optimizer is not None:
			self.backbone_optimizer.zero_grad(set_to_none=False)
		self.encoder_optimizer.zero_grad(set_to_none=False)

		backbone_state = self.backbone(state)
		backbone_next_state = self.backbone(next_state)
		encoder_loss = self.get_encoder_loss(backbone_state, env_action, backbone_next_state)

		# Pre-compute shared forward passes (must be after backbone_state for alignment)
		with torch.no_grad():
			# For current state: use main backbone output (aligned with critic input)
			fixed_zs = self.fixed_encoder.zs(backbone_state)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, env_action)
			# For target computation: use target backbone (more stable for Q-targets)
			fixed_backbone_next_state = self.backbone_target(next_state)
			fixed_target_zs = self.fixed_encoder_target.zs(fixed_backbone_next_state)

		encoder_loss.backward(retain_graph=self.is_image_state)

		if self.training_steps % self.loss_record_freq == 0:
			self._log_gradients(self.encoder, "encoder")
			self._log_gradients(self.backbone, "backbone")

		self.encoder_optimizer.step()

		#########################
		# Update Critic
		#########################
		self.critic_optimizer.zero_grad(set_to_none=False)

		critic_loss = self.get_critic_loss(backbone_state, env_action, backbone_next_state, reward, not_done, fixed_zs, fixed_zsa, fixed_target_zs)

		critic_loss.backward(retain_graph=self.is_image_state)

		if self.training_steps % self.loss_record_freq == 0:
			self._log_gradients(self.critic, "critic")

		self.critic_optimizer.step()

		#########################
		# Update Actor
		#########################
		if self.training_steps % self.hp.policy_freq == 0:
			self.actor_optimizer.zero_grad(set_to_none=False)

			actor_loss = self.get_actor_loss(backbone_state, fixed_zs)

			actor_loss.backward()

			if self.training_steps % self.loss_record_freq == 0:
				self._log_gradients(self.actor, "actor")

			self.actor_optimizer.step()

		if self.backbone_optimizer is not None:
			self.backbone_optimizer.step()

		#########################
		# Update Decoder & reward_predictor
		#########################
		self.reward_predictor_optimizer.zero_grad(set_to_none=False)
		self.decoder_optimizer.zero_grad(set_to_none=False)

		decoder_loss = self.get_decoder_loss(state, next_state, fixed_zs, fixed_zsa)
		reward_predictor_loss = self.get_reward_predictor_loss(env_action, reward, fixed_zs, fixed_zsa)
		acc_loss = decoder_loss + reward_predictor_loss

		acc_loss.backward()

		if self.training_steps % self.loss_record_freq == 0:
			self._log_gradients(self.reward_predictor, "reward_predictor")
			self._log_gradients(self.decoder, "decoder")

		self.reward_predictor_optimizer.step()
		self.decoder_optimizer.step()


		#########################
		# Update Iteration
		#########################
		if not self.args.hard_updates:
			self.soft_update_targets()
		if self.training_steps % self.hp.target_update_rate == 0:
			self.replay_buffer.reset_max_priority()
			self.max_target = self.max
			self.min_target = self.min
			if self.args.hard_updates:
				self.hard_update_targets()
				

			

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
			self.backbone_checkpoint.load_state_dict(self.backbone.state_dict())
			self.checkpoint_actor.load_state_dict(self.actor.state_dict())
			self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
			self.checkpoint_decoder.load_state_dict(self.fixed_decoder.state_dict())
			self.checkpoint_reward_predictor.load_state_dict(self.reward_predictor.state_dict())

			
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

	def save(self, filepath: str):
		"""
		Save all networks, optimizers, checkpoints, and hyperparameters.
		"""
		logger.info(f"Saving checkpoint to {filepath}")
		checkpoint = {
			# Reconstruction info
			"state_dim": self.state_dim,
			"action_dim": self.action_dim,
			"low_action_arr": self.low_action_arr.squeeze(0).cpu().numpy(),
			"high_action_arr": self.high_action_arr.squeeze(0).cpu().numpy(),
			"args": self.args,
			"is_image_state": self.is_image_state,

			# Hyperparameters (stored as a plain dict for robustness)
			"hp": dict(vars(self.hp)),

			# Networks
			"backbone": self.backbone.state_dict(),
			"backbone_target": self.backbone_target.state_dict(),

			"critic": self.critic.state_dict(),
			"critic_target": self.critic_target.state_dict(),

			"encoder": self.encoder.state_dict(),
			"fixed_encoder": self.fixed_encoder.state_dict(),
			"fixed_encoder_target": self.fixed_encoder_target.state_dict(),

			"decoder": self.decoder.state_dict(),
			"fixed_decoder": self.fixed_decoder.state_dict(),
			"fixed_decoder_target": self.fixed_decoder_target.state_dict(),

			"reward_predictor": self.reward_predictor.state_dict(),


			"actor": self.actor.state_dict(),
			"actor_target": self.actor_target.state_dict(),

			"backbone_checkpoint": self.backbone_checkpoint.state_dict(),
			"checkpoint_actor": self.checkpoint_actor.state_dict(),
			"checkpoint_encoder": self.checkpoint_encoder.state_dict(),
			"checkpoint_decoder": self.checkpoint_decoder.state_dict(),
			"checkpoint_reward_predictor": self.checkpoint_reward_predictor.state_dict(),

			# Training state (optional but useful)
			"training_steps": self.training_steps,
			"eps_since_update": self.eps_since_update,
			"timesteps_since_update": self.timesteps_since_update,
			"max_eps_before_update": self.max_eps_before_update,
			"min_return": self.min_return,
			"best_min_return": self.best_min_return,
			"max": self.max,
			"min": self.min,
			"max_target": self.max_target,
			"min_target": self.min_target,
		}

		torch.save(checkpoint, filepath)
		logger.info(f"Checkpoint saved successfully")

	@classmethod
	def load(cls, filepath: str, args_override=None, hp_override: Hyperparameters | None = None):
		"""
		Load an Agent from a checkpoint.
		Re-instantiates the networks/optimizers/targets and restores their state.
		
		- args_override: if provided, overrides args stored in the checkpoint.
		- hp_override: if provided, overrides hyperparameters stored in the checkpoint.
		"""
		checkpoint = torch.load(filepath, map_location="cpu",  weights_only=False)

		# Restore / construct hyperparameters
		hp_dict = checkpoint["hp"]
		if hp_override is not None:
			hp = hp_override
		else:
			hp = Hyperparameters(**hp_dict)

		# Restore / construct args
		if args_override is not None:
			args = args_override
		else:
			args = checkpoint["args"]

		state_dim = checkpoint["state_dim"]
		action_dim = checkpoint["action_dim"]
		low_action_arr = checkpoint["low_action_arr"]
		high_action_arr = checkpoint["high_action_arr"]

		# Instantiate a fresh Agent (this builds all modules & optimizers)
		agent = cls(
			state_dim=state_dim,
			action_dim=action_dim,
			low_action_arr=low_action_arr,
			high_action_arr=high_action_arr,
			args=args,
			hp=hp,
		)

		# Networks
		agent.backbone.load_state_dict(checkpoint["backbone"])
		agent.backbone_target.load_state_dict(checkpoint["backbone_target"])

		agent.critic.load_state_dict(checkpoint["critic"])
		agent.critic_target.load_state_dict(checkpoint["critic_target"])

		agent.encoder.load_state_dict(checkpoint["encoder"])
		agent.fixed_encoder.load_state_dict(checkpoint["fixed_encoder"])
		agent.fixed_encoder_target.load_state_dict(checkpoint["fixed_encoder_target"])

		agent.decoder.load_state_dict(checkpoint["decoder"])
		agent.fixed_decoder.load_state_dict(checkpoint["fixed_decoder"])
		agent.fixed_decoder_target.load_state_dict(checkpoint["fixed_decoder_target"])

		agent.actor.load_state_dict(checkpoint["actor"])
		agent.actor_target.load_state_dict(checkpoint["actor_target"])

		agent.reward_predictor.load_state_dict(checkpoint["reward_predictor"])


		agent.backbone_checkpoint.load_state_dict(checkpoint["backbone_checkpoint"])
		agent.checkpoint_actor.load_state_dict(checkpoint["checkpoint_actor"])
		agent.checkpoint_encoder.load_state_dict(checkpoint["checkpoint_encoder"])
		agent.checkpoint_decoder.load_state_dict(checkpoint["checkpoint_decoder"])
		agent.checkpoint_reward_predictor.load_state_dict(checkpoint["checkpoint_reward_predictor"])

		# Training state (if present)
		agent.training_steps = checkpoint.get("training_steps", 0)
		agent.eps_since_update = checkpoint.get("eps_since_update", 0)
		agent.timesteps_since_update = checkpoint.get("timesteps_since_update", 0)
		agent.max_eps_before_update = checkpoint.get("max_eps_before_update", agent.max_eps_before_update)
		agent.min_return = checkpoint.get("min_return", agent.min_return)
		agent.best_min_return = checkpoint.get("best_min_return", agent.best_min_return)
		agent.max = checkpoint.get("max", agent.max)
		agent.min = checkpoint.get("min", agent.min)
		agent.max_target = checkpoint.get("max_target", agent.max_target)
		agent.min_target = checkpoint.get("min_target", agent.min_target)

		logger.info(f"Loaded checkpoint from {filepath}")
		return agent

	def set_device(self, device):
		"""
		Move all models to the specified device.

		Args:
			device: Device string ('cpu', 'cuda', 'mps') or torch.device object
		"""
		if isinstance(device, str):
			device = torch.device(device)

		self.device = device
		logger.info(f"Moving all models to device: {device}")

		# Move all models to the new device
		self.backbone.to(device)
		self.backbone_target.to(device)


		self.encoder.to(device)
		self.fixed_encoder.to(device)
		self.fixed_encoder_target.to(device)

		self.decoder.to(device)
		self.fixed_decoder.to(device)
		self.fixed_decoder_target.to(device)

		self.actor.to(device)
		self.actor_target.to(device)

		self.critic.to(device)
		self.critic_target.to(device)

		self.reward_predictor.to(device)
		

		self.checkpoint_actor.to(device)
		self.checkpoint_encoder.to(device)
		self.checkpoint_decoder.to(device)
		self.checkpoint_reward_predictor.to(device)
		self.backbone_checkpoint.to(device)

		# Update replay buffer device
		self.replay_buffer.device = device

		# Move action scaling tensors to new device
		self.low_action_arr = self.low_action_arr.to(device)
		self.high_action_arr = self.high_action_arr.to(device)
		self.action_scale = self.action_scale.to(device)
		self.action_center = self.action_center.to(device)

		logger.info(f"Successfully moved all models to {device}")

	def _log_gradients(self, module: nn.Module, module_name: str):
		"""Log global L2 norm and max |grad| for a module."""
		if self.writer is None:
			return

		# Skip gradient logging if disabled (avoids CPU-GPU sync overhead)
		if not self.args.log_gradients:
			return

		# Collect all gradients first (stay on GPU)
		grads = [p.grad.detach() for p in module.parameters() if p.grad is not None]
		if not grads:
			return

		# Batch GPU operations to minimize syncs
		norms = torch.stack([g.norm(2) for g in grads])
		maxes = torch.stack([g.abs().max() for g in grads])

		# Single sync at the end (2 total instead of 2*num_params)
		global_l2 = norms.norm(2).item()
		max_abs = maxes.max().item()

		step = self.training_steps
		self.writer.add_scalar(f"grad/{module_name}/global_l2", global_l2, step)
		self.writer.add_scalar(f"grad/{module_name}/max_abs", max_abs, step)