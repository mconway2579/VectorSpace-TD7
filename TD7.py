import copy


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer

from state_action_codecs import TD7Encoder, AdditionEncoder, NFlowEncoder, IdentityDecoder, MLPDecoder
from utils import AvgL1Norm, LAP_huber, DummyOptimizer, Hyperparameters
class Actor(nn.Module):
	#essentially mlp that maps a vector in R state_dim+encoder_dim to R_action_dim
	def __init__(self, state_dim, action_dim, encoder_dim=256, hdim=256, activ=F.relu):
		super(Actor, self).__init__()

		self.activ = activ

		self.l0 = nn.Linear(state_dim, hdim)
		self.l1 = nn.Linear(encoder_dim + hdim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)

	def forward(self, state, zs):
		a = AvgL1Norm(self.l0(state))
		a = torch.cat([a, zs], 1)
		a = self.activ(self.l1(a))
		a = self.activ(self.l2(a))
		actor_out = torch.tanh(self.l3(a))
		# actor_out = self.l3(a)
		return actor_out
	
class Critic(nn.Module):
	#essentially mlp that maps a vector in R state_dim+action_dim+2*encoder_dim to R^2
	def __init__(self, state_dim, action_dim, encoder_dim=256, hdim=256, activ=F.elu):
		super(Critic, self).__init__()

		self.activ = activ
		
		self.q01 = nn.Linear(state_dim + action_dim, hdim)
		self.q1 = nn.Linear(2*encoder_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(state_dim + action_dim, hdim)
		self.q4 = nn.Linear(2*encoder_dim + hdim, hdim)
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
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.loss_record_freq = 100

		self.args = args
		self.loss_histories = {
			"encoder_reconstruction_loss": [],
			"encoder_log_loss": [],
			"encoder_loss": [],

			"decoder_reconstruction_loss": [],
			"decoder_bc_loss": [],
			"decoder_q_loss": [],
			"decoder_loss": [],

			"critic_td_loss": [],
			"critic_loss": [],

			"actor_loss": [],
		}
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
		print(f"Using device: {self.device}")
		self.hp = hp

		self.critic = Critic(state_dim, action_dim, hp.encoder_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		self.critic_target = copy.deepcopy(self.critic)

		self.encoder = None
		if args.encoder == "addition":
			self.encoder = AdditionEncoder(state_dim, action_dim, hp.encoder_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		elif args.encoder == "nflow":
			self.encoder = NFlowEncoder(state_dim, action_dim, hp.encoder_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		elif args.encoder == "td7":
			self.encoder = TD7Encoder(state_dim, action_dim, hp.encoder_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		else:
			raise ValueError(f"Unknown encoder type: {args.encoder}")
		
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		self.fixed_encoder = copy.deepcopy(self.encoder)
		self.fixed_encoder_target = copy.deepcopy(self.encoder)
		
		self.action_space_dim = None
		# self.action_space_dim = action_dim
		self.decoder = None
		self.decoder_optimizer = None
		if args.action_space == "environment":
			self.action_space_dim = action_dim
			self.decoder = IdentityDecoder().to(self.device)
			self.decoder_optimizer = DummyOptimizer()
		elif args.action_space == "embedding":
			self.action_space_dim = hp.encoder_dim if args.encoder != "td7" else action_dim
			self.decoder = MLPDecoder(action_dim, self.action_space_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
			self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=hp.encoder_lr)

		self.fixed_decoder = copy.deepcopy(self.decoder)
		self.fixed_decoder_target = copy.deepcopy(self.decoder)

		self.actor = Actor(state_dim, self.action_space_dim, hp.encoder_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		self.actor_target = copy.deepcopy(self.actor)

		# self.critic = Critic(state_dim, self.action_space_dim, hp.encoder_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		# self.critic_target = copy.deepcopy(self.critic)

		self.checkpoint_actor = copy.deepcopy(self.actor)
		self.checkpoint_encoder = copy.deepcopy(self.encoder)
		self.checkpoint_decoder = copy.deepcopy(self.decoder)

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

		print("#"*20)
		print("Initialized TD7 Agent")
		print(f"Encoder: {self.encoder}")
		print(f"Decoder: {self.decoder}")
		print(f"Critic: {self.critic}")
		print(f"Actor: {self.actor}")
		print("#"*20)

	def select_action(self, state, use_checkpoint=False, use_exploration=True):
		with torch.no_grad():
			state = torch.tensor(state.reshape(1,-1), dtype=torch.float, device=self.device)

			actor_out = None
			if use_checkpoint: 
				zs = self.checkpoint_encoder.zs(state)
				actor_out = self.checkpoint_actor(state, zs) 
			else: 
				zs = self.fixed_encoder.zs(state)
				actor_out = self.actor(state, zs) 
			
			if use_exploration: 
				actor_out = actor_out + torch.randn_like(actor_out) * self.hp.exploration_noise

			env_action = actor_out
			if use_checkpoint: 
				env_action = self.checkpoint_decoder.decode_action(actor_out)
			else:
				env_action = self.fixed_decoder.decode_action(actor_out)

			return env_action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action

	def get_encoder_loss(self, state, env_action, next_state):
		with torch.no_grad():
			next_zs = self.encoder.zs(next_state)
		
		zs = self.encoder.zs(state)
		za = self.encoder.za(env_action)
		pred_zs = self.encoder.zsa(zs, za)
		encoder_loss = None
		reconstruction_loss = F.mse_loss(pred_zs, next_zs)
		log_loss = None
		if self.args.encoder == "nflow":
			context = torch.cat([zs, za], dim=-1)
			log_prob = self.encoder.flow.log_prob(next_zs, context=context)
			log_loss = -log_prob.mean()
			encoder_loss = (log_loss * self.hp.log_loss_weight) + reconstruction_loss
		elif self.args.encoder == "addition" or self.args.encoder == "td7":
			encoder_loss = reconstruction_loss
		else:
			raise ValueError(f"[train] Unknown encoder type: {self.args.encoder}")
		if self.training_steps % self.loss_record_freq == 0:
			if self.args.encoder == "nflow":
				self.loss_histories["encoder_log_loss"].append((self.training_steps, float(log_loss.detach().cpu().item())))
			self.loss_histories["encoder_reconstruction_loss"].append((self.training_steps, float(reconstruction_loss.detach().cpu().item())))
			self.loss_histories["encoder_loss"].append((self.training_steps, float(encoder_loss.detach().cpu().item())))
		return encoder_loss
	
	def get_decoder_loss(self, state, env_action, next_state):
		with torch.no_grad():
			zs = self.fixed_encoder.zs(state)
			za = self.fixed_encoder.za(env_action)
			# actor_out = self.actor(state, zs).detach()
			actor_out = self.actor_target(state, zs).detach()
			zsa = self.fixed_encoder.zsa(zs, za)
			# q_true = self.critic(state, env_action, zsa, zs)
			q_true = self.critic_target(state, env_action, zsa, zs)
			q_true = q_true.min(1, keepdim=True)[0].detach()


		pred_action = self.decoder.decode_action(za)
		reconstruction_loss = F.mse_loss(pred_action, env_action)
		
		a_pred = self.decoder.decode_action(actor_out)
		bc_loss = F.mse_loss(a_pred, env_action)

		# q_decode = self.critic(state, pred_action, zsa, zs)			
		q_decode = self.critic_target(state, pred_action, zsa, zs)	
		q_decode = q_decode.min(1, keepdim=True)[0].detach()		
		q_loss = F.mse_loss(q_decode, q_true)

		decoder_loss = reconstruction_loss + (self.hp.decoder_bc_lambda * bc_loss) + (self.hp.decoder_q_lambda * q_loss)

		if self.training_steps % self.loss_record_freq == 0:
			self.loss_histories["decoder_reconstruction_loss"].append((self.training_steps, float(reconstruction_loss.detach().cpu().item())))
			self.loss_histories["decoder_bc_loss"].append((self.training_steps, float(bc_loss.detach().cpu().item())))			
			self.loss_histories["decoder_q_loss"].append((self.training_steps, float(q_loss.detach().cpu().item())))
			self.loss_histories["decoder_loss"].append((self.training_steps, float(decoder_loss.detach().cpu().item())))


		return decoder_loss
	
	def get_actor_loss(self, state):
		with torch.no_grad():
			fixed_zs = self.fixed_encoder.zs(state)
		actor_out = self.actor(state, fixed_zs)
		a = self.fixed_decoder.decode_action(actor_out)
		fixed_za = self.fixed_encoder.za(a)
		fixed_zsa = self.fixed_encoder.zsa(fixed_zs, fixed_za)

		# with torch.no_grad():
		# 	action_rep = actor_out
		# 	fixed_zsa = None
		# 	if self.args.action_space == "embedding":
		# 		action_rep = actor_out
		# 		fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_out)
		# 	elif self.args.action_space == "environment":
		# 		action_rep = actor_out
		# 		fixed_za = self.fixed_encoder.za(action_rep)
		# 		fixed_zsa = self.fixed_encoder.zsa(fixed_zs, fixed_za)

		# Q = self.critic(state, actor_out, fixed_zsa, fixed_zs)
		Q = self.critic(state, a, fixed_zsa, fixed_zs)

		actor_loss = -Q.mean()

		if self.training_steps % self.loss_record_freq == 0:
			self.loss_histories["actor_loss"].append((self.training_steps, float(actor_loss.detach().cpu().item())))

		return actor_loss
	
	def update_lap(self, td_loss):
		priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		self.replay_buffer.update_priority(priority)

	def get_critic_loss(self, state, env_action, next_state, reward, not_done):
		Q_target = None
		fixed_zs = None
		fixed_za = None
		fixed_zsa = None
		with torch.no_grad():
			fixed_target_zs = self.fixed_encoder_target.zs(next_state) #zs at t+1

			next_actor_out = self.actor_target(next_state, fixed_target_zs) #actor_out at t+1
			next_action = self.fixed_decoder_target.decode_action(next_actor_out) #env_action at t+1
			
			noise = (torch.randn_like(next_action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
			next_action = (next_action + noise).clamp(-1,1) #noised action at t+1
			
			fixed_target_za = self.fixed_encoder_target.za(next_action) #noised_action at t+1
			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, fixed_target_za) #zs at t+2

			# action_rep = None
			# if self.args.action_space == "environment":
			# 	action_rep = next_action
			# elif self.args.action_space == "embedding":
			# 	action_rep = fixed_target_za
			action_rep = next_action

			Q_target = self.critic_target(next_state, action_rep, fixed_target_zsa, fixed_target_zs) #q for t+1 to t+2
			# print(f"{Q_target.shape=}")
			min_Q_target = Q_target.min(1,keepdim=True)[0]
			# print(f"{min_Q_target.shape=}")
			Q_target = reward + not_done * self.hp.discount * min_Q_target.clamp(self.min_target, self.max_target) #q for t to t+1
			# print(f"{Q_target.shape=}")
			self.max = max(self.max, float(Q_target.max()))
			self.min = min(self.min, float(Q_target.min()))

			fixed_zs = self.fixed_encoder.zs(state)
			fixed_za = self.fixed_encoder.za(env_action)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, fixed_za)
		# action_rep = None
		# if self.args.action_space == "environment":
		# 	action_rep = env_action
		# if self.args.action_space == "embedding":
		# 	action_rep = fixed_za
		action_rep = env_action
		Q = self.critic(state, action_rep, fixed_zsa, fixed_zs)
		# print(f"{Q.shape=}")
		td_loss = (Q - Q_target).abs()
		# print(f"{td_loss.shape=}")
		critic_loss = LAP_huber(td_loss)
		self.update_lap(td_loss)

		# print(f"{critic_loss.shape=}")

		if self.training_steps % self.loss_record_freq == 0:
			self.loss_histories["critic_td_loss"].append((self.training_steps, float(td_loss.mean().detach().cpu().item())))
			self.loss_histories["critic_loss"].append((self.training_steps, float(critic_loss.detach().cpu().item())))

		return critic_loss
	
	def soft_update_targets(self):
		tau = 1/self.hp.target_update_rate
		target_source = [
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

	def train(self):
		self.training_steps += 1

		state, env_action, next_state, reward, not_done = self.replay_buffer.sample()

		#########################
		# Update Encoder
		#########################
		encoder_loss = self.get_encoder_loss(state, env_action, next_state)
		self.encoder_optimizer.zero_grad()
		encoder_loss.backward()
		# torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.hp.gradient_clip)
		self.encoder_optimizer.step()

		#########################
		# Update Critic
		#########################
		critic_loss = self.get_critic_loss(state, env_action, next_state, reward, not_done)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.hp.gradient_clip)
		self.critic_optimizer.step()
		

		#########################
		# Update Actor
		#########################
		if self.training_steps % self.hp.policy_freq == 0:
			actor_loss = self.get_actor_loss(state)
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			# torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.hp.gradient_clip)
			self.actor_optimizer.step()

		#########################
		# Update Decoder
		#########################
		if not isinstance(self.decoder, IdentityDecoder):
			decoder_loss = self.get_decoder_loss(state, env_action, next_state)
			self.decoder_optimizer.zero_grad()
			decoder_loss.backward()
			# torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.hp.gradient_clip)
			self.decoder_optimizer.step()



		#########################
		# Update Iteration
		#########################
		if self.training_steps % self.hp.target_update_rate == 0:
			# self.actor_target.load_state_dict(self.actor.state_dict())
			# self.critic_target.load_state_dict(self.critic.state_dict())
			# self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
			# self.fixed_encoder.load_state_dict(self.encoder.state_dict())
			# self.fixed_decoder_target.load_state_dict(self.fixed_decoder.state_dict())
			# self.fixed_decoder.load_state_dict(self.decoder.state_dict())
			self.replay_buffer.reset_max_priority()
			self.max_target = self.max
			self.min_target = self.min
		self.soft_update_targets()

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
			self.checkpoint_decoder.load_state_dict(self.fixed_decoder.state_dict())
			
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
		print(f"[Agent.save] Saving checkpoint to {filepath}")
		checkpoint = {
			# Reconstruction info
			"state_dim": self.state_dim,
			"action_dim": self.action_dim,
			"max_action": self.max_action,
			"args": self.args,

			# Hyperparameters (stored as a plain dict for robustness)
			"hp": dict(vars(self.hp)),

			# Networks
			"critic": self.critic.state_dict(),
			"critic_target": self.critic_target.state_dict(),

			"encoder": self.encoder.state_dict(),
			"fixed_encoder": self.fixed_encoder.state_dict(),
			"fixed_encoder_target": self.fixed_encoder_target.state_dict(),

			"decoder": None if self.decoder is None else self.decoder.state_dict(),
			"fixed_decoder": None if self.fixed_decoder is None else self.fixed_decoder.state_dict(),
			"fixed_decoder_target": None if self.fixed_decoder_target is None else self.fixed_decoder_target.state_dict(),

			"actor": self.actor.state_dict(),
			"actor_target": self.actor_target.state_dict(),

			"checkpoint_actor": self.checkpoint_actor.state_dict(),
			"checkpoint_encoder": self.checkpoint_encoder.state_dict(),
			"checkpoint_decoder": self.checkpoint_decoder.state_dict(),

			# Optimizers
			"critic_optimizer": self.critic_optimizer.state_dict(),
			"encoder_optimizer": self.encoder_optimizer.state_dict(),
			"decoder_optimizer": (
				None
				if self.decoder_optimizer is None or isinstance(self.decoder_optimizer, DummyOptimizer)
				else self.decoder_optimizer.state_dict()
			),
			"actor_optimizer": self.actor_optimizer.state_dict(),

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
		print(f"[Agent.save] Saved checkpoint to {filepath}")

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
		max_action = checkpoint["max_action"]

		# Instantiate a fresh Agent (this builds all modules & optimizers)
		agent = cls(
			state_dim=state_dim,
			action_dim=action_dim,
			max_action=max_action,
			args=args,
			hp=hp,
		)

		# Networks
		agent.critic.load_state_dict(checkpoint["critic"])
		agent.critic_target.load_state_dict(checkpoint["critic_target"])

		agent.encoder.load_state_dict(checkpoint["encoder"])
		agent.fixed_encoder.load_state_dict(checkpoint["fixed_encoder"])
		agent.fixed_encoder_target.load_state_dict(checkpoint["fixed_encoder_target"])

		if checkpoint["decoder"] is not None and agent.decoder is not None:
			agent.decoder.load_state_dict(checkpoint["decoder"])
		if checkpoint["fixed_decoder"] is not None and agent.fixed_decoder is not None:
			agent.fixed_decoder.load_state_dict(checkpoint["fixed_decoder"])
		if checkpoint["fixed_decoder_target"] is not None and agent.fixed_decoder_target is not None:
			agent.fixed_decoder_target.load_state_dict(checkpoint["fixed_decoder_target"])

		agent.actor.load_state_dict(checkpoint["actor"])
		agent.actor_target.load_state_dict(checkpoint["actor_target"])

		agent.checkpoint_actor.load_state_dict(checkpoint["checkpoint_actor"])
		agent.checkpoint_encoder.load_state_dict(checkpoint["checkpoint_encoder"])
		if agent.checkpoint_decoder is not None and checkpoint["checkpoint_decoder"] is not None:
			agent.checkpoint_decoder.load_state_dict(checkpoint["checkpoint_decoder"])

		# Optimizers
		agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
		agent.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])

		if (
			checkpoint["decoder_optimizer"] is not None
			and agent.decoder_optimizer is not None
			and not isinstance(agent.decoder_optimizer, DummyOptimizer)
		):
			agent.decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])

		agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

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

		print(f"[Agent.load] Loaded checkpoint from {filepath}")
		return agent
