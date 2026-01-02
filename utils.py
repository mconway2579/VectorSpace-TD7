import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable
import torch.nn.functional as F
import torch.nn as nn


def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

def LAP_huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class DummyOptimizer:
	def __init__(self):
		self.param_groups = []
	def zero_grad(self, set_to_none=True):
		pass
	
	def step(self):
		pass


@dataclass
class Hyperparameters:
	# Generic
	batch_size: int = 256
	buffer_size: int = 1_000_000  # Standard DQN buffer size
	discount: float = 0.99
	target_update_rate: int = 256  # DQN: update target network every 10k steps
	exploration_noise: float = 0.1
	gradient_clip: float = 10.0
	
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
	encoder_dim: int = 256
	enc_hdim: int = 256
	enc_activ: Callable = nn.ELU
	encoder_lr: float = 3e-4
	log_loss_weight: float = 1.0
	prediction_loss_lambda: float = 1.0
	reconstruction_loss_lambda: float = 1.0

	#backbone Model
	backbone_lr: float = 2.5e-4

	# Decoder Model
	decoder_hdim: int = 256
	decoder_activ: Callable = nn.ELU
	decoder_bc_lambda: float = 1.0
	decoder_q_lambda: float = 1.0
	decoder_lr: float = 3e-4

	# Critic Model
	critic_hdim: int = 256
	critic_activ: Callable = nn.ELU
	critic_lr: float = 3e-4
	
	# Actor Model
	actor_hdim: int = 256
	actor_activ: Callable = nn.ReLU
	actor_lr: float = 3e-4

