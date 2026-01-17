import logging
import torch

logger = logging.getLogger(__name__)


class LAP(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		device,
		low_action_arr,
		high_action_arr,
		max_size=1e6,
		batch_size=256,
		normalize_actions=True,
		prioritized=True
	):

		max_size = int(max_size)
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size
		self.action_dim = action_dim

		# Store everything in VRAM
		state_buffer_shape = None
		if isinstance(state_dim, tuple):
			state_buffer_shape = (max_size, *state_dim)
		else:
			state_buffer_shape = (max_size, state_dim)

		self.state = torch.zeros(state_buffer_shape, dtype=torch.float32, device=device)
		self.next_state = torch.zeros(state_buffer_shape, dtype=torch.float32, device=device)
		self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
		self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
		self.not_done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

		self.prioritized = prioritized
		if prioritized:
			self.priority = torch.zeros(max_size, device=device)
			self.max_priority = torch.tensor(1.0, device=device)  # Keep on GPU to avoid sync

		# Store action bounds for per-dimension normalization (on GPU)
		self.normalize_actions = normalize_actions
		self.low_action_arr = torch.tensor(low_action_arr, dtype=torch.float32, device=device)
		self.high_action_arr = torch.tensor(high_action_arr, dtype=torch.float32, device=device)
		self.action_range = self.high_action_arr - self.low_action_arr

		# Pre-allocate index buffer
		self.ind = torch.zeros(batch_size, dtype=torch.long, device=device)

	@torch.compile(mode="reduce-overhead")
	def _normalize_action(self, action, low_arr, action_range):
		"""Normalize action - compiled for performance."""
		return 2.0 * (action - low_arr) / action_range - 1.0

	def add(self, state, action, next_state, reward, done):
		# Normalize action inline
		if self.normalize_actions:
			action = self._normalize_action(action, self.low_action_arr, self.action_range)

		# Store values - use non_blocking=True for async GPU transfers
		self.state[self.ptr].copy_(state, non_blocking=True)
		self.action[self.ptr].copy_(action, non_blocking=True)
		self.next_state[self.ptr].copy_(next_state, non_blocking=True)
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - float(done)

		if self.prioritized:
			self.priority[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def _sample_indices_prioritized(self):
		"""Prioritized sampling - dynamic size incompatible with CUDA graphs."""
		csum = torch.cumsum(self.priority[:self.size], 0)
		rand_scaled = torch.rand(self.batch_size, device=self.device) * csum[-1]
		return torch.searchsorted(csum, rand_scaled).clamp(max=self.size - 1)

	def sample(self):
		if self.prioritized:
			self.ind = self._sample_indices_prioritized()
		else:
			self.ind = torch.randint(0, self.size, size=(self.batch_size,), device=self.device)

		return (
			self.state[self.ind],
			self.action[self.ind],
			self.next_state[self.ind],
			self.reward[self.ind],
			self.not_done[self.ind]
		)

	def update_priority(self, priority):
		"""Update priorities for sampled transitions."""
		priority_flat = priority.detach().reshape(-1)
		self.priority[self.ind] = priority_flat
		self.max_priority = torch.maximum(priority_flat.max(), self.max_priority)


	def reset_max_priority(self):
		self.max_priority = self.priority[:self.size].max()  # Keep on GPU
