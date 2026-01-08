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
		if isinstance(state_dim, tuple):
			self.state_shape = state_dim
			self.state = torch.zeros((max_size, *state_dim), dtype=torch.float32, device=device)
			self.next_state = torch.zeros((max_size, *state_dim), dtype=torch.float32, device=device)
		else:
			self.state_shape = (state_dim,)
			self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
			self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)

		self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
		self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
		self.not_done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

		self.prioritized = prioritized
		if prioritized:
			self.priority = torch.zeros(max_size, device=device)
			self.max_priority = torch.tensor(1.0, device=device)  # Keep on GPU to avoid sync
			self._csum_buffer = torch.zeros(max_size, device=device)  # Pre-allocated for cumsum

		# Store action bounds for per-dimension normalization (on GPU)
		self.normalize_actions = normalize_actions
		self.low_action_arr = torch.tensor(low_action_arr, dtype=torch.float32, device=device)
		self.high_action_arr = torch.tensor(high_action_arr, dtype=torch.float32, device=device)
		self.action_range = self.high_action_arr - self.low_action_arr


	def add(self, state, action, next_state, reward, done):
		logger.debug(f"[add] {self.ptr=}, {self.size=}, {state.shape=}, {action.shape=}")
		# Ensure action has correct shape
		if action.ndim == 0:
			action = action.unsqueeze(0)
		# Store values
		self.state[self.ptr] = state
		# Normalize action to [-1, 1] per dimension: normalized = 2 * (action - low) / (high - low) - 1
		if self.normalize_actions:
			normalized_action = 2.0 * (action - self.low_action_arr) / self.action_range - 1.0
			if torch.isnan(normalized_action).any() or torch.isinf(normalized_action).any():
				logger.error(f"[add] NaN/Inf in normalized action! {action=}, {self.low_action_arr=}, {self.action_range=}")
			self.action[self.ptr] = normalized_action
		else:
			self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - float(done)

		if self.prioritized:
			self.priority[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self):
		# Sample indices
		if self.prioritized:
			# Reuse pre-allocated buffer to avoid GPU memory fragmentation
			torch.cumsum(self.priority[:self.size], 0, out=self._csum_buffer[:self.size])
			csum = self._csum_buffer[:self.size]
			val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
			self.ind = torch.searchsorted(csum, val)
			logger.debug(f"[sample] {self.size=}, {csum.shape=}, {csum[-1]=}, {val.min()=}, {val.max()=}")
			logger.debug(f"[sample] {self.ind.min()=}, {self.ind.max()=}, max_valid={self.size - 1}")
			# Clamp to valid range - searchsorted can return self.size due to float precision
			if self.ind.max() >= self.size:
				logger.warning(f"[sample] Out of bounds index detected! {self.ind.max()=} >= {self.size=}")
				self.ind = self.ind.clamp(max=self.size - 1)
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
		logger.debug(f"[update_priority] {priority.shape=}, {self.ind.shape=}, {self.ind.max()=}")
		if torch.isnan(priority).any() or torch.isinf(priority).any():
			logger.error(f"[update_priority] NaN/Inf in priority! {priority=}")
		self.priority[self.ind] = priority.reshape(-1).detach()
		# Keep max_priority on GPU to avoid CPU-GPU sync every step
		batch_max = priority.detach().max()
		self.max_priority = torch.maximum(batch_max, self.max_priority)


	def reset_max_priority(self):
		self.max_priority = self.priority[:self.size].max()  # Keep on GPU
