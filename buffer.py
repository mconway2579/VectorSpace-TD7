import numpy as np
import torch


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

        # Handle state_dim as int or tuple
        if isinstance(state_dim, tuple):
            state_shape = (max_size, *state_dim)
        else:
            state_shape = (max_size, state_dim)

        # Buffers on CPU
        # self.state = np.zeros(state_shape, dtype=np.float32)
        # self.next_state = np.zeros(state_shape, dtype=np.float32)
        # self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        # self.reward = np.zeros((max_size, 1), dtype=np.float32)
        # self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.state = np.zeros(state_shape)
        self.next_state = np.zeros(state_shape)
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        # # Prioritized replay
        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1.0

        # Per-dimension normalization
        self.normalize_actions = normalize_actions
        self.low_action_arr = np.array(low_action_arr, dtype=np.float32)
        self.high_action_arr = np.array(high_action_arr, dtype=np.float32)
        self.action_range = self.high_action_arr - self.low_action_arr

    def _normalize_action(self, action):
        """Per-dimension action normalization to [-1, 1]."""
        return 2.0 * (action - self.low_action_arr) / self.action_range - 1.0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        if self.normalize_actions:
            action = self._normalize_action(action)
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)

        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        if self.prioritized:
            csum = torch.cumsum(self.priority[:self.size], 0)
            val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
            ind = torch.searchsorted(csum, val).cpu().numpy()
        else:
            ind = np.random.randint(0, self.size, size=self.batch_size)

        self.ind = ind  # Save sampled indices for priority updates
        return (
            torch.tensor(self.state[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.action[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_state[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.reward[ind], dtype=torch.float32, device=self.device),
            torch.tensor(self.not_done[ind], dtype=torch.float32, device=self.device)
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.detach().max().item()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[:self.size].max())
