#!/usr/bin/env python3
import argparse
import gc
import logging
import os
import shutil
import time

import gymnasium as gym
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

import TD7

logger = logging.getLogger(__name__)
from gymnasium.wrappers import TransformObservation, AtariPreprocessing, FrameStackObservation
from gymnasium.vector import AsyncVectorEnv
from eval import maybe_record_videos, maybe_evaluate_and_print
from torch.utils.tensorboard import SummaryWriter
import ale_py
from utils import Hyperparameters
from contextlib import nullcontext


class ContinuousAtariPreprocessing(AtariPreprocessing):
	"""AtariPreprocessing wrapper that handles continuous action spaces."""

	def reset(self, **kwargs):
		"""Reset the environment, handling continuous action spaces during NOOP steps."""
		_, info = self.env.reset(**kwargs)

		# Execute NOOP actions
		noops = (
			self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
			if self.noop_max > 0
			else 0
		)
		# For continuous action space, create a zero action vector
		if hasattr(self.env.action_space, 'shape'):
			# Continuous action space
			noop_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
		else:
			# Discrete action space
			noop_action = 0

		for _ in range(noops):
			_, _, terminated, truncated, step_info = self.env.step(noop_action)
			info.update(step_info)
			if terminated or truncated:
				_, info = self.env.reset(**kwargs)

		# Get the current observation
		obs = self._get_obs()
		self.lives = self.env.unwrapped.ale.lives()

		return obs, info

def make_env_with_wrappers(env_name, seed=None, render_mode=None, continuous_action_threshold=None):
	"""Create a single environment with all necessary wrappers."""
	# Check if this is an Atari environment
	is_atari = env_name.startswith("ALE/")

	# For Atari, disable frameskip in base env (AtariPreprocessing will handle it)
	if is_atari:
		# Use continuous_action_threshold if provided, otherwise default to None (fully continuous)
		if continuous_action_threshold is not None:
			env = gym.make(env_name, render_mode=render_mode, frameskip=1, continuous=True, continuous_action_threshold=continuous_action_threshold)
		else:
			env = gym.make(env_name, render_mode=render_mode, frameskip=1, continuous=True)
		
		# Use standard Atari preprocessing: resize to 84x84, grayscale, etc.
		env = ContinuousAtariPreprocessing(
			env,
			screen_size=84,
			grayscale_obs=True,
			scale_obs=True,  # Normalize to [0, 1]
			terminal_on_life_loss=True
		)
		# Stack 4 frames for temporal information
		env = FrameStackObservation(env, stack_size=4)
		# Convert to numpy array and ensure correct shape (channels, height, width)
		env = TransformObservation(
			env,
			lambda obs: np.array(obs, dtype=np.float32),  # Stacked frames -> (4, 84, 84)
			gym.spaces.Box(low=0.0, high=1.0, shape=(4, 84, 84), dtype=np.float32)
		)
	else:
		env = gym.make(env_name, render_mode=render_mode)
	if seed is not None:
		env.reset(seed=seed)
		env.action_space.seed(seed)		
	
	return env
	

def train_online(RL_agent, env, eval_env, args, writer):

	evals = []
	start_time = time.time()
	allow_train = False

	# Setup profiler if requested
	profiler_ctx = nullcontext()
	if args.profile:
		# Minimal profiling to prevent TensorBoard crashes
		# wait=5000: very large gaps (profile every ~5000 steps)
		# warmup=100, active=300: only 300 steps per window
		# repeat=8: only 8 total windows (~8 small trace files)
		prof_schedule = schedule(wait=5000, warmup=100, active=300, repeat=8)

		# Create profile directory for TensorBoard plugin
		profile_dir = os.path.join(writer.log_dir, 'plugins', 'profile')
		os.makedirs(profile_dir, exist_ok=True)

		profiler_ctx = profile(
			activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
			schedule=prof_schedule,
			on_trace_ready=tensorboard_trace_handler(profile_dir),
			record_shapes=False,  # Disabled to reduce data size
			profile_memory=False,  # Disabled to reduce data size
			with_stack=False  # Disabled to reduce data size
		)		
	logger.info(f"Starting training with profiler {profiler_ctx}")
	with profiler_ctx as prof:
		# Initialize state and episode tracking
		state_np, info = env.reset(seed=args.seed)
		# Pre-allocate GPU tensors to avoid repeated CPU->GPU copies
		state_gpu = torch.empty(state_np.shape, dtype=torch.float32, device=RL_agent.device)
		action_gpu = torch.empty((env.action_space.shape[0],), dtype=torch.float32, device=RL_agent.device)
		reward_gpu = torch.empty((1,), dtype=torch.float32, device=RL_agent.device)
		next_state_gpu = torch.empty_like(state_gpu)

		state_gpu.copy_(torch.from_numpy(state_np))
		logger.debug(f"[train_online] {state_gpu.shape=}")

		ep_num = 0
		ep_reward = 0
		ep_timesteps = 0
		total_steps = 0
		while total_steps < args.max_timesteps:
			maybe_evaluate_and_print(RL_agent, eval_env, evals, total_steps, start_time, args, writer)
			maybe_record_videos(RL_agent, eval_env, total_steps, args)

			# Get actions for current state
			if allow_train:
				action_np = RL_agent.select_action(state_gpu)
			else:
				action_np = env.action_space.sample()

			action_gpu.copy_(torch.from_numpy(action_np))
			logger.debug(f"[train_online] {action_np.shape=}")
			logger.debug(f"[train_online] {action_gpu=}")

			next_state_np, reward, terminated, truncated, _ = env.step(action_np)
			# Reuse pre-allocated tensor instead of creating new one
			reward_gpu.fill_(reward)
			next_state_gpu.copy_(torch.from_numpy(next_state_np))
			# print(f"[train_online] {next_states.shape=}")
			done = terminated | truncated

			# Update episode tracking
			ep_reward += reward
			ep_timesteps += 1
			total_steps += 1

			# Add transitions to replay buffer
			RL_agent.replay_buffer.add(state_gpu, action_gpu, next_state_gpu, reward_gpu, done)

			# Train after each step (for non-checkpoint mode)
			if allow_train and not args.use_checkpoints:
				RL_agent.train()

			# Handle episode terminations
			if done:
				ep_num += 1
				logger.info(f"Total T: {total_steps}/{int(args.max_timesteps)} | Episode {ep_num} | Steps: {total_steps} | Reward: {ep_reward:.3f} | Term: {terminated}, Trunc: {truncated}")
				writer.add_scalar("train/episode_reward", ep_reward, total_steps)
				writer.add_scalar("train/episode_timesteps", ep_timesteps, total_steps)

				if allow_train and args.use_checkpoints:
					RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_reward)

				ep_timesteps = 0
				ep_reward = 0
				reset_state_np, info = env.reset()
				next_state_gpu.copy_(torch.from_numpy(reset_state_np))

			if total_steps >= args.timesteps_before_training:
				if not allow_train:  # First time training starts
					logger.info(f"Training started at step {total_steps}")
				allow_train = True

			# Swap state and next_state tensors to avoid copying
			state_gpu, next_state_gpu = next_state_gpu, state_gpu

			# Step profiler if enabled (only after training begins)
			if prof is not None and allow_train:
				prof.step()

		return evals


def main(args):
	# Setup logging
	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format='%(message)s'
	)

	if not os.path.exists("./results"):
		os.makedirs("./results")
	

	save_dir = os.path.join("./results", args.dir_name)
	os.makedirs(save_dir, exist_ok=True)
	args.save_dir = save_dir

	recording_dir = os.path.join(save_dir, "recordings")
	os.makedirs(recording_dir, exist_ok=True)
	args.recording_dir = recording_dir

	model_dir = os.path.join(save_dir, "models")
	os.makedirs(model_dir, exist_ok=True)
	args.model_dir = model_dir
	
	tb_dir = os.path.join(save_dir, "tensorboard")
	if os.path.exists(tb_dir):
		logger.warning(f"Tensorboard directory {tb_dir} already exists - removing it")
		shutil.rmtree(tb_dir)
	writer = SummaryWriter(log_dir=tb_dir, flush_secs=args.writer_flush_seconds)


	# Create training environment
	env = make_env_with_wrappers(args.env, seed=args.seed, render_mode="rgb_array", continuous_action_threshold=args.continuous_action_threshold)


	# Evaluation environment
	eval_env = make_env_with_wrappers(args.env, seed=args.seed + 100, render_mode="rgb_array", continuous_action_threshold=args.continuous_action_threshold)

	logger.info("---------------------------------------")
	logger.info(f"Algorithm: TD7, Env: {args.env}, Seed: {args.seed}")
	logger.info(f"Saved directory: {save_dir}")
	logger.info("---------------------------------------")
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# # CUDA optimizations for speed
	if torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
		# Configure precision mode
		# FP32 precision modes for matmul/conv
		torch.backends.cudnn.conv.fp32_precision = args.precision
		torch.backends.cuda.matmul.fp32_precision = args.precision

	# Get action space info from the single (unwrapped) action space
	# AsyncVectorEnv has .single_action_space attribute
	# single_action_space = env.single_action_space
	# state_dim = env.single_observation_space.shape
	action_space = env.action_space
	action_dim = env.action_space.shape[0]
	state_dim = env.observation_space.shape

	low_action = action_space.low
	high_action = action_space.high
	logger.info(f"{low_action=}, {high_action=}")

	# Configure hyperparameters based on environment type
	is_atari = args.env.startswith("ALE/")
	if is_atari:
		hp = Hyperparameters(
			# Atari-specific hyperparameters
			batch_size=32,
			buffer_size=50_000,
			discount=0.99,
			target_update_rate=1_000,
			exploration_noise=0.1,
			target_policy_noise=0.2,
			noise_clip=0.5,
			policy_freq=2,
			alpha=0.4,
			min_priority=1,
			encoder_dim=512,
			enc_hdim=512,
			encoder_lr=6.25e-5,
			backbone_lr=6.25e-5,
			decoder_lr=6.25e-5,
			critic_lr=6.25e-5,
			actor_lr=6.25e-5,
			actor_hdim=512,
			critic_hdim=512,
			decoder_hdim=512,
		)
		logger.info("Using Atari hyperparameters")
	else:
		hp = Hyperparameters()
		logger.info("Using continuous control hyperparameters")

	RL_agent = TD7.Agent(state_dim, action_dim, low_action, high_action, args, writer, hp)
	logger.info(f"Agent device: {RL_agent.device}")
	logger.info(f"Encoder on GPU: {next(RL_agent.encoder.parameters()).is_cuda}")
	logger.info(f"Buffer device: {RL_agent.replay_buffer.device}")

	total_reward_samples = train_online(RL_agent, env, eval_env, args, writer)
	maybe_record_videos(RL_agent, eval_env, 0, args, extension="_final")

	final_model_save_path = os.path.join(args.model_dir, "final_model.pt")
	logger.info(f"saving final model to {final_model_save_path}")
	RL_agent.save(final_model_save_path)

	# tmp_agent = TD7.Agent.load(final_model_save_path)

	# Close gym envs explicitly
	env.close()
	eval_env.close()
	writer.close()

	# Drop references to big objects
	del RL_agent
	del env, eval_env
	del total_reward_samples


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Encoder
	parser.add_argument("--encoder", type=str, choices=["addition", "td7", "nflow"], default="td7",
						help="Which encoder to use ('addition', 'td7', or 'nflow').")
	# RL
	# parser.add_argument("--env", default="HalfCheetah-v5", type=str)
	parser.add_argument("--env", default="ALE/Breakout-v5", type=str)

	parser.add_argument("--deterministic_actor", default=True, action=argparse.BooleanOptionalAction)
	parser.add_argument("--hard_updates", default=False, action=argparse.BooleanOptionalAction,
						help="Use hard target updates (like reference TD7) instead of soft Polyak averaging")

	
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument('--use_checkpoints', default=True, action=argparse.BooleanOptionalAction)

	# Evaluation
	parser.add_argument("--timesteps_before_training", default=25e3, type=int)
	parser.add_argument("--eval_freq", default=5e3, type=int)
	# parser.add_argument("--eval_freq", default=10e3, type=int)

	parser.add_argument("--eval_eps", default=10, type=int)
	# parser.add_argument("--max_timesteps", default=5_000_000, type=int)
	parser.add_argument("--max_timesteps", default=1_000_000, type=int)
	# parser.add_argument("--max_timesteps", default=100_000, type=int)
	# parser.add_argument("--max_timesteps", default=35_000, type=int)

	parser.add_argument("--precision", default="tf32", type=str, choices=["tf32", "ieee"], help="Precision mode for FP32 matmul/conv")
	# Recording
	parser.add_argument("--record_videos", default=True, action=argparse.BooleanOptionalAction)
	parser.add_argument("--record_freq", default=None, type=int)
	parser.add_argument("--record_eps", default=5, type=int)

	# File
	parser.add_argument('--dir_name', default=None)
	parser.add_argument("--writer_flush_seconds", default=30, type=int)

	# Logging
	# parser.add_argument("--log_level", default="DEBUG", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
	parser.add_argument("--log_level", default="INFO", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

	# Atari continuous action threshold
	parser.add_argument("--continuous_action_threshold", default=0.5, type=float, help="Threshold for continuous action space in Atari (0-1). CALE paper uses 0.5.")

	# Profiling
	parser.add_argument("--profile", action="store_true", default=False, help="Enable TensorBoard profiler (profiles steps with schedule)")
	parser.add_argument("--log_gradients", action="store_true", default=False, help="Enable gradient logging to TensorBoard (causes CPU-GPU sync, reduces performance)")

	def apply_dynamic_defaults(args):
		if args.record_freq is None:
			dynamic_record_freq = args.max_timesteps // 10 if args.max_timesteps >= 1_000_000 else np.inf
			args.record_freq = dynamic_record_freq

		if args.profile:
			logger.info("Profiling enabled - this may slow down training significantly!")
			args.max_timesteps = 35_000  # Limit to 35k steps when profiling
			args.eval_freq = 5_000
			args.timesteps_before_training = 5_000

		# Disable checkpointing for Atari environments (designed for continuous control)
		is_atari = args.env.startswith("ALE/")
		if is_atari:
			logger.info(f"Detected Atari environment ({args.env}).")
			args.use_checkpoints = False
			args.max_timesteps = max(args.max_timesteps, 5_000_000)  # Ensure sufficient training time for Atari
		else:
			args.use_checkpoints = True

		return args
	
	for env in ["HalfCheetah-v5"]:#, "Ant-v5", "Hopper-v5", "ALE/Assault-v5"]:#["ALE/Assault-v5", "HalfCheetah-v5"]:#["ALE/Pong-v5", "ALE/Breakout-v5", "HalfCheetah-v5"]:#["Ant-v5", "Hopper-v5"]:#["HalfCheetah-v5"]:#,  #, , "Humanoid-v5", ]:["Humanoid-v5"]:#
		for deterministic_actor in [True]:#[False, True]:
			for encoder in ["td7"]:#, "nflow"]:#, "addition"]:
				args = parser.parse_args()

				args.env = env
				args.encoder = encoder
				args.deterministic_actor = deterministic_actor

				args = apply_dynamic_defaults(args)
				profiler_str = "_profiler" if args.profile else ""
				actor_str = f"DeterministicActor" if args.deterministic_actor else "ProbabilisticActor"
				args.dir_name = f"test_{encoder}_{args.env.replace('/', '_')}_{actor_str}_seed_{args.seed}_{int(args.max_timesteps)}{profiler_str}"

				main(args)
				gc.collect()  # Python garbage collection
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
					torch.cuda.ipc_collect()
				# For MPS (Apple Silicon) if you ever use it:
				if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
					torch.mps.empty_cache()

	
