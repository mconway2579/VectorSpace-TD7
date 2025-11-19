import argparse
import gc
import os
import time

import gymnasium as gym
import numpy as np
import torch

import TD7
from gymnasium.wrappers import TimeLimit
from eval import maybe_record_videos, plot_rewards, maybe_plot_loss_histories

def train_online(RL_agent, env, eval_env, args):
	evals = []
	start_time = time.time()
	allow_train = False

	state, info = env.reset(seed=args.seed)
	ep_finished = False
	ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

	for t in range(int(args.max_timesteps+1)):
		maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
		maybe_record_videos(RL_agent, eval_env, t, args)
		maybe_plot_loss_histories(RL_agent.loss_histories, t, args)
		
		if allow_train:
			action = RL_agent.select_action(np.array(state))
		else:
			action = env.action_space.sample()

		next_state, reward, terminated, truncated, _ = env.step(action)
		ep_finished = terminated or truncated
		
		ep_total_reward += reward
		ep_timesteps += 1

		RL_agent.replay_buffer.add(state, action, next_state, reward, ep_finished)

		state = next_state

		if allow_train and not args.use_checkpoints:
			RL_agent.train()

		if ep_finished: 
			print(f"Total T: {t+1}/{int(args.max_timesteps)} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f} {terminated=}, {truncated=}")

			if allow_train and args.use_checkpoints:
				RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

			if t >= args.timesteps_before_training:
				allow_train = True

			# state, info = env.reset(seed=args.seed)
			state, info = env.reset()

			ep_finished = False
			ep_total_reward, ep_timesteps = 0, 0
			ep_num += 1 
	return evals


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args):
	if t % args.eval_freq == 0:
		print("---------------------------------------")
		print(f"Evaluation at {t} time steps")
		print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

		total_reward = np.zeros(args.eval_eps)
		for ep in range(args.eval_eps):
			state, info = eval_env.reset()
			done = False
			while not done:
				action = RL_agent.select_action(state, args.use_checkpoints, use_exploration=False)
				state, reward, terminated, truncated, _ = eval_env.step(action)
				done = terminated or truncated
				total_reward[ep] += reward

		print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
		print("---------------------------------------")

		evals.append(total_reward)
		np.save(f"{args.save_dir}/evals.npy", evals)
		plot_rewards(evals, args)


def main(args):
	if args.dir_name is None:
		args.dir_name = f"{args.encoder}_{args.env}_seed_{args.seed}"

	if not os.path.exists("./results"):
		os.makedirs("./results")

	save_dir = os.path.join("./results", args.dir_name)
	os.makedirs(save_dir, exist_ok=True)
	args.save_dir = save_dir

	plot_dir = os.path.join(save_dir, "plots")
	os.makedirs(plot_dir, exist_ok=True)
	args.plot_dir = plot_dir

	recording_dir = os.path.join(save_dir, "recordings")
	os.makedirs(recording_dir, exist_ok=True)
	args.recording_dir = recording_dir

	env = gym.make(args.env, render_mode="rgb_array")
	eval_env = gym.make(args.env, render_mode="rgb_array")

	print("---------------------------------------")
	print(f"Algorithm: TD7, Env: {args.env}, Seed: {args.seed}")
	print(f"Saved directory: {save_dir}")
	print("---------------------------------------")

	env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	eval_env.reset(seed=args.seed+100)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	RL_agent = TD7.Agent(state_dim, action_dim, max_action, args)

	total_reward_samples = train_online(RL_agent, env, eval_env, args)
	plot_rewards(total_reward_samples, args)
	maybe_record_videos(RL_agent, eval_env, 0, args, extension="_final")
	maybe_plot_loss_histories(RL_agent.loss_histories, np.ceil(args.max_timesteps/args.plot_loss_freq)*args.plot_loss_freq, args)

	model_dir = os.path.join(save_dir, "models")
	os.makedirs(model_dir, exist_ok=True)
	print("this is where I would save the models")

	# Close gym envs explicitly
	env.close()
	eval_env.close()

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
	parser.add_argument("--env", default="HalfCheetah-v5", type=str)
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument('--use_checkpoints', default=True, action=argparse.BooleanOptionalAction)

	#Action Behavior
	parser.add_argument("--action_space", type=str, choices=["environment", "embedding"], default="environment",
						help="Which space to produce actions in ('environment', or 'embedding').")
	# Evaluation
	parser.add_argument("--timesteps_before_training", default=25e3, type=int)
	parser.add_argument("--eval_freq", default=5e3, type=int)
	parser.add_argument("--eval_eps", default=10, type=int)
	# parser.add_argument("--max_timesteps", default=5e6, type=int)
	parser.add_argument("--max_timesteps", default=1e6, type=int)
	# parser.add_argument("--max_timesteps", default=4e4, type=int)

	# Recording
	parser.add_argument("--record_videos", default=True, action=argparse.BooleanOptionalAction)
	parser.add_argument("--record_freq", default=1e5, type=int)
	parser.add_argument("--record_eps", default=5, type=int)
	parser.add_argument("--plot_loss_freq", default=1e4, type=int)
	# File
	parser.add_argument('--dir_name', default=None)
	
	#main(args)
	for action_space in ["environment", "embedding"]:
		for encoder in ["td7", "nflow", "addition"]:
			args = parser.parse_args()
			# if action_space == "embedding" and encoder == "td7":
			# 	continue
			args.encoder = encoder
			args.action_space = action_space
			if action_space == "environment":
				args.dir_name = f"{encoder}_{args.env}_envaction_seed_{args.seed}"
			else:
				args.dir_name = f"{encoder}_{args.env}_embaction_seed_{args.seed}"
			main(args)
			gc.collect()  # Python garbage collection
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				torch.cuda.ipc_collect()
			# For MPS (Apple Silicon) if you ever use it:
			if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				torch.mps.empty_cache()

	
