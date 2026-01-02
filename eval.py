import matplotlib.pyplot as plt
import os
import time
import numpy as np
import imageio
import torch
import logging

logger = logging.getLogger(__name__)

def maybe_record_videos(RL_agent, eval_env, t, args, extension=""):
    logger.debug(f"[maybe_record_videos] Recording videos at timestep {t}")
    if (t % args.record_freq == 0 or extension != "") and args.record_videos:
        # Calculate current epsilon for recording

        header = f"_{t}" if extension == "" else extension
        video_dir = os.path.join(args.recording_dir, f"videos" + header)
        os.makedirs(video_dir, exist_ok=True)
        logger.info(f"[maybe_record_videos] Recording videos to {video_dir}")

        # Pre-allocate tensors to avoid repeated allocations
        state_np, info = eval_env.reset()
        state_gpu = torch.empty(state_np.shape, dtype=torch.float32, device=RL_agent.device)

        for ep in range(args.record_eps):
            state_np, info = eval_env.reset()
            state_gpu.copy_(torch.from_numpy(state_np))
            done = False
            frames = []
            while not done:
                frame = eval_env.render()
                frames.append(frame)
                action_np = RL_agent.select_action(state_gpu, args.use_checkpoints, use_exploration=True)
                logger.debug(f"[maybe_record_videos] Action taken: {action_np}")
                state_np, reward, terminated, truncated, _ = eval_env.step(action_np)
                state_gpu.copy_(torch.from_numpy(state_np))
                done = terminated or truncated
            video_path = os.path.join(video_dir, f"eval_episode_{ep+1}.mp4")
            imageio.mimwrite(video_path, frames, fps=30)
            logger.info(f"[maybe_record_videos] Saved video to {video_path}")


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, writer):
	if t % args.eval_freq == 0:
		logger.info("---------------------------------------")
		logger.info(f"Evaluation at {t} time steps")
		logger.info(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

		total_reward = np.zeros(args.eval_eps)

		# Pre-allocate tensors to avoid repeated allocations
		state_np, info = eval_env.reset()
		state_gpu = torch.empty(state_np.shape, dtype=torch.float32, device=RL_agent.device)

		for ep in range(args.eval_eps):
			state_np, info = eval_env.reset()
			state_gpu.copy_(torch.from_numpy(state_np))
			logger.debug(f"[maybe_evaluate_and_print] {state_gpu.shape=}")
			done = False
			while not done:
				action_np = RL_agent.select_action(state_gpu, args.use_checkpoints, use_exploration=False)
				logger.debug(f"[maybe_evaluate_and_print] {action_np=}")
				state_np, reward, terminated, truncated, _ = eval_env.step(action_np)
				state_gpu.copy_(torch.from_numpy(state_np))
				logger.debug(f"[maybe_evaluate_and_print] next {state_gpu.shape=}")
				done = terminated or truncated
				total_reward[ep] += reward

		logger.info(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
		logger.info("---------------------------------------")
		evals.append(total_reward)
		np.save(f"{args.save_dir}/evals.npy", evals)

		writer.add_scalar("eval/mean_reward", total_reward.mean(), t)
		writer.add_scalar("eval/std_reward", total_reward.std(), t)

