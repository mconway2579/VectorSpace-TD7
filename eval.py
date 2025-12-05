import matplotlib.pyplot as plt
import os
import numpy as np
import imageio

def maybe_record_videos(RL_agent, eval_env, t, args, extension=""):
    # print(f"Recording videos at timestep {t}")
    if (t % args.record_freq == 0 or extension != "") and args.record_videos:
        header = f"_{t}" if extension == "" else extension
        video_dir = os.path.join(args.recording_dir, f"videos" + header)
        os.makedirs(video_dir, exist_ok=True)
        # print(f"Recording videos to {video_dir}")
        for ep in range(args.record_eps):
            state, info = eval_env.reset()
            done = False
            frames = []
            while not done:
                frame = eval_env.render()
                frames.append(frame)
                action = RL_agent.select_action(state, args.use_checkpoints, use_exploration=False)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
            video_path = os.path.join(video_dir, f"eval_episode_{ep+1}.mp4")
            imageio.mimwrite(video_path, frames, fps=30)
            print(f"Saved video to {video_path}")

