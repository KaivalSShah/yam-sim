from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ball_env import BallBounceEnv

ROOT = Path(__file__).parent

if __name__ == "__main__":
    n = 1
    while (ROOT / "checkpoints" / f"train{n}").exists():
        n += 1
    name = f"train{n}"
    save_dir = ROOT / "checkpoints" / name
    save_dir.mkdir(parents=True)
    print(f"saving checkpoints to {save_dir}")

    env = make_vec_env(BallBounceEnv, n_envs=16, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"})
    model = PPO("MultiInputPolicy", env, n_steps=256, verbose=1, tensorboard_log=str(ROOT / "runs"), device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.learn(
        total_timesteps=2_000_000,
        callback=CheckpointCallback(save_freq=50_000 // 16, save_path=str(save_dir)),
        tb_log_name=name,
    )
    model.save(save_dir / "final")
