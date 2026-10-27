import sys
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from ball_env import BallBounceEnv
from initialize_environment import initialize_camera

CHECKPOINTS = Path(__file__).parent / "checkpoints"
follow = len(sys.argv) == 1


def newest_checkpoint():
    while True:
        files = sorted(CHECKPOINTS.rglob("*.zip"), key=lambda p: p.stat().st_mtime)
        if files:
            return files[-1]
        print("waiting for the first checkpoint...")
        time.sleep(5)


path = newest_checkpoint() if follow else Path(sys.argv[1])
policy = PPO.load(path)
print(f"playing {path.parent.name}/{path.name}")

env = BallBounceEnv()
obs, _ = env.reset()
recorders = initialize_camera(env.model, Path(__file__).parent / "recordings")
dt = env.FRAME_SKIP * env.model.opt.timestep

try:
    with mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            start = time.time()
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            for recorder in recorders:
                recorder.capture(env.data)
            if terminated or truncated:
                print(f"{path.parent.name}/{path.name}: {info['bounces']} bounces")
                if follow and newest_checkpoint() != path:
                    try:
                        path = newest_checkpoint()
                        policy = PPO.load(path)
                        print(f"playing {path.parent.name}/{path.name}")
                    except Exception:
                        pass
                obs, _ = env.reset()
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - start)))
finally:
    for recorder in recorders:
        recorder.close()
