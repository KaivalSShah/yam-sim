from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from cameras import CameraFeed
from initialize_environment import initialize_scene
from reward import compute_reward
from utils import detect_ball_contact

MODEL = Path(__file__).parent / "model" / "scene_linear_4310.xml"


class BallBounceEnv(gym.Env):
    FRAME_SKIP = 10
    ACTION_SCALE = 0.05
    MAX_STEPS = 1000
    MAX_STEPS_WITHOUT_BOUNCE = 100
    MAX_LATERAL = 0.3

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.ctrl_low, self.ctrl_high = self.model.actuator_ctrlrange[:6].T
        self.overhead = CameraFeed(self.model, "overhead_cam", width=64, height=64)
        self.wrist = CameraFeed(self.model, "wrist_cam", width=96, height=54)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32),
            "overhead": gym.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
            "wrist": gym.spaces.Box(0, 255, shape=(54, 96, 3), dtype=np.uint8),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initialize_scene(self.model, self.data)
        blade = self.data.geom("blade").xpos
        ball = self.data.joint("ball_free")
        ball.qpos[:3] = blade + [*self.np_random.uniform(-0.03, 0.03, 2), 0.15]
        ball.qvel[:] = 0.0
        ball.qvel[:2] = self.np_random.uniform(-0.2, 0.2, 2)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.last_bounce = 0
        self.bounces = 0
        self.was_on_paddle = False
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:6] = np.clip(self.data.ctrl[:6] + self.ACTION_SCALE * action, self.ctrl_low, self.ctrl_high)

        bounced = on_floor = False
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            on_paddle, floor = detect_ball_contact(self.model, self.data)
            if self.was_on_paddle and not on_paddle and self.data.joint("ball_free").qvel[2] > 0:
                bounced = True
            self.was_on_paddle = on_paddle
            on_floor |= floor

        reward = compute_reward(self.data, bounced)
        self.steps += 1
        if reward > 0:
            self.bounces += 1
            self.last_bounce = self.steps

        offset = self.data.body("ball").xpos - self.data.geom("blade").xpos
        terminated = (
            on_floor
            or np.linalg.norm(offset[:2]) > self.MAX_LATERAL
            or self.steps - self.last_bounce > self.MAX_STEPS_WITHOUT_BOUNCE
        )
        truncated = self.steps >= self.MAX_STEPS
        return self._obs(), reward, terminated, truncated, {"bounces": self.bounces}

    def _obs(self):
        blade = self.data.geom("blade")
        ball_pos = self.data.body("ball").xpos
        ball_vel = self.data.joint("ball_free").qvel[:3]
        state = np.concatenate([
            self.data.qpos[:6],
            self.data.qvel[:6],
            blade.xpos,
            ball_pos - blade.xpos,
            ball_vel,
        ]).astype(np.float32)
        return {
            "state": state,
            "overhead": self.overhead.frame(self.data),
            "wrist": self.wrist.frame(self.data),
        }
