"""Minimal MuJoCo sim of the YAM arm. On macOS run with `mjpython sim.py`."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

from initialize_environment import initialize_camera, initialize_scene

MODEL = Path(__file__).parent / "model" / "scene_linear_4310.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)

initialize_scene(model, data, with_paddle=False)
recorders = initialize_camera(model, Path(__file__).parent / "recordings")

try:
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()

            ball_pos = data.body("ball").xpos
            ball_vel = data.joint("ball_free").qvel[:3]
            
            if step % 100 == 0:
                print(f"t={data.time:.2f}  ball pos={ball_pos}  vel={ball_vel}")
            step += 1

            mujoco.mj_step(model, data)
            for recorder in recorders:
                recorder.capture(data)
            viewer.sync()

            # Sleep so the sim runs in real time instead of as fast as possible.
            time_left = model.opt.timestep - (time.time() - step_start)
            if time_left > 0:
                time.sleep(time_left)
finally:
    for recorder in recorders:
        recorder.close()
