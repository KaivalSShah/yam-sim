"""Minimal MuJoCo sim of the YAM arm. On macOS run with `mjpython sim.py`."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

from initialize_environment import initialize_camera, initialize_scene
from utils import detect_ball_contact

MODEL = Path(__file__).parent / "model" / "scene_linear_4310.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)

initialize_scene(model, data, with_paddle=True)
recorders = initialize_camera(model, Path(__file__).parent / "recordings")

try:
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        was_on_paddle = was_on_floor = False
        while viewer.is_running():
            step_start = time.time()

            # Print only when a contact starts, not on every step it persists.
            on_paddle, on_floor = detect_ball_contact(model, data)
            if on_paddle and not was_on_paddle:
                print(f"t={data.time:.2f}  ball hit the paddle")
            if on_floor and not was_on_floor:
                print(f"t={data.time:.2f}  ball hit the floor")
            was_on_paddle, was_on_floor = on_paddle, on_floor

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
