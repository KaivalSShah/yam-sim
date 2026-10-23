"""Minimal MuJoCo sim of the YAM arm. On macOS run with `mjpython sim.py`."""

import time
import math
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL = Path(__file__).parent / "model" / "scene_linear_4310.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)

# Start from the "home" keyframe defined in the model (sets qpos and ctrl).
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

# Hide the side panels for a clean window. Press Tab / Shift+Tab in the
# viewer to bring the left / right panel back (Control sliders live on the right).
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        time_left = model.opt.timestep - (time.time() - step_start)
        if time_left > 0:
            time.sleep(time_left)