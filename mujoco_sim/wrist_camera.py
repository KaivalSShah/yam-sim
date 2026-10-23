"""Render one frame from the wrist camera to wrist_camera.png (no viewer needed)."""

from pathlib import Path

import mujoco
from PIL import Image

MODEL = Path(__file__).parent / "model" / "scene_linear_4310.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=360, width=640)
renderer.update_scene(data, camera="wrist_cam")
Image.fromarray(renderer.render()).save("wrist_camera.png")
print("saved wrist_camera.png")
