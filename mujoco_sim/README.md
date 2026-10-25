# mujoco_sim

A bare-bones MuJoCo simulation of one I2RT YAM arm with the `linear_4310` gripper
and the wrist-mounted Intel RealSense D405 camera, holding a ping-pong paddle.

The robot model in `model/` is the `i2rt_yam` model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/i2rt_yam)
(MIT license, see `model/LICENSE`), copied as-is.

## Setup

```bash
cd mujoco_sim
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
```

## Run

```bash
# Interactive viewer. On macOS you must use mjpython instead of python.
# Every run records the overhead camera to recordings/<timestamp>/overhead_cam.mp4
# (written when you close the window).
mjpython sim.py
```

## Files

| File | What it is |
|------|------------|
| `model/scene_linear_4310.xml` | Scene: floor, skybox, lights. Includes the robot file below. |
| `model/yam_linear_4310.xml` | The robot: links, joints, geoms, gripper, camera, actuators, home keyframe. |
| `model/assets/` | STL meshes. |
| `sim.py` | Loads the scene, initializes the environment, steps physics, shows the viewer, records the overhead camera. |
| `recordings/` | One timestamped folder per run with its `overhead_cam.mp4`. |
| `initialize_environment.py` | Resets to home and places the paddle in the gripper (`with_paddle=False` parks it). |
| `cameras.py` | `CameraFeed`: renders any model camera (`overhead_cam`, `wrist_cam`) to a numpy RGB array. `VideoRecorder`: writes a camera to MP4 during a run. |

## Things to try

- The viewer starts with its side panels hidden. Press `Shift+Tab` to show the right panel
  and drag the **Control** sliders (`Tab` shows the left panel).
  Each slider is one `<position>` actuator in `yam_linear_4310.xml`. The last one
  (`gripper`) opens and closes the fingers.
- Switch the viewer camera to the overhead or wrist camera: right panel, **Rendering** > **Camera**,
  or press `[` / `]` to cycle cameras. The overhead camera is a fixed `<camera>` in the scene file.
- Press `Ctrl+Backspace` to reset to the home keyframe. Press `Space` to pause.
- Set `data.ctrl[:]` in `sim.py` to command joint targets from code, e.g.
  `data.ctrl[0] = 1.0` to turn joint1. Indices match the `<actuator>` order.
- Drop a box in the scene by adding a free body to `scene_linear_4310.xml`:
  ```xml
  <body pos="0.4 0 0.05"><freejoint/><geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"/></body>
  ```
