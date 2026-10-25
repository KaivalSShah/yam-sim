"""Environment setup: reset the arm to home, place the paddle in the gripper, set up cameras."""

from datetime import datetime

import mujoco
import numpy as np

from cameras import VideoRecorder


def initialize_scene(model, data, with_paddle=True):
    """Reset to the home keyframe. With `with_paddle`, place the paddle in the gripper,
    blade facing up; otherwise park it on the floor out of the way."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    if not with_paddle:
        data.joint("paddle_free").qpos[:3] = [-0.5, 0.5, 0.02]
        data.eq_active[model.equality("hold").id] = 0
        return

    # Open the fingers just wider than the handle so it spawns between the pads.
    handle_half_width, _, handle_half_len = model.geom("handle").size
    grip_open = handle_half_width + 0.002
    data.joint("left_finger").qpos[0] = grip_open
    data.joint("right_finger").qpos[0] = grip_open
    data.ctrl[model.actuator("gripper").id] = grip_open
    mujoco.mj_forward(model, data)

    # Orient the paddle from the grasp site between the pads:
    # handle along the fingers, pads on its narrow faces, blade facing up.
    site = data.site("grasp_site")
    rot = site.xmat.reshape(3, 3) @ np.array([[0.0, -1.0, 0.0],
                                              [1.0,  0.0, 0.0],
                                              [0.0,  0.0, 1.0]])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())

    # Grip 2 cm from the handle's end so the rest of it sticks out past the fingertips.
    pos = site.xpos + rot[:, 2] * (handle_half_len - 0.02)

    paddle = data.joint("paddle_free")
    paddle.qpos[:3] = pos
    paddle.qpos[3:] = quat
    paddle.qvel[:] = 0.0

    # Squeeze and let the contacts settle.
    data.ctrl[model.actuator("gripper").id] = -0.005
    for _ in range(20):
        mujoco.mj_step(model, data)

    # Weld the paddle to the wrist at its settled pose so it cannot creep out.
    wrist, paddle_body = data.body("link_6"), data.body("paddle")
    wrist_quat_inv = np.empty(4)
    mujoco.mju_negQuat(wrist_quat_inv, wrist.xquat)
    rel_quat = np.empty(4)
    mujoco.mju_mulQuat(rel_quat, wrist_quat_inv, paddle_body.xquat)
    rel_pos = wrist.xmat.reshape(3, 3).T @ (paddle_body.xpos - wrist.xpos)

    eq = model.equality("hold").id
    model.eq_data[eq, 0:3] = 0.0
    model.eq_data[eq, 3:6] = rel_pos
    model.eq_data[eq, 6:10] = rel_quat
    model.eq_data[eq, 10] = 1.0
    data.eq_active[eq] = 1


def initialize_camera(model, recordings_dir, fps=30):
    """Create a video recorder per camera, all writing into recordings_dir/<timestamp>/."""
    run_dir = recordings_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True)
    return [
        VideoRecorder(model, "overhead_cam", run_dir / "overhead_cam.mp4", fps=fps),
        VideoRecorder(model, "wrist_cam", run_dir / "wrist_cam.mp4", fps=fps, width=640, height=360),
    ]
