"""Offscreen camera feeds. Works with any <camera> in the model, e.g. "overhead_cam" or "wrist_cam"."""

import imageio.v2 as imageio
import mujoco


class CameraFeed:
    """Renders a named model camera to an RGB numpy array of shape (height, width, 3)."""

    def __init__(self, model, camera_name, width=640, height=480, shadows=False):
        self.camera_name = camera_name
        self.shadows = shadows  # shadows cost ~10 ms per frame on macOS; off by default
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def frame(self, data):
        """Return the current view from this camera as a uint8 RGB array."""
        self.renderer.update_scene(data, camera=self.camera_name)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows
        return self.renderer.render()

    def close(self):
        self.renderer.close()

class VideoRecorder:
    """Records a model camera to an MP4 at a fixed rate in *simulation* time.

    Call `capture(data)` once per physics step; it only renders when the next video
    frame is due, so playback runs at 1x sim speed no matter how fast the sim ran.
    """

    def __init__(self, model, camera_name, path, fps=30, width=640, height=480):
        self.feed = CameraFeed(model, camera_name, width, height)
        self.path = str(path)
        self.frame_dt = 1.0 / fps
        self.next_frame_time = None
        self.frames = 0
        self.writer = imageio.get_writer(
            self.path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
            output_params=["-movflags", "frag_keyframe+empty_moov"],
        )

    def capture(self, data):
        if self.next_frame_time is None or data.time < self.next_frame_time - self.frame_dt:
            self.next_frame_time = data.time  # first frame, or the sim was reset
        if data.time >= self.next_frame_time:
            self.writer.append_data(self.feed.frame(data))
            self.frames += 1
            self.next_frame_time += self.frame_dt

    def close(self):
        self.writer.close()
        self.feed.close()
        print(f"saved {self.path} ({self.frames} frames)")
