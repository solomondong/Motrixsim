"""LeRobot dataset recording utilities for the shelves demo.

Provides FrameData and LerobotData classes for recording robot episodes
in the LeRobot HDF5/Parquet format, compatible with lerobot training pipelines.

State/Action dimension layout (20D):
    [0:7]   left arm joint positions  (l_joint1..l_joint7)
    [7:14]  right arm joint positions (r_joint1..r_joint7)
    [14]    left gripper state        (0.0 closed, 1.0 open)
    [15]    right gripper state       (0.0 closed, 1.0 open)
    [16]    lifter position           (z, metres)
    [17]    base x position           (metres)
    [18]    base y position           (metres)
    [19]    base heading angle        (radians)
"""

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None
    _LEROBOT_INSTALL_HINT = (
        "lerobot is required for dataset recording. "
        "Install it with:  conda run -n MotrixLab pip install lerobot"
    )


STATE_DIM = 20
ACTION_DIM = 20
IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1280
IMAGE_CHANNELS = 3


@dataclass
class FrameData:
    """Single timestep data for LeRobot recording."""

    image: np.ndarray   # (480, 640, 3) head camera RGB
    left_arm_image: np.ndarray   # (480, 640, 3) left arm camera RGB
    right_arm_image: np.ndarray  # (480, 640, 3) right arm camera RGB
    state: np.ndarray   # (20,) robot proprioceptive state
    actions: np.ndarray  # (20,) robot target state (same layout as state)
    task: str           # task instruction prompt


class LerobotData:
    """Wrapper around LeRobotDataset for the shelves bottle pick-and-place task.

    Handles dataset creation, frame addition, and episode save/drop with
    video encoding via AV1 codec.
    """

    dataset: LeRobotDataset
    _num_frames: int = 0
    _episode_frames: int = 0  # frames in current episode buffer

    def __init__(
        self,
        fps: int = 10,
        save_to: str = "./datasets/bottle_pick",
    ):
        if LeRobotDataset is None:
            raise ImportError(_LEROBOT_INSTALL_HINT)

        dataset_path = Path(save_to)
        if dataset_path.exists():
            print(f"Removing existing dataset directory: {dataset_path}")
            shutil.rmtree(dataset_path)
        print(f"lerobot fps = {fps}")

        video_info = {
            "video.height": IMAGE_HEIGHT,
            "video.width": IMAGE_WIDTH,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30,
            "video.channels": IMAGE_CHANNELS,
            "has_audio": False,
        }

        self.dataset = LeRobotDataset.create(
            repo_id="D-Robotics/ShelvesBottleLerobot",
            fps=fps,
            root=save_to,
            features={
                "observation.image": {
                    "dtype": "video",
                    "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                    "names": ["height", "width", "channel"],
                    "info": video_info,
                },
                "observation.images.left_arm": {
                    "dtype": "video",
                    "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                    "names": ["height", "width", "channel"],
                    "info": video_info,
                },
                "observation.images.right_arm": {
                    "dtype": "video",
                    "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                    "names": ["height", "width", "channel"],
                    "info": video_info,
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (STATE_DIM,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (ACTION_DIM,),
                    "names": ["action"],
                },
            },
            robot_type="realman_wr75s",
            image_writer_threads=16,
        )

    def add_frame(self, frame: FrameData):
        """Add a single timestep frame to the dataset buffer."""
        self.dataset.add_frame(
            {
                "observation.image": frame.image,
                "observation.images.left_arm": frame.left_arm_image,
                "observation.images.right_arm": frame.right_arm_image,
                "observation.state": frame.state,
                "action": frame.actions,
            },
            task=frame.task,
        )
        self._num_frames += 1
        self._episode_frames += 1

    def save_episode(self):
        """Commit the current episode buffer to the dataset."""
        self.dataset.save_episode()
        self._episode_frames = 0

    def drop_episode(self):
        """Discard the current episode buffer (e.g. failed episode)."""
        self.dataset._wait_image_writer()
        self.dataset.clear_episode_buffer()
        self._episode_frames = 0
