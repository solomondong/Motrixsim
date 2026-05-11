#!/usr/bin/env python3
"""Record LeRobot dataset for the bottle pick-and-place task.

Runs multiple episodes of the bottle grasp demo, recording robot state
and head camera images at a fixed control FPS.  Successful episodes are
saved to the LeRobot dataset; failed episodes are discarded.

Usage:
    DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/record_bottle_demo.py
    DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/record_bottle_demo.py --headless --episode-count 100
"""

import argparse
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
import time

import numpy as np

from motrixsim import SceneData, load_model, step
from motrixsim.render import Color, RenderApp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shelves_demo.controller import RobotController, RobotControllerError, RobotPhase, RobotStatus
from shelves_demo.lerobot_data import FrameData, LerobotData


# ---------------------------------------------------------------------------
# Constants (from grasp_bottle_demo.py)
# ---------------------------------------------------------------------------
STARTUP_STEPS = 400
POST_ACTION_SETTLE_STEPS = 40
GRIPPER_SETTLE_STEPS = 80
VERIFY_LIFT_Z = 0.05
VERIFY_SUCCESS_DELTA_Z = 0.005

GRASP_OFFSET_X = 0.42
GRASP_OFFSET_Y = 1.0
SAFE_APPROACH_OFFSET_Y = 0.5
GRASP_OFFSET_Z = 0.1
PLACE_CLEARANCE_Z = 0.10
SHELF_FRONT_Y = 0.30
TABLE_SURFACE_Z = 0.816
TABLE_FRONT_EDGE_OFFSET_Y = -0.25
TABLE_DEPTH = 0.50
TABLE_TARGET_FRACTION = 0.25
ARM_RETREAT_OFFSET_Y = 0.10
PREGRASP_EXTRA_Y = 0.10

SHELF_HEADING = -0.5 * np.pi
TABLE_HEADING = 0.5 * np.pi

FRONT_ROW_BOTTLES = (
    "B20_001_bottle_base2",
    "B21_001_bottle_base2",
    "B22_001_bottle_base2",
    "B23_001_bottle_base2",
    "B28_001_bottle_base3",
    "B29_001_bottle_base3",
    "B30_001_bottle_base3",
    "B31_001_bottle_base3",
    "B36_001_bottle_base4",
    "B37_001_bottle_base4",
    "B38_001_bottle_base4",
    "B39_001_bottle_base4",
    "B44_001_bottle_base5",
    "B45_001_bottle_base5",
    "B46_001_bottle_base5",
    "B47_001_bottle_base5",
)

TASK_PROMPT = "Pick up the bottle from the shelf and place it on the table"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _yaw_from_quat(quat: np.ndarray) -> float:
    qx, qy, qz, qw = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _heading_from_body_yaw(body_yaw: float) -> float:
    return float(np.arctan2(np.sin(body_yaw - 0.5 * np.pi), np.cos(body_yaw - 0.5 * np.pi)))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Episode state machine
# ---------------------------------------------------------------------------
class DemoStage(str, Enum):
    SELECT_TARGET = "select_target"
    MOVE_TO_SAFE_GRASP = "move_to_safe_grasp"
    MOVE_ARM_TO_PREGRASP = "move_arm_to_pregrasp"
    APPROACH_SHELF = "approach_shelf"
    MOVE_ARM_TO_GRASP = "move_arm_to_grasp"
    CLOSE_GRIPPER = "close_gripper"
    VERIFY_GRASP_LIFT = "verify_grasp_lift"
    RETREAT_FROM_SHELF = "retreat_from_shelf"
    RETURN_TO_START = "return_to_start"
    MOVE_ARM_TO_PLACE = "move_arm_to_place"
    MOVE_TO_TABLE = "move_to_table"
    OPEN_GRIPPER = "open_gripper"
    RETREAT_ARM = "retreat_arm"
    RETREAT_FROM_TABLE = "retreat_from_table"
    RESET_ROBOT = "reset_robot"
    DONE = "done"


@dataclass
class BottleTarget:
    name: str
    body: object


@dataclass
class MotionMarkers:
    safe_base_target: np.ndarray | None = None
    approach_base_target: np.ndarray | None = None
    table_base_target: np.ndarray | None = None
    pregrasp_pose: np.ndarray | None = None
    grasp_pose: np.ndarray | None = None
    verify_pose: np.ndarray | None = None
    place_pose: np.ndarray | None = None
    retreat_pose: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Async capture task
# ---------------------------------------------------------------------------
@dataclass
class RecordTask:
    """Holds async camera captures along with state/action for one frame."""

    capture: object        # mx.render.CaptureTask (head camera)
    left_arm_capture: object   # mx.render.CaptureTask (left arm camera)
    right_arm_capture: object  # mx.render.CaptureTask (right arm camera)
    state: np.ndarray      # (20,)
    actions: np.ndarray    # (20,)
    task: str

    def is_ready(self) -> bool:
        pending = ("pending",)
        return (self.capture.state not in pending
                and self.left_arm_capture.state not in pending
                and self.right_arm_capture.state not in pending)

    def is_success(self) -> bool:
        success = ("success", "done")
        return (self.capture.state in success
                and self.left_arm_capture.state in success
                and self.right_arm_capture.state in success)

    def to_frame(self) -> FrameData:
        return FrameData(
            image=self.capture.take_image().pixels,
            left_arm_image=self.left_arm_capture.take_image().pixels,
            right_arm_image=self.right_arm_capture.take_image().pixels,
            state=self.state,
            actions=self.actions,
            task=self.task,
        )


# ---------------------------------------------------------------------------
# Robot state helper
# ---------------------------------------------------------------------------
def get_robot_state(model, data, controller: RobotController) -> np.ndarray:
    """Construct the 20-D state vector from current simulation data."""
    left_arm = np.array(controller._current_arm_qpos(data, "left"), dtype=np.float32)
    right_arm = np.array(controller._current_arm_qpos(data, "right"), dtype=np.float32)

    # Gripper joint position (0 = closed, 1 = open)
    left_gripper = float(model.get_joint("rb_6/ltool_Left_1_Joint").get_dof_pos(data)[0])
    right_gripper = float(model.get_joint("rb_6/rtool_Left_1_Joint").get_dof_pos(data)[0])

    lifter = float(controller.get_lifter_position(data))

    base_pose = controller.get_base_pose(data)
    base_x = float(base_pose[0])
    base_y = float(base_pose[1])
    base_heading = _heading_from_body_yaw(_yaw_from_quat(base_pose[3:7]))

    return np.array(
        [
            *left_arm,        # 0:7
            *right_arm,       # 7:14
            left_gripper,     # 14
            right_gripper,    # 15
            lifter,           # 16
            base_x,           # 17
            base_y,           # 18
            base_heading,     # 19
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Recording runner
# ---------------------------------------------------------------------------
class RecordBottleRunner:
    """Orchestrates multi-episode recording of the bottle pick-and-place task."""

    def __init__(
        self,
        model_path: str = "xmls/bottle_demo.xml",
        fps: int = 10,
        episode_count: int = 50,
        save_to: str = "./datasets/bottle_pick",
        headless: bool = False,
        debug: bool = False,
    ):
        self.model_path = model_path
        self.fps = fps
        self.ctrl_dt = 1.0 / fps
        self.episode_count = episode_count
        self.save_to = save_to
        self.headless = headless
        self.debug = debug

    # ---- model / renderer / camera setup ----

    def _find_camera_by_name(self, model, name: str) -> object:
        """Find a camera by name in the model."""
        for cam in model.cameras:
            if name in cam.name:
                return cam
        raise RuntimeError(f"camera '{name}' not found in model cameras")

    # ---- episode state machine (adapted from GraspBottleDemo) ----

    def _select_target(self, model, data, controller, rng) -> tuple[BottleTarget, str, MotionMarkers, np.ndarray | None]:
        """Select a random bottle and compute base targets. Returns (target, active_arm, markers, safe_base_target)."""
        available: list[BottleTarget] = []
        for name in FRONT_ROW_BOTTLES:
            body = model.get_body(name)
            if body is not None:
                available.append(BottleTarget(name=name, body=body))
        if not available:
            raise RuntimeError("no front-row bottle bodies found in model")

        index = int(rng.integers(0, len(available)))
        target = available[index]
        target_pose = np.array(target.body.get_pose(data), dtype=np.float64)

        active_arm = "right" if target_pose[0] < 0.0 else "left"
        controller.open_gripper(data, active_arm)

        x_offset = GRASP_OFFSET_X if active_arm == "right" else -GRASP_OFFSET_X
        safe_base_target = np.array(
            [target_pose[0] + x_offset, SHELF_FRONT_Y + GRASP_OFFSET_Y, SHELF_HEADING],
            dtype=np.float64,
        )
        approach_base_target = safe_base_target.copy()
        approach_base_target[1] = SHELF_FRONT_Y + SAFE_APPROACH_OFFSET_Y

        markers = MotionMarkers(
            safe_base_target=safe_base_target,
            approach_base_target=approach_base_target,
        )

        print(f"  target: {target.name}  arm: {active_arm}  safe_base: ({safe_base_target[0]:.2f}, {safe_base_target[1]:.2f})")
        return target, active_arm, markers, safe_base_target

    # ---- main recording loop ----

    def run(self):
        # Load model once
        model = load_model(self.model_path)

        # Setup head camera render target
        head_cam_model = self._find_camera_by_name(model, "vr_camera")
        head_cam_model.set_render_target("image", 1280, 800)

        # Setup left arm camera render target
        left_arm_cam_model = self._find_camera_by_name(model, "left_arm_camera")
        left_arm_cam_model.set_render_target("image", 1280, 800)

        # Setup right arm camera render target
        right_arm_cam_model = self._find_camera_by_name(model, "right_arm_camera")
        right_arm_cam_model.set_render_target("image", 1280, 800)

        # Table reference (constant across episodes)
        table_body = model.get_body("Tab_0/table")
        if table_body is None:
            raise RuntimeError("Tab_0/table body not found")

        sim_dt = float(model.options.timestep)
        num_sim_steps_per_record = max(1, round(self.ctrl_dt / sim_dt))

        rng = np.random.default_rng()

        print(f"\n{'='*60}")
        print(f"Recording {self.episode_count} successful episodes at {self.fps} FPS")
        print(f"Saving to: {self.save_to}")
        print(f"Physics dt={sim_dt:.5f}s  ctrl_dt={self.ctrl_dt:.3f}s  steps/record={num_sim_steps_per_record}")
        print(f"{'='*60}\n")

        with RenderApp(headless=self.headless) as renderer:
            renderer.launch(model)
            head_camera = renderer.get_camera(head_cam_model.index)
            left_arm_camera = renderer.get_camera(left_arm_cam_model.index)
            right_arm_camera = renderer.get_camera(right_arm_cam_model.index)
            data = SceneData(model)

            # Initialize dataset
            dataset = LerobotData(fps=self.fps, save_to=self.save_to)

            success_count = 0
            episode_idx = 0
            max_attempts = self.episode_count * 10  # safety limit

            while success_count < self.episode_count and episode_idx < max_attempts:
                episode_result = self._run_episode(
                    model, data, controller_cls=RobotController,
                    renderer=renderer, head_camera=head_camera,
                    left_arm_camera=left_arm_camera,
                    right_arm_camera=right_arm_camera,
                    dataset=dataset, table_body=table_body,
                    rng=rng, num_sim_steps=num_sim_steps_per_record,
                )

                if episode_result == "success":
                    # Only save if we actually recorded frames
                    if dataset._episode_frames > 0:
                        frame_count = dataset._episode_frames
                        success_count += 1
                        dataset.save_episode()
                        print(f"  Episode {episode_idx} SAVED ({frame_count} frames)  ({success_count}/{self.episode_count})")
                    else:
                        dataset.drop_episode()
                        print(f"  Episode {episode_idx} succeeded but 0 frames recorded - dropped")
                else:
                    dataset.drop_episode()
                    print(f"  Episode {episode_idx} FAILED - dropped")

                episode_idx += 1

        print(f"\nRecording complete: {success_count} successful episodes saved to {self.save_to}")

    # ---- single episode ----

    def _run_episode(
        self,
        model,
        data,
        controller_cls,
        renderer,
        head_camera,
        left_arm_camera,
        right_arm_camera,
        dataset: LerobotData,
        table_body,
        rng,
        num_sim_steps: int,
    ) -> str:
        """Run one episode and return 'success' or 'failed'."""

        try:
            return self._run_episode_inner(
                model, data, controller_cls, renderer, head_camera,
                left_arm_camera, right_arm_camera,
                dataset, table_body, rng, num_sim_steps,
            )
        except RobotControllerError as exc:
            print(f"  Controller error: {exc}")
            return "failed"

    def _run_episode_inner(
        self,
        model,
        data,
        controller_cls,
        renderer,
        head_camera,
        left_arm_camera,
        right_arm_camera,
        dataset: LerobotData,
        table_body,
        rng,
        num_sim_steps: int,
    ) -> str:

        # ---- reset ----
        data.reset(model)
        controller = controller_cls(model)
        controller.initialize(data)
        for _ in range(STARTUP_STEPS):
            step(model, data)

        base_pose = controller.get_base_pose(data)
        start_heading = _heading_from_body_yaw(_yaw_from_quat(base_pose[3:7]))
        start_point = (float(base_pose[0]), float(base_pose[1]), start_heading)

        table_pose = np.array(table_body.get_pose(data), dtype=np.float64)
        table_target_x = float(table_pose[0])
        table_target_y = float(table_pose[1] + TABLE_FRONT_EDGE_OFFSET_Y + TABLE_DEPTH * TABLE_TARGET_FRACTION)

        # ---- select target ----
        target, active_arm, markers, safe_base_target = self._select_target(
            model, data, controller, rng,
        )

        # Start moving base to safe grasp position
        controller.move_base_world(
            data,
            x=float(safe_base_target[0]),
            y=float(safe_base_target[1]),
            heading=float(safe_base_target[2]),
            drive_direction="forward",
        )

        # ---- episode state ----
        stage = DemoStage.MOVE_TO_SAFE_GRASP
        active_motion_label: str | None = "move_to_safe_grasp"
        wait_steps = 0
        grasp_verify_start_z: float | None = None
        verify_lifter_target_z: float | None = None
        records: deque[RecordTask] = deque()

        def _get_target_pose():
            return np.array(target.body.get_pose(data), dtype=np.float64)

        # ---- advance helpers (state machine transitions) ----

        def _move_arm_to_pregrasp():
            nonlocal stage, active_motion_label
            eef_pose = controller.get_end_effector_pose(data, active_arm).copy()
            tp = _get_target_pose()
            pregrasp_pose = eef_pose.copy()
            pregrasp_pose[0] = float(tp[0])
            pregrasp_pose[1] = float(tp[1] + (GRASP_OFFSET_Y - SAFE_APPROACH_OFFSET_Y + PREGRASP_EXTRA_Y))
            pregrasp_pose[2] = float(tp[2] + GRASP_OFFSET_Z)
            markers.pregrasp_pose = pregrasp_pose.copy()
            controller.move_single_arm(data, arm=active_arm, target_pose=pregrasp_pose, constrain_orientation=False)
            active_motion_label = "move_arm_to_pregrasp"
            stage = DemoStage.MOVE_ARM_TO_PREGRASP

        def _move_to_approach():
            nonlocal stage, active_motion_label
            t = markers.approach_base_target
            controller.move_base_world(data, x=float(t[0]), y=float(t[1]), heading=float(t[2]), drive_direction="forward")
            active_motion_label = "approach_shelf"
            stage = DemoStage.APPROACH_SHELF

        def _move_arm_to_grasp():
            nonlocal stage, active_motion_label
            tp = _get_target_pose()
            eef_pose = controller.get_end_effector_pose(data, active_arm).copy()
            grasp_pose = eef_pose.copy()
            grasp_pose[:3] = [tp[0], tp[1], tp[2] + GRASP_OFFSET_Z]
            markers.grasp_pose = grasp_pose.copy()
            controller.move_single_arm(data, arm=active_arm, target_pose=grasp_pose, constrain_orientation=False)
            active_motion_label = "move_arm_to_grasp"
            stage = DemoStage.MOVE_ARM_TO_GRASP

        def _close_gripper():
            nonlocal stage, wait_steps, grasp_verify_start_z
            controller.close_gripper(data, active_arm)
            grasp_verify_start_z = float(_get_target_pose()[2])
            wait_steps = GRIPPER_SETTLE_STEPS
            stage = DemoStage.CLOSE_GRIPPER

        def _verify_grasp_lift():
            nonlocal stage, active_motion_label, verify_lifter_target_z
            current_lifter_z = controller.get_lifter_position(data)
            verify_lifter_target_z = current_lifter_z + VERIFY_LIFT_Z
            controller.move_lifter_to(data, verify_lifter_target_z)
            active_motion_label = "verify_grasp_lift"
            stage = DemoStage.VERIFY_GRASP_LIFT

        def _finish_grasp_verification() -> str | None:
            """Returns 'failed' if grasp check failed, None to continue."""
            if grasp_verify_start_z is None:
                return "failed"
            current_bottle_z = float(_get_target_pose()[2])
            delta_z = current_bottle_z - grasp_verify_start_z
            if delta_z >= VERIFY_SUCCESS_DELTA_Z:
                _retreat_from_shelf()
                return None
            # Grasp failed
            return "failed"

        def _retreat_from_shelf():
            nonlocal stage, active_motion_label
            t = markers.safe_base_target
            controller.move_base_world(data, x=float(t[0]), y=float(t[1]), heading=float(t[2]), drive_direction="backward")
            active_motion_label = "retreat_from_shelf"
            stage = DemoStage.RETREAT_FROM_SHELF

        def _return_to_start():
            nonlocal stage, active_motion_label
            controller.move_base_world(data, x=start_point[0], y=start_point[1], heading=start_point[2], drive_direction="forward")
            active_motion_label = "return_to_start"
            stage = DemoStage.RETURN_TO_START

        def _move_arm_to_place():
            nonlocal stage, active_motion_label
            eef_pose = controller.get_end_effector_pose(data, active_arm).copy()
            place_pose = eef_pose.copy()
            place_pose[2] = TABLE_SURFACE_Z + GRASP_OFFSET_Z + PLACE_CLEARANCE_Z
            markers.place_pose = place_pose.copy()
            controller.move_single_arm(data, arm=active_arm, target_pose=place_pose, constrain_orientation=False)
            active_motion_label = "move_arm_to_place"
            stage = DemoStage.MOVE_ARM_TO_PLACE

        def _move_to_table():
            nonlocal stage, active_motion_label
            place_x_offset = GRASP_OFFSET_X if active_arm == "left" else -GRASP_OFFSET_X
            table_base_target = np.array(
                [table_target_x + place_x_offset, table_target_y, TABLE_HEADING],
                dtype=np.float64,
            )
            markers.table_base_target = table_base_target
            controller.move_base_world(data, x=float(table_base_target[0]), y=float(table_base_target[1]), heading=float(table_base_target[2]), drive_direction="forward")
            active_motion_label = "move_to_table"
            stage = DemoStage.MOVE_TO_TABLE

        def _open_gripper():
            nonlocal stage, wait_steps
            controller.open_gripper(data, active_arm)
            wait_steps = GRIPPER_SETTLE_STEPS
            stage = DemoStage.OPEN_GRIPPER

        def _retreat_arm():
            nonlocal stage, active_motion_label
            retreat_pose = controller.get_end_effector_pose(data, active_arm).copy()
            retreat_pose[1] -= ARM_RETREAT_OFFSET_Y
            markers.retreat_pose = retreat_pose.copy()
            controller.move_single_arm(data, arm=active_arm, target_pose=retreat_pose, constrain_orientation=False)
            active_motion_label = "retreat_arm"
            stage = DemoStage.RETREAT_ARM

        def _retreat_from_table():
            nonlocal stage, active_motion_label
            controller.move_base_world(data, x=start_point[0], y=start_point[1], heading=TABLE_HEADING, drive_direction="backward")
            active_motion_label = "retreat_from_table"
            stage = DemoStage.RETREAT_FROM_TABLE

        def _reset_robot():
            nonlocal stage, wait_steps
            controller.initialize(data)
            wait_steps = POST_ACTION_SETTLE_STEPS
            stage = DemoStage.RESET_ROBOT

        def _advance_after_motion_done():
            nonlocal active_motion_label
            motion = active_motion_label
            active_motion_label = None
            dispatch = {
                "move_to_safe_grasp": _move_arm_to_pregrasp,
                "move_arm_to_pregrasp": _move_to_approach,
                "approach_shelf": _move_arm_to_grasp,
                "move_arm_to_grasp": _close_gripper,
                "verify_grasp_lift": lambda: _finish_grasp_verification(),
                "retreat_from_shelf": _return_to_start,
                "return_to_start": _move_arm_to_place,
                "move_arm_to_place": _move_to_table,
                "retreat_arm": _retreat_from_table,
                "retreat_from_table": _reset_robot,
            }
            handler = dispatch.get(motion)
            if handler is None:
                raise RuntimeError(f"unexpected completed motion label: {motion}")
            result = handler()
            # _finish_grasp_verification can return 'failed'
            return result

        def _advance_after_wait() -> str | None:
            nonlocal stage
            if stage == DemoStage.CLOSE_GRIPPER:
                _verify_grasp_lift()
                return None
            if stage == DemoStage.OPEN_GRIPPER:
                _retreat_arm()
                return None
            if stage == DemoStage.RESET_ROBOT:
                stage = DemoStage.DONE
                return "success"
            raise RuntimeError(f"unexpected wait stage: {stage.value}")

        # ---- episode main loop ----

        episode_done = False
        episode_result = "failed"
        step_count = 0
        max_episode_steps = 40000
        last_logged_stage = None

        while True:
            t0 = time.monotonic()

            # Log stage transitions
            if stage.value != last_logged_stage:
                print(f"    [{step_count}] stage -> {stage.value}  motion={active_motion_label}  wait={wait_steps}")
                last_logged_stage = stage.value

            # --- physics + state machine for N sub-steps ---
            if not episode_done:
                for _ in range(num_sim_steps):
                    step_count += 1

                    # Episode timeout
                    if step_count > max_episode_steps:
                        print(f"    Episode timeout after {step_count} steps at stage={stage.value}")
                        episode_done = True
                        episode_result = "failed"
                        break

                    status = controller.update(data)

                    if status.phase == RobotPhase.ERROR:
                        print(f"  Controller error: {status.error}")
                        episode_done = True
                        episode_result = "failed"
                        break

                    # Per-step stage checks
                    if stage == DemoStage.MOVE_TO_TABLE:
                        tp = _get_target_pose()
                        if table_target_y is not None and float(tp[1]) >= table_target_y:
                            controller.stop(data)
                            _open_gripper()

                    elif stage == DemoStage.VERIFY_GRASP_LIFT:
                        if verify_lifter_target_z is not None and controller.is_lifter_at(data, verify_lifter_target_z):
                            result = _finish_grasp_verification()
                            if result == "failed":
                                episode_done = True
                                episode_result = "failed"
                                break

                    elif wait_steps > 0:
                        wait_steps -= 1
                        if wait_steps == 0:
                            result = _advance_after_wait()
                            if result == "success":
                                episode_done = True
                                episode_result = "success"
                                break

                    elif active_motion_label is not None and status.phase == RobotPhase.DONE:
                        result = _advance_after_motion_done()
                        if result == "failed":
                            episode_done = True
                            episode_result = "failed"
                            break

                    step(model, data)

            # --- capture → sync → process (matches motrix-lerobot reference) ---
            # Only add new records while episode is still running
            if not episode_done:
                cap = head_camera.capture()
                left_cap = left_arm_camera.capture()
                right_cap = right_arm_camera.capture()
                state = get_robot_state(model, data, controller)
                action = state.copy()  # position-control: action ≈ current state
                records.append(RecordTask(
                    capture=cap, left_arm_capture=left_cap, right_arm_capture=right_cap,
                    state=state, actions=action, task=TASK_PROMPT,
                ))

            # Sync triggers render pipeline which fulfills pending captures
            if self.debug and not episode_done:
                self._draw_debug_gizmos(renderer, data, controller, target, active_arm, markers)
            renderer.sync(data)

            # --- process ready records ---
            while records and records[0].is_ready():
                rec = records.popleft()
                if rec.is_success():
                    dataset.add_frame(rec.to_frame())
                else:
                    print(f"    [WARN] capture failed: state={rec.capture.state}")

            # --- timing ---
            passed = time.monotonic() - t0
            sleep_time = self.ctrl_dt - passed
            if not self.headless and sleep_time > 0:
                time.sleep(sleep_time)

            # Episode done AND all records flushed → exit loop
            if episode_done and len(records) == 0:
                print(f"    episode_done={episode_result}  frames_recorded={dataset._episode_frames}  pending_records=0")
                break

        return episode_result

    # ---- debug gizmos ----

    def _draw_debug_gizmos(self, renderer, data, controller, target, active_arm, markers):
        gizmos = renderer.gizmos
        base_pose = controller.get_base_pose(data)
        base_pos = np.array(base_pose[:3], dtype=np.float64)

        if target is not None:
            target_pose = np.array(target.body.get_pose(data), dtype=np.float64)
            gizmos.draw_sphere(0.045, target_pose[:3], color=Color.rgb(1.0, 0.0, 0.0))
            gizmos.draw_sphere(0.022, target_pose[:3], color=Color.rgb(1.0, 1.0, 0.0))
            gizmos.draw_axes(target_pose[:3], target_pose[3:7], length=0.16)

        if active_arm is not None:
            active_eef_pose = controller.get_end_effector_pose(data, active_arm)
            arm_color = Color.rgb(1.0, 0.4, 0.0) if active_arm == "left" else Color.rgb(0.2, 0.7, 1.0)
            gizmos.draw_sphere(0.04, active_eef_pose[:3], color=arm_color)
            gizmos.draw_axes(active_eef_pose[:3], active_eef_pose[3:7], length=0.14)

        for marker, color in (
            (markers.safe_base_target, Color.rgb(1.0, 0.0, 0.0)),
            (markers.approach_base_target, Color.rgb(1.0, 0.6, 0.0)),
            (markers.table_base_target, Color.rgb(0.6, 0.2, 1.0)),
        ):
            if marker is not None:
                marker_pos = np.array([marker[0], marker[1], base_pos[2]], dtype=np.float64)
                marker_quat = _quat_from_yaw(float(marker[2]))
                gizmos.draw_sphere(0.035, marker_pos, color=color)
                gizmos.draw_axes(marker_pos, marker_quat, length=0.16)
                gizmos.draw_line(base_pos, marker_pos, color=Color.rgb(1.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Record LeRobot dataset for bottle pick-and-place task")
    parser.add_argument("--model", default="xmls/bottle_demo.xml", help="scene XML path")
    parser.add_argument("--fps", type=int, default=15, help="recording control FPS")
    parser.add_argument("--episode-count", type=int, default=50, help="number of successful episodes to record")
    parser.add_argument("--save-to", default="./datasets/bottle_pick", help="dataset output directory")
    parser.add_argument("--headless", action="store_true", help="run without GUI window")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--debug", action="store_true", help="draw debug gizmos")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    runner = RecordBottleRunner(
        model_path=args.model,
        fps=args.fps,
        episode_count=args.episode_count,
        save_to=args.save_to,
        headless=args.headless,
        debug=args.debug,
    )
    runner.run()


if __name__ == "__main__":
    main()

# Example commands:
#   DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/record_bottle_demo.py --headless --episode-count 50
#   DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/record_bottle_demo.py --debug --episode-count 10
