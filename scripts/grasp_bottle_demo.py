import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys

import numpy as np

from motrixsim import SceneData, load_model, run, step
from motrixsim.render import Color, RenderApp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shelves_demo.controller import RobotController, RobotPhase, RobotStatus


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


def _yaw_from_quat(quat: np.ndarray) -> float:
    qx, qy, qz, qw = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _heading_from_body_yaw(body_yaw: float) -> float:
    return float(np.arctan2(np.sin(body_yaw - 0.5 * np.pi), np.cos(body_yaw - 0.5 * np.pi)))


def _body_yaw_from_heading(heading: float) -> float:
    return float(np.arctan2(np.sin(heading + 0.5 * np.pi), np.cos(heading + 0.5 * np.pi)))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64)


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


class GraspBottleDemo:
    def __init__(self, model_path: str, seed: int | None, debug: bool) -> None:
        self.model_path = model_path
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        self.seed = seed
        self.debug = debug

        # Convert to absolute path to avoid path duplication bug in load_model
        abs_model_path = str(Path(model_path).resolve())
        self.model = load_model(abs_model_path)
        self.data = SceneData(self.model)
        self.controller = RobotController(self.model)
        self.rng = np.random.default_rng(seed)

        self.stage = DemoStage.SELECT_TARGET
        self.last_robot_phase: RobotPhase | None = None
        self.last_robot_detail: str | None = None
        self.wait_steps = 0

        self.start_point: tuple[float, float, float] | None = None
        self.table_target_x: float | None = None
        self.table_target_y: float | None = None
        self.table_body = self.model.get_body("Tab_0/table")

        self.target: BottleTarget | None = None
        self.active_arm: str | None = None
        self.markers = MotionMarkers()
        self.active_motion_label: str | None = None
        self.grasp_verify_start_z: float | None = None
        self.verify_lifter_target_z: float | None = None

    def setup(self) -> None:
        print(f"random seed: {self.seed}")
        self.controller.initialize(self.data)
        for _ in range(STARTUP_STEPS):
            step(self.model, self.data)
        base_pose = self.controller.get_base_pose(self.data)
        start_heading = _heading_from_body_yaw(_yaw_from_quat(base_pose[3:7]))
        self.start_point = (float(base_pose[0]), float(base_pose[1]), start_heading)

        if self.table_body is None:
            raise RuntimeError("failed to resolve table body: Tab_0/table")
        table_pose = np.array(self.table_body.get_pose(self.data), dtype=np.float64)
        self.table_target_x = float(table_pose[0])
        table_front_edge_y = float(table_pose[1] + TABLE_FRONT_EDGE_OFFSET_Y)
        self.table_target_y = float(table_front_edge_y + TABLE_DEPTH * TABLE_TARGET_FRACTION)

    def _log_stage(self, message: str) -> None:
        print(f"[{self.stage.value}] {message}")

    def _report_robot_status(self, status: RobotStatus) -> None:
        if status.phase != self.last_robot_phase or status.detail != self.last_robot_detail:
            detail = f" ({status.detail})" if status.detail else ""
            print(f"controller phase -> {status.phase.value}{detail}")
            self.last_robot_phase = status.phase
            self.last_robot_detail = status.detail

    def _get_target_pose(self) -> np.ndarray:
        if self.target is None:
            raise RuntimeError("target bottle has not been selected")
        return np.array(self.target.body.get_pose(self.data), dtype=np.float64)

    def _select_target(self) -> None:
        available: list[BottleTarget] = []
        for name in FRONT_ROW_BOTTLES:
            body = self.model.get_body(name)
            if body is not None:
                available.append(BottleTarget(name=name, body=body))
        if not available:
            raise RuntimeError("failed to find any front-row bottle bodies in model")

        index = int(self.rng.integers(0, len(available)))
        self.target = available[index]
        target_pose = self._get_target_pose()
        self.active_arm = "right" if target_pose[0] < 0.0 else "left"
        self.controller.open_gripper(self.data, self.active_arm)

        x_offset = GRASP_OFFSET_X if self.active_arm == "right" else -GRASP_OFFSET_X
        safe_base_target = np.array(
            [target_pose[0] + x_offset, SHELF_FRONT_Y + GRASP_OFFSET_Y, SHELF_HEADING],
            dtype=np.float64,
        )
        approach_base_target = safe_base_target.copy()
        approach_base_target[1] = SHELF_FRONT_Y + SAFE_APPROACH_OFFSET_Y

        self.markers.safe_base_target = safe_base_target
        self.markers.approach_base_target = approach_base_target

        print("=" * 60)
        print(f"selected target: {self.target.name}")
        print(f"target xyz:      ({target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f})")
        print(f"active arm:      {self.active_arm}")
        print(f"safe base:       ({safe_base_target[0]:.3f}, {safe_base_target[1]:.3f}, {safe_base_target[2]:.3f})")
        print(f"approach base:   ({approach_base_target[0]:.3f}, {approach_base_target[1]:.3f}, {approach_base_target[2]:.3f})")
        print(f"table target y:  {self.table_target_y:.3f}")
        print("=" * 60)

        self.controller.move_base_world(
            self.data,
            x=float(safe_base_target[0]),
            y=float(safe_base_target[1]),
            heading=float(safe_base_target[2]),
            drive_direction="forward",
        )
        self.active_motion_label = "move_to_safe_grasp"
        self.stage = DemoStage.MOVE_TO_SAFE_GRASP
        self._log_stage("moving base to safe grasp pose")

    def _move_to_approach(self) -> None:
        target = self.markers.approach_base_target
        if target is None:
            raise RuntimeError("approach target not prepared")
        self.controller.move_base_world(
            self.data,
            x=float(target[0]),
            y=float(target[1]),
            heading=float(target[2]),
            drive_direction="forward",
        )
        self.active_motion_label = "approach_shelf"
        self.stage = DemoStage.APPROACH_SHELF
        self._log_stage("approaching shelf")

    def _move_arm_to_pregrasp(self) -> None:
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        target_pose = self._get_target_pose()
        eef_pose = self.controller.get_end_effector_pose(self.data, self.active_arm).copy()
        pregrasp_pose = eef_pose.copy()
        pregrasp_pose[0] = float(target_pose[0])
        pregrasp_pose[1] = float(target_pose[1] + (GRASP_OFFSET_Y - SAFE_APPROACH_OFFSET_Y + PREGRASP_EXTRA_Y))
        pregrasp_pose[2] = float(target_pose[2] + GRASP_OFFSET_Z)
        self.markers.pregrasp_pose = pregrasp_pose.copy()
        self.controller.move_single_arm(self.data, arm=self.active_arm, target_pose=pregrasp_pose, constrain_orientation=False)
        self.active_motion_label = "move_arm_to_pregrasp"
        self.stage = DemoStage.MOVE_ARM_TO_PREGRASP
        self._log_stage("moving arm to pregrasp pose")

    def _move_arm_to_grasp(self) -> None:
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        target_pose = self._get_target_pose()
        eef_pose = self.controller.get_end_effector_pose(self.data, self.active_arm).copy()
        grasp_pose = eef_pose.copy()
        grasp_pose[:3] = np.array(
            [target_pose[0], target_pose[1], target_pose[2] + GRASP_OFFSET_Z],
            dtype=np.float64,
        )
        self.markers.grasp_pose = grasp_pose.copy()
        self.controller.move_single_arm(self.data, arm=self.active_arm, target_pose=grasp_pose, constrain_orientation=False)
        self.active_motion_label = "move_arm_to_grasp"
        self.stage = DemoStage.MOVE_ARM_TO_GRASP
        self._log_stage("moving arm to bottle grasp pose")

    def _close_gripper(self) -> None:
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        self.controller.close_gripper(self.data, self.active_arm)
        self.grasp_verify_start_z = float(self._get_target_pose()[2])
        self.wait_steps = GRIPPER_SETTLE_STEPS
        self.stage = DemoStage.CLOSE_GRIPPER
        self._log_stage("closing gripper")

    def _verify_grasp_lift(self) -> None:
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        current_lifter_z = self.controller.get_lifter_position(self.data)
        target_lifter_z = current_lifter_z + VERIFY_LIFT_Z
        verify_pose = self.controller.get_end_effector_pose(self.data, self.active_arm).copy()
        verify_pose[2] += VERIFY_LIFT_Z
        self.markers.verify_pose = verify_pose.copy()
        self.verify_lifter_target_z = float(target_lifter_z)
        self.controller.move_lifter_to(self.data, target_lifter_z)
        self.active_motion_label = "verify_grasp_lift"
        self.stage = DemoStage.VERIFY_GRASP_LIFT
        self._log_stage("lifting platform to verify grasp success")

    def _finish_grasp_verification(self) -> None:
        if self.grasp_verify_start_z is None:
            raise RuntimeError("grasp verification start height is not recorded")
        current_bottle_z = float(self._get_target_pose()[2])
        delta_z = current_bottle_z - self.grasp_verify_start_z
        print(f"verify bottle delta z: {delta_z:.4f}")
        if delta_z >= VERIFY_SUCCESS_DELTA_Z:
            print("grasp verification: success")
            self._retreat_from_shelf()
            return

        print("grasp verification: failed")
        if self.active_arm is not None:
            self.controller.open_gripper(self.data, self.active_arm)
        raise SystemExit(1)

    def _retreat_from_shelf(self) -> None:
        target = self.markers.safe_base_target
        if target is None:
            raise RuntimeError("safe retreat target not prepared")
        self.controller.move_base_world(
            self.data,
            x=float(target[0]),
            y=float(target[1]),
            heading=float(target[2]),
            drive_direction="backward",
        )
        self.active_motion_label = "retreat_from_shelf"
        self.stage = DemoStage.RETREAT_FROM_SHELF
        self._log_stage("retreating from shelf")

    def _return_to_start(self) -> None:
        if self.start_point is None:
            raise RuntimeError("start point not initialized")
        self.controller.move_base_world(
            self.data,
            x=self.start_point[0],
            y=self.start_point[1],
            heading=self.start_point[2],
            drive_direction="forward",
        )
        self.active_motion_label = "return_to_start"
        self.stage = DemoStage.RETURN_TO_START
        self._log_stage("returning base to start point")

    def _move_arm_to_place(self) -> None:
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        eef_pose = self.controller.get_end_effector_pose(self.data, self.active_arm).copy()
        place_pose = eef_pose.copy()
        place_pose[2] = TABLE_SURFACE_Z + GRASP_OFFSET_Z + PLACE_CLEARANCE_Z
        self.markers.place_pose = place_pose.copy()
        self.controller.move_single_arm(self.data, arm=self.active_arm, target_pose=place_pose, constrain_orientation=False)
        self.active_motion_label = "move_arm_to_place"
        self.stage = DemoStage.MOVE_ARM_TO_PLACE
        self._log_stage("moving arm to place height")

    def _move_to_table(self) -> None:
        if self.start_point is None or self.table_target_x is None or self.table_target_y is None:
            raise RuntimeError("table motion reference is not initialized")
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        place_x_offset = GRASP_OFFSET_X if self.active_arm == "left" else -GRASP_OFFSET_X
        table_base_target = np.array(
            [self.table_target_x + place_x_offset, self.table_target_y, TABLE_HEADING],
            dtype=np.float64,
        )
        self.markers.table_base_target = table_base_target
        self.controller.move_base_world(
            self.data,
            x=float(table_base_target[0]),
            y=float(table_base_target[1]),
            heading=float(table_base_target[2]),
            drive_direction="forward",
        )
        self.active_motion_label = "move_to_table"
        self.stage = DemoStage.MOVE_TO_TABLE
        self._log_stage("driving bottle toward table target line")

    def _open_gripper(self) -> None:
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        self.controller.open_gripper(self.data, self.active_arm)
        self.wait_steps = GRIPPER_SETTLE_STEPS
        self.stage = DemoStage.OPEN_GRIPPER
        self._log_stage("opening gripper to release bottle")

    def _retreat_arm(self) -> None:
        if self.active_arm is None:
            raise RuntimeError("active arm is not set")
        retreat_pose = self.controller.get_end_effector_pose(self.data, self.active_arm).copy()
        retreat_pose[1] -= ARM_RETREAT_OFFSET_Y
        self.markers.retreat_pose = retreat_pose.copy()
        self.controller.move_single_arm(self.data, arm=self.active_arm, target_pose=retreat_pose, constrain_orientation=False)
        self.active_motion_label = "retreat_arm"
        self.stage = DemoStage.RETREAT_ARM
        self._log_stage("retreating arm away from placed bottle")

    def _retreat_from_table(self) -> None:
        if self.start_point is None:
            raise RuntimeError("start point not initialized")
        self.controller.move_base_world(
            self.data,
            x=self.start_point[0],
            y=self.start_point[1],
            heading=TABLE_HEADING,
            drive_direction="backward",
        )
        self.active_motion_label = "retreat_from_table"
        self.stage = DemoStage.RETREAT_FROM_TABLE
        self._log_stage("retreating base from table")

    def _reset_robot(self) -> None:
        self.controller.initialize(self.data)
        self.wait_steps = POST_ACTION_SETTLE_STEPS
        self.stage = DemoStage.RESET_ROBOT
        self._log_stage("resetting robot to standby pose")

    def _advance_after_motion_done(self) -> None:
        motion = self.active_motion_label
        self.active_motion_label = None
        if motion == "move_to_safe_grasp":
            self._move_arm_to_pregrasp()
            return
        if motion == "move_arm_to_pregrasp":
            self._move_to_approach()
            return
        if motion == "approach_shelf":
            self._move_arm_to_grasp()
            return
        if motion == "move_arm_to_grasp":
            self._close_gripper()
            return
        if motion == "verify_grasp_lift":
            self._finish_grasp_verification()
            return
        if motion == "retreat_from_shelf":
            self._return_to_start()
            return
        if motion == "return_to_start":
            self._move_arm_to_place()
            return
        if motion == "move_arm_to_place":
            self._move_to_table()
            return
        if motion == "retreat_arm":
            self._retreat_from_table()
            return
        if motion == "retreat_from_table":
            self._reset_robot()
            return
        raise RuntimeError(f"unexpected completed motion label: {motion}")

    def _advance_after_wait(self) -> None:
        if self.stage == DemoStage.CLOSE_GRIPPER:
            self._verify_grasp_lift()
            return
        if self.stage == DemoStage.OPEN_GRIPPER:
            self._retreat_arm()
            return
        if self.stage == DemoStage.RESET_ROBOT:
            self.stage = DemoStage.DONE
            self._log_stage("task complete")
            raise SystemExit(0)
        raise RuntimeError(f"unexpected wait stage: {self.stage.value}")

    def phys_step(self) -> None:
        status = self.controller.update(self.data)
        self._report_robot_status(status)
        if status.phase == RobotPhase.ERROR:
            raise SystemExit(status.error or "demo entered controller error state")

        if self.stage == DemoStage.SELECT_TARGET:
            self._select_target()
        elif self.stage == DemoStage.MOVE_TO_TABLE:
            target_pose = self._get_target_pose()
            if self.table_target_y is not None and float(target_pose[1]) >= self.table_target_y:
                self.controller.stop(self.data)
                self._open_gripper()
        elif self.stage == DemoStage.VERIFY_GRASP_LIFT:
            if self.grasp_verify_start_z is None:
                raise RuntimeError("grasp verification start height is not recorded")
            if self.verify_lifter_target_z is None:
                raise RuntimeError("grasp verification lifter target is not recorded")
            if self.controller.is_lifter_at(self.data, self.verify_lifter_target_z):
                self._finish_grasp_verification()
        elif self.wait_steps > 0:
            self.wait_steps -= 1
            if self.wait_steps == 0:
                self._advance_after_wait()
        elif self.active_motion_label is not None and status.phase == RobotPhase.DONE:
            self._advance_after_motion_done()

        step(self.model, self.data)

    def render_step(self, render: RenderApp) -> None:
        if self.debug:
            gizmos = render.gizmos
            if self.target is not None:
                target_pose = self._get_target_pose()
                gizmos.draw_sphere(0.045, target_pose[:3], color=Color.rgb(1.0, 0.0, 0.0))
                gizmos.draw_sphere(0.022, target_pose[:3], color=Color.rgb(1.0, 1.0, 0.0))
                gizmos.draw_axes(target_pose[:3], target_pose[3:7], length=0.16)

            base_pose = self.controller.get_base_pose(self.data)
            base_pos = np.array(base_pose[:3], dtype=np.float64)

            if self.active_arm is not None:
                active_eef_pose = self.controller.get_end_effector_pose(self.data, self.active_arm)
                arm_color = Color.rgb(1.0, 0.4, 0.0) if self.active_arm == "left" else Color.rgb(0.2, 0.7, 1.0)
                gizmos.draw_sphere(0.04, active_eef_pose[:3], color=arm_color)
                gizmos.draw_axes(active_eef_pose[:3], active_eef_pose[3:7], length=0.14)
                if self.target is not None:
                    gizmos.draw_line(active_eef_pose[:3], self._get_target_pose()[:3], color=arm_color)

            for marker, color in (
                (self.markers.safe_base_target, Color.rgb(1.0, 0.0, 0.0)),
                (self.markers.approach_base_target, Color.rgb(1.0, 0.6, 0.0)),
                (self.markers.table_base_target, Color.rgb(0.6, 0.2, 1.0)),
            ):
                if marker is None:
                    continue
                marker_pos = np.array([marker[0], marker[1], base_pos[2]], dtype=np.float64)
                marker_quat = _quat_from_yaw(_body_yaw_from_heading(float(marker[2])))
                gizmos.draw_sphere(0.035, marker_pos, color=color)
                gizmos.draw_axes(marker_pos, marker_quat, length=0.16)
                gizmos.draw_line(base_pos, marker_pos, color=Color.rgb(1.0, 1.0, 0.0))

            if self.table_target_y is not None:
                line_a = np.array([-0.55, self.table_target_y, TABLE_SURFACE_Z + 0.01], dtype=np.float64)
                line_b = np.array([0.55, self.table_target_y, TABLE_SURFACE_Z + 0.01], dtype=np.float64)
                gizmos.draw_line(line_a, line_b, color=Color.rgb(0.8, 0.0, 0.8))

            for pose, color in (
                (self.markers.pregrasp_pose, Color.rgb(1.0, 0.8, 0.2)),
                (self.markers.grasp_pose, Color.rgb(1.0, 0.4, 0.0)),
                (self.markers.verify_pose, Color.rgb(0.9, 0.9, 0.2)),
                (self.markers.place_pose, Color.rgb(0.2, 0.7, 1.0)),
                (self.markers.retreat_pose, Color.rgb(0.3, 1.0, 0.6)),
            ):
                if pose is None:
                    continue
                gizmos.draw_cuboid(np.array([0.05, 0.05, 0.05]), pose[:3], rot=pose[3:7])
                gizmos.draw_axes(pose[:3], pose[3:7], length=0.12)
                if self.active_arm is not None:
                    current_pose = self.controller.get_end_effector_pose(self.data, self.active_arm)
                    gizmos.draw_line(current_pose[:3], pose[:3], color=Color.rgb(1.0, 1.0, 0.0))

        render.sync(self.data)

    def run(self) -> None:
        with RenderApp() as render:
            render.launch(self.model)
            self.setup()
            run.render_loop(
                self.model.options.timestep,
                60,
                self.phys_step,
                lambda: self.render_step(render),
            )


def main(model_path: str, seed: int | None, debug: bool) -> None:
    demo = GraspBottleDemo(model_path=model_path, seed=seed, debug=debug)
    demo.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly grasp a front-row bottle from the shelf and place it on the table")
    parser.add_argument("--model", default="xmls/bottle_demo.xml", help="model xml path")
    parser.add_argument("--seed", type=int, default=None, help="random seed for target bottle selection")
    parser.add_argument("--debug", action="store_true", help="draw debug gizmos for current targets and paths")
    args = parser.parse_args()
    main(model_path=args.model, seed=args.seed, debug=args.debug)


# Example command:
# DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/grasp_bottle_demo.py --debug
