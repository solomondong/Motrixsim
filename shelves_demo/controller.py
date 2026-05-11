from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence

import numpy as np
from motrixsim import SceneData, ik, step


def _as_np(values: Sequence[float], expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    return array


def _normalize_quat(quat: Sequence[float]) -> np.ndarray:
    q = _as_np(quat, 4, "quat")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("quat norm is too small")
    return q / norm


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _quat_rotate(quat: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = _normalize_quat(quat)
    x, y, z = _as_np(vector, 3, "vector")
    uv = np.cross(np.array([qx, qy, qz]), np.array([x, y, z]))
    uuv = np.cross(np.array([qx, qy, qz]), uv)
    return np.array([x, y, z], dtype=np.float64) + 2.0 * (qw * uv + uuv)


def _yaw_from_quat(quat: Sequence[float]) -> float:
    qx, qy, qz, qw = _normalize_quat(quat)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_midpoint(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    left = _as_np(left, 7, "left_pose")
    right = _as_np(right, 7, "right_pose")
    midpoint = (np.array(left[:3], dtype=np.float64) + np.array(right[:3], dtype=np.float64)) * 0.5
    left_quat = np.array(left[3:7], dtype=np.float64)
    right_quat = np.array(right[3:7], dtype=np.float64)
    quat = left_quat + right_quat
    if np.dot(left_quat, right_quat) < 0.0:
        quat = left_quat - right_quat
    if np.linalg.norm(quat) < 1e-12:
        quat = left_quat
    return np.concatenate((midpoint, _normalize_quat(quat)))


def _heading_from_body_yaw(body_yaw: float) -> float:
    # In the current robot XML, the work side of the robot is local -Y.
    # We define heading as the world yaw of the robot front direction.
    # MuJoCo body yaw is aligned with the body's local +X axis, so the two
    # differ by +pi/2.
    return _wrap_angle(body_yaw - 0.5 * math.pi)


@dataclass
class BaseTarget:
    x: float
    y: float
    heading: float


@dataclass
class BaseMovePlan:
    target: BaseTarget
    path_heading: float
    drive_direction: str = "forward"

    @property
    def target_xy(self) -> np.ndarray:
        return np.array([self.target.x, self.target.y], dtype=np.float64)


@dataclass
class ArmMovePlan:
    mode: "RobotMode"
    active_arm: Optional[str]
    left_target: np.ndarray
    right_target: np.ndarray
    lifter_target_z: float
    left_arm_qpos: np.ndarray
    right_arm_qpos: np.ndarray
    left_residual: float
    right_residual: float

    def __post_init__(self) -> None:
        self.left_target = _as_np(self.left_target, 7, "left_target")
        self.right_target = _as_np(self.right_target, 7, "right_target")
        self.left_arm_qpos = _as_np(self.left_arm_qpos, 7, "left_arm_qpos")
        self.right_arm_qpos = _as_np(self.right_arm_qpos, 7, "right_arm_qpos")


class RobotMode(str, Enum):
    BASE = "base"
    DUAL_ARM = "dual_arm"
    SINGLE_ARM = "single_arm"


class RobotPhase(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class BaseStage(str, Enum):
    ROTATE_TO_PATH = "rotate_to_path"
    MOVE_TO_TARGET = "move_to_target"
    ROTATE_TO_HEADING = "rotate_to_heading"


@dataclass
class RobotStatus:
    mode: Optional[RobotMode]
    phase: RobotPhase
    active: bool
    detail: Optional[str] = None
    error: Optional[str] = None
    plan: Optional[BaseMovePlan | ArmMovePlan] = None


@dataclass(frozen=True)
class RobotControllerConfig:
    robot_prefix: Optional[str] = "rb_6"
    base_body: str = "base_link"
    left_shoulder_body: str = "l_base_link"
    right_shoulder_body: str = "r_base_link"
    left_chain_root: str = "l_base_link"
    left_chain_tip: str = "L_Link_7"
    right_chain_root: str = "r_base_link"
    right_chain_tip: str = "R_Link_7"
    end_effector_offset: tuple[float, float, float, float, float, float, float] = (0, 0, 0.2, 0.0, 0.0, 0.0, 1.0)
    left_home: tuple[float, ...] = (0.5, -1.6, 0.0, -0.95, -1.2, -1.8, -2.9)
    right_home: tuple[float, ...] = (-0.5, 1.6, 0.0, 0.95, 1.2, 1.8, 2.9)
    # left_home: tuple[float, ...] = (0.0, -1.25, 0.0, -2.35, -1.57, -1.57, -0.36)
    # right_home: tuple[float, ...] = (0.0, 1.25, 0.0, 2.35, 1.57, 1.57, 0.36)
    lifter_home: float = 0.5
    lifter_min: float = 0.0
    lifter_max: float = 0.87
    min_drive_speed_cmd: float = 0.4
    max_drive_speed_cmd: float = 5.0
    max_turn_cmd: float = 5.0
    min_turn_cmd: float = 0.6
    drive_gain: float = 15.0
    heading_gain: float = 19.0
    heading_tolerance: float = 0.05
    move_heading_tolerance: float = 0.18
    position_tolerance: float = 0.03
    final_heading_tolerance: float = 0.03
    ik_residual_tolerance: float = 5e-3
    ik_max_iter: int = 100
    ik_step_size: float = 0.5
    ik_damping: float = 1e-3
    max_lifter_speed: float = 0.2
    arm_tolerance: float = 0.1
    lifter_tolerance: float = 0.01
    lifter_search_step: float = 0.1
    gripper_open: float = 1.0
    gripper_closed: float = 0.0


class RobotControllerError(RuntimeError):
    pass


class _RobotResolver:
    def __init__(self, model, robot_prefix: Optional[str]) -> None:
        self.model = model
        self.prefix = self._resolve_prefix(robot_prefix)

    def _resolve_prefix(self, robot_prefix: Optional[str]) -> Optional[str]:
        if robot_prefix is None:
            return None
        candidate = f"{robot_prefix}/wheel_l"
        if self.model.get_actuator(candidate) is not None:
            return robot_prefix
        if self.model.get_actuator("wheel_l") is not None:
            return None
        raise RobotControllerError("failed to resolve robot actuator namespace")

    def _candidate_names(self, name: str) -> list[str]:
        candidates: list[str] = []
        if self.prefix:
            candidates.append(f"{self.prefix}/{name}")
        candidates.append(name)
        return candidates

    def body(self, name: str) -> str:
        for candidate in self._candidate_names(name):
            if self.model.get_body(candidate) is not None:
                return candidate
        raise RobotControllerError(f"failed to resolve body name: {name}")

    def joint(self, name: str) -> str:
        for candidate in self._candidate_names(name):
            if self.model.get_joint(candidate) is not None:
                return candidate
        raise RobotControllerError(f"failed to resolve joint name: {name}")

    def actuator(self, name: str) -> str:
        for candidate in self._candidate_names(name):
            if self.model.get_actuator(candidate) is not None:
                return candidate
        raise RobotControllerError(f"failed to resolve actuator name: {name}")

    def link(self, name: str) -> str:
        for candidate in self._candidate_names(name):
            if self.model.get_link(candidate) is not None:
                return candidate
        raise RobotControllerError(f"failed to resolve link name: {name}")


class RobotController:
    """Unified controller with three modes: base, dual-arm, and single-arm.

    Public motion inputs stay intentionally small:
    - `move_base_world`: world-frame x, y, heading.
    - `move_dual_arm`: world-frame poses for both end effectors.
    - `move_single_arm`: world-frame pose for one end effector.

    The controller hides the robot-specific coordinate convention internally.
    For this robot, the front direction is local -Y while MuJoCo body yaw is
    measured from local +X, so heading conversion is handled in one place.
    """

    def __init__(self, model, config: RobotControllerConfig = RobotControllerConfig()) -> None:
        self.model = model
        self.config = config
        self._resolver = _RobotResolver(model, config.robot_prefix)
        resolver = self._resolver

        self._base_body = self.model.get_body(resolver.body(self.config.base_body))
        self._wheel_l = self.model.get_actuator(resolver.actuator("wheel_l"))
        self._wheel_r = self.model.get_actuator(resolver.actuator("wheel_r"))

        self._left_shoulder_link = self.model.get_link(resolver.link(self.config.left_shoulder_body))
        self._right_shoulder_link = self.model.get_link(resolver.link(self.config.right_shoulder_body))

        self._platform_joint = self.model.get_joint(resolver.joint("platform_base_joint"))
        self._platform_actuator = self.model.get_actuator(resolver.actuator("platform_base"))

        self._left_joints = [self.model.get_joint(resolver.joint(f"l_joint{i}")) for i in range(1, 8)]
        self._right_joints = [self.model.get_joint(resolver.joint(f"r_joint{i}")) for i in range(1, 8)]
        self._left_actuators = [self.model.get_actuator(resolver.actuator(f"L_joint{i}")) for i in range(1, 8)]
        self._right_actuators = [self.model.get_actuator(resolver.actuator(f"R_joint{i}")) for i in range(1, 8)]
        self._left_gripper_actuator = self.model.get_actuator(resolver.actuator("mimic_relation_l"))
        self._right_gripper_actuator = self.model.get_actuator(resolver.actuator("mimic_relation_r"))

        offset = np.asarray(self.config.end_effector_offset, dtype=np.float64)
        self._left_chain = ik.IkChain(
            self.model,
            end_link=resolver.link(self.config.left_chain_tip),
            start_link=resolver.link(self.config.left_chain_root),
            end_effector_offset=offset,
        )
        self._right_chain = ik.IkChain(
            self.model,
            end_link=resolver.link(self.config.right_chain_tip),
            start_link=resolver.link(self.config.right_chain_root),
            end_effector_offset=offset,
        )
        self._ik_solver = ik.DlsSolver(
            max_iter=self.config.ik_max_iter,
            step_size=self.config.ik_step_size,
            tolerance=self.config.ik_residual_tolerance,
            damping=self.config.ik_damping,
        )

        self._home_left = np.asarray(self.config.left_home, dtype=np.float64)
        self._home_right = np.asarray(self.config.right_home, dtype=np.float64)

        self._mode: Optional[RobotMode] = None
        self._phase = RobotPhase.IDLE
        self._detail: Optional[str] = None
        self._plan: Optional[BaseMovePlan | ArmMovePlan] = None
        self._error: Optional[str] = None
        self._base_stage: Optional[BaseStage] = None
        self._shoulder_height_offset = 0.0
        self._lifter_cmd = float(self.config.lifter_home)
        self._initialized = False

    def initialize(self, data: SceneData) -> None:
        self._command_wheels(data, 0.0, 0.0)
        self._command_home_pose(data)
        self._refresh_kinematics(data)

        current_lifter = float(self._platform_joint.get_dof_pos(data)[0])
        left_shoulder_pose = self._left_shoulder_link.get_pose(data)
        right_shoulder_pose = self._right_shoulder_link.get_pose(data)
        shoulder_mid_z = 0.5 * (left_shoulder_pose[2] + right_shoulder_pose[2])
        self._shoulder_height_offset = shoulder_mid_z - current_lifter

        # Keep the internal lifter command aligned with the commanded home
        # target instead of the transient measured position during startup.
        self._lifter_cmd = float(self.config.lifter_home)
        self._mode = None
        self._phase = RobotPhase.IDLE
        self._detail = None
        self._plan = None
        self._error = None
        self._base_stage = None
        self._initialized = True

    def status(self) -> RobotStatus:
        return RobotStatus(
            mode=self._mode,
            phase=self._phase,
            active=self._phase == RobotPhase.RUNNING and self._plan is not None,
            detail=self._detail,
            error=self._error,
            plan=self._plan,
        )

    def stop(self, data: SceneData, error: Optional[str] = None) -> RobotStatus:
        self._command_wheels(data, 0.0, 0.0)
        self._hold_current_upper_body(data)
        self._mode = None
        self._phase = RobotPhase.ERROR if error else RobotPhase.IDLE
        self._detail = None
        self._plan = None
        self._error = error
        self._base_stage = None
        return self.status()

    def get_base_pose(self, data: SceneData) -> np.ndarray:
        return np.array(self._base_body.get_pose(data), copy=True)

    def get_end_effector_pose(self, data: SceneData, arm: str) -> np.ndarray:
        chain = self._chain_for_arm(arm)
        return np.array(chain.get_end_effector_pose(data), copy=True)

    def set_gripper(self, data: SceneData, arm: str, opening: float) -> None:
        actuator = self._gripper_for_arm(arm)
        min_val = min(self.config.gripper_open, self.config.gripper_closed)
        max_val = max(self.config.gripper_open, self.config.gripper_closed)
        actuator.set_ctrl(data, float(np.clip(opening, min_val, max_val)))

    def open_gripper(self, data: SceneData, arm: str) -> None:
        self.set_gripper(data, arm, self.config.gripper_open)

    def close_gripper(self, data: SceneData, arm: str) -> None:
        self.set_gripper(data, arm, self.config.gripper_closed)

    def get_lifter_position(self, data: SceneData) -> float:
        return float(self._platform_joint.get_dof_pos(data)[0])

    def move_lifter_to(self, data: SceneData, target_z: float) -> None:
        clamped_target = float(np.clip(target_z, self.config.lifter_min, self.config.lifter_max))
        self._lifter_cmd = clamped_target
        self._platform_actuator.set_ctrl(data, self._lifter_cmd)

    def is_lifter_at(self, data: SceneData, target_z: float, tolerance: float | None = None) -> bool:
        if tolerance is None:
            tolerance = self.config.lifter_tolerance
        current_z = self.get_lifter_position(data)
        return abs(current_z - target_z) <= tolerance

    def move_base_world(
        self,
        data: SceneData,
        x: float,
        y: float,
        heading: float,
        *,
        drive_direction: str = "forward",
    ) -> BaseMovePlan:
        self._require_initialized()
        self._hold_current_upper_body(data)
        plan = self._plan_base_world(data, x=x, y=y, heading=heading, drive_direction=drive_direction)

        current_xy, _ = self._get_xy_heading(data)
        distance = float(np.linalg.norm(plan.target_xy - current_xy))
        base_stage = BaseStage.ROTATE_TO_HEADING if distance <= self.config.position_tolerance else BaseStage.ROTATE_TO_PATH
        self._activate_plan(mode=RobotMode.BASE, plan=plan, detail=None, base_stage=base_stage)
        return plan

    def move_dual_arm(
        self,
        data: SceneData,
        left_target: Sequence[float],
        right_target: Sequence[float],
        *,
        constrain_orientation: bool = False,
    ) -> ArmMovePlan:
        self._require_initialized()
        self._command_wheels(data, 0.0, 0.0)
        plan = self._plan_dual_arm(
            data,
            left_target=np.asarray(left_target, dtype=np.float64),
            right_target=np.asarray(right_target, dtype=np.float64),
            constrain_orientation=constrain_orientation,
        )
        self._activate_plan(mode=RobotMode.DUAL_ARM, plan=plan, detail="move_arms")
        return plan

    def move_single_arm(
        self,
        data: SceneData,
        arm: str,
        target_pose: Sequence[float],
        *,
        constrain_orientation: bool = False,
    ) -> ArmMovePlan:
        self._require_initialized()
        self._command_wheels(data, 0.0, 0.0)
        plan = self._plan_single_arm(
            data,
            arm=arm,
            target_pose=np.asarray(target_pose, dtype=np.float64),
            constrain_orientation=constrain_orientation,
        )
        self._activate_plan(mode=RobotMode.SINGLE_ARM, plan=plan, detail="move_arms")
        return plan

    def update(self, data: SceneData) -> RobotStatus:
        if self._plan is None or self._phase != RobotPhase.RUNNING:
            return self.status()

        try:
            if self._mode == RobotMode.BASE:
                self._update_base(data)
            elif self._mode in {RobotMode.DUAL_ARM, RobotMode.SINGLE_ARM}:
                self._update_arms(data)
            return self.status()
        except RobotControllerError as exc:
            return self.stop(data, error=str(exc))

    def _plan_base_world(
        self,
        data: SceneData,
        *,
        x: float,
        y: float,
        heading: float,
        drive_direction: str,
    ) -> BaseMovePlan:
        if drive_direction not in {"forward", "backward"}:
            raise ValueError("drive_direction must be 'forward' or 'backward'")
        current_xy, current_heading = self._get_xy_heading(data)
        target_xy = np.array([x, y], dtype=np.float64)
        delta = target_xy - current_xy
        distance = np.linalg.norm(delta)
        
        if distance <= 1e-9:
            return BaseMovePlan(
                target=BaseTarget(float(x), float(y), float(heading)),
                path_heading=float(current_heading),
                drive_direction=drive_direction,
            )
        
        path_heading = math.atan2(delta[1], delta[0])
        if drive_direction == "backward":
            path_heading = _wrap_angle(path_heading + math.pi)
        
        return BaseMovePlan(
            target=BaseTarget(float(x), float(y), float(heading)),
            path_heading=float(path_heading),
            drive_direction=drive_direction,
        )

    def _plan_dual_arm(
        self,
        data: SceneData,
        *,
        left_target: np.ndarray,
        right_target: np.ndarray,
        constrain_orientation: bool,
    ) -> ArmMovePlan:
        left_target = _as_np(left_target, 7, "left_target")
        right_target = _as_np(right_target, 7, "right_target")

        torso_ref = _pose_midpoint(left_target, right_target)
        lifter_target_z, left_residual, right_residual, left_qpos, right_qpos = self._search_arm_solution(
            data,
            torso_ref=torso_ref,
            left_target=left_target,
            right_target=right_target,
            solve_left=True,
            solve_right=True,
            constrain_left_orientation=constrain_orientation,
            constrain_right_orientation=constrain_orientation,
        )

        return ArmMovePlan(
            mode=RobotMode.DUAL_ARM,
            active_arm=None,
            left_target=left_target,
            right_target=right_target,
            lifter_target_z=lifter_target_z,
            left_arm_qpos=left_qpos,
            right_arm_qpos=right_qpos,
            left_residual=left_residual,
            right_residual=right_residual,
        )

    def _plan_single_arm(
        self,
        data: SceneData,
        *,
        arm: str,
        target_pose: np.ndarray,
        constrain_orientation: bool,
    ) -> ArmMovePlan:
        active_arm = self._normalize_arm_name(arm)
        target_pose = _as_np(target_pose, 7, "target_pose")
        other_arm = "right" if active_arm == "left" else "left"
        other_pose = self.get_end_effector_pose(data, other_arm)
        left_target, right_target = self._targets_for_active_arm(active_arm, target_pose, other_pose)

        torso_ref = _pose_midpoint(left_target, right_target)
        lifter_target_z, left_residual, right_residual, left_qpos, right_qpos = self._search_arm_solution(
            data,
            torso_ref=torso_ref,
            left_target=left_target,
            right_target=right_target,
            solve_left=active_arm == "left",
            solve_right=active_arm == "right",
            constrain_left_orientation=constrain_orientation,
            constrain_right_orientation=constrain_orientation,
        )

        return ArmMovePlan(
            mode=RobotMode.SINGLE_ARM,
            active_arm=active_arm,
            left_target=left_target,
            right_target=right_target,
            lifter_target_z=lifter_target_z,
            left_arm_qpos=left_qpos,
            right_arm_qpos=right_qpos,
            left_residual=left_residual,
            right_residual=right_residual,
        )

    def _search_arm_solution(
        self,
        data: SceneData,
        *,
        torso_ref: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
        solve_left: bool,
        solve_right: bool,
        constrain_left_orientation: bool,
        constrain_right_orientation: bool,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        saved_lifter = float(self._platform_joint.get_dof_pos(data)[0])
        nominal_lifter = self._compute_lifter_target(torso_ref)
        candidates = self._candidate_lifter_targets(nominal_lifter, saved_lifter)
        best_result = None
        best_score = None

        try:
            for candidate in candidates:
                self._platform_joint.set_dof_pos(data, np.array([candidate], dtype=np.float64))
                self._refresh_kinematics(data)

                if solve_left:
                    left_ok, left_residual, left_qpos = self._solve_arm_ik(
                        data,
                        arm="left",
                        target_pose=left_target,
                        constrain_orientation=constrain_left_orientation,
                    )
                else:
                    left_residual = 0.0
                    left_qpos = self._current_arm_qpos(data, "left")
                    left_ok = True

                if solve_right:
                    right_ok, right_residual, right_qpos = self._solve_arm_ik(
                        data,
                        arm="right",
                        target_pose=right_target,
                        constrain_orientation=constrain_right_orientation,
                    )
                else:
                    right_residual = 0.0
                    right_qpos = self._current_arm_qpos(data, "right")
                    right_ok = True

                if not left_ok or not right_ok:
                    continue

                score = (
                    left_residual + right_residual,
                    abs(candidate - saved_lifter),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_result = (candidate, left_residual, right_residual, left_qpos, right_qpos)
        finally:
            self._platform_joint.set_dof_pos(data, np.array([saved_lifter], dtype=np.float64))
            self._refresh_kinematics(data)

        if best_result is None:
            raise RobotControllerError("arm target is not reachable with the current base pose")

        return best_result

    def _update_base(self, data: SceneData) -> None:
        plan = self._require_base_plan()
        stage = self._base_stage
        if stage is None:
            raise RobotControllerError("base controller stage is not set")

        if stage == BaseStage.ROTATE_TO_PATH:
            current_xy, _ = self._get_xy_heading(data)
            distance = float(np.linalg.norm(plan.target_xy - current_xy))
            self._detail = stage.value
            if distance <= self.config.position_tolerance:
                self._command_wheels(data, 0.0, 0.0)
                self._base_stage = BaseStage.ROTATE_TO_HEADING
                return

            heading_error = self._command_turn_only(data, plan.path_heading)
            if heading_error <= self.config.heading_tolerance:
                self._base_stage = BaseStage.MOVE_TO_TARGET
            return

        if stage == BaseStage.MOVE_TO_TARGET:
            current_xy, current_heading = self._get_xy_heading(data)
            delta = plan.target_xy - current_xy
            distance = float(np.linalg.norm(delta))
            self._detail = stage.value
            if distance <= self.config.position_tolerance:
                self._command_wheels(data, 0.0, 0.0)
                self._base_stage = BaseStage.ROTATE_TO_HEADING
                return

            path_heading = math.atan2(delta[1], delta[0])
            if plan.drive_direction == "backward":
                path_heading = _wrap_angle(path_heading + math.pi)
            heading_error = _wrap_angle(path_heading - current_heading)
            if abs(heading_error) > self.config.move_heading_tolerance:
                self._command_wheels(data, 0.0, 0.0)
                plan.path_heading = float(path_heading)
                self._base_stage = BaseStage.ROTATE_TO_PATH
                return

            linear_scale = max(0.0, math.cos(heading_error))
            drive_speed_cmd = float(
                np.clip(
                    self.config.drive_gain * distance * linear_scale,
                    self.config.min_drive_speed_cmd,
                    self.config.max_drive_speed_cmd,
                )
            )
            turn_cmd = self._compute_turn_cmd(heading_error)
            if plan.drive_direction == "backward":
                drive_speed_cmd = -drive_speed_cmd
            self._command_wheels(data, drive_speed_cmd - turn_cmd, drive_speed_cmd + turn_cmd)
            return

        if stage == BaseStage.ROTATE_TO_HEADING:
            self._detail = stage.value
            heading_error = self._command_turn_only(data, plan.target.heading)
            if heading_error <= self.config.final_heading_tolerance:
                self._command_wheels(data, 0.0, 0.0)
                self._phase = RobotPhase.DONE
                self._detail = stage.value
                self._plan = None
                self._base_stage = None
            return

        raise RobotControllerError(f"unsupported base stage: {stage}")

    def _update_arms(self, data: SceneData) -> None:
        plan = self._require_arm_plan()
        self._detail = "move_arms"
        self._check_arm_safety(plan)
        self._command_lifter(data, plan.lifter_target_z)
        self._command_arms(data, plan.left_arm_qpos, plan.right_arm_qpos)
        if self._is_lifter_close(data, plan) and self._are_arms_close(data, plan):
            self._phase = RobotPhase.DONE
            self._plan = None

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RobotControllerError("controller.initialize(data) must be called before planning")

    def _require_base_plan(self) -> BaseMovePlan:
        if not isinstance(self._plan, BaseMovePlan):
            raise RobotControllerError("active plan is not a base plan")
        return self._plan

    def _activate_plan(
        self,
        *,
        mode: RobotMode,
        plan: BaseMovePlan | ArmMovePlan,
        detail: Optional[str],
        base_stage: Optional[BaseStage] = None,
    ) -> None:
        self._mode = mode
        self._phase = RobotPhase.RUNNING
        self._detail = detail
        self._plan = plan
        self._error = None
        self._base_stage = base_stage

    def _require_arm_plan(self) -> ArmMovePlan:
        if not isinstance(self._plan, ArmMovePlan):
            raise RobotControllerError("active plan is not an arm plan")
        return self._plan

    def _normalize_arm_name(self, arm: str) -> str:
        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'")
        return arm

    def _targets_for_active_arm(
        self,
        active_arm: str,
        active_target: np.ndarray,
        passive_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if active_arm == "left":
            return active_target, passive_target
        return passive_target, active_target

    def _chain_for_arm(self, arm: str):
        arm = self._normalize_arm_name(arm)
        return self._left_chain if arm == "left" else self._right_chain

    def _gripper_for_arm(self, arm: str):
        arm = self._normalize_arm_name(arm)
        return self._left_gripper_actuator if arm == "left" else self._right_gripper_actuator

    def _command_home_pose(self, data: SceneData) -> None:
        self._lifter_cmd = float(self.config.lifter_home)
        self._platform_actuator.set_ctrl(data, self._lifter_cmd)
        for actuator, target in zip(self._left_actuators, self._home_left):
            actuator.set_ctrl(data, float(target))
        for actuator, target in zip(self._right_actuators, self._home_right):
            actuator.set_ctrl(data, float(target))
        self.open_gripper(data, "left")
        self.open_gripper(data, "right")

    def _hold_current_upper_body(self, data: SceneData) -> None:
        self._lifter_cmd = float(self._platform_joint.get_dof_pos(data)[0])
        self._platform_actuator.set_ctrl(data, self._lifter_cmd)
        self._command_arms(data, self._current_arm_qpos(data, "left"), self._current_arm_qpos(data, "right"))

    def _refresh_kinematics(self, data: SceneData) -> None:
        step(self.model, data)

    def _get_xy_heading(self, data: SceneData) -> tuple[np.ndarray, float]:
        pose = self._base_body.get_pose(data)
        body_yaw = _yaw_from_quat(pose[3:7])
        return np.array(pose[:2], dtype=np.float64), _heading_from_body_yaw(body_yaw)

    def _compute_turn_cmd(self, heading_error: float) -> float:
        turn_cmd = float(np.clip(self.config.heading_gain * heading_error, -self.config.max_turn_cmd, self.config.max_turn_cmd))
        if abs(turn_cmd) < self.config.min_turn_cmd and abs(heading_error) > self.config.heading_tolerance:
            return self.config.min_turn_cmd if turn_cmd > 0.0 else -self.config.min_turn_cmd
        return turn_cmd

    def _command_wheels(self, data: SceneData, left_cmd: float, right_cmd: float) -> None:
        self._wheel_l.set_ctrl(data, float(left_cmd))
        self._wheel_r.set_ctrl(data, float(right_cmd))

    def _command_turn_only(self, data: SceneData, target_heading: float) -> float:
        _, current_heading = self._get_xy_heading(data)
        heading_error = _wrap_angle(target_heading - current_heading)
        turn_cmd = self._compute_turn_cmd(heading_error)
        self._command_wheels(data, -turn_cmd, turn_cmd)
        return abs(heading_error)

    def _current_arm_qpos(self, data: SceneData, arm: str) -> np.ndarray:
        arm = self._normalize_arm_name(arm)
        joints = self._left_joints if arm == "left" else self._right_joints
        return np.array([joint.get_dof_pos(data)[0] for joint in joints], dtype=np.float64)

    def _solve_arm_ik(
        self,
        data: SceneData,
        *,
        arm: str,
        target_pose: np.ndarray,
        constrain_orientation: bool,
    ) -> tuple[bool, float, np.ndarray]:
        arm = self._normalize_arm_name(arm)
        chain = self._left_chain if arm == "left" else self._right_chain
        desired_pose = np.array(target_pose, copy=True)
        if not constrain_orientation:
            desired_pose[3:7] = chain.get_end_effector_pose(data)[3:7]
        result = self._ik_solver.solve(chain, data, desired_pose)
        residual = float(result[1])
        qpos = np.asarray(result[2:], dtype=np.float64)
        success = residual <= self.config.ik_residual_tolerance and self._within_joint_limits(arm, qpos)
        return success, residual, qpos

    def _within_joint_limits(self, arm: str, qpos: np.ndarray) -> bool:
        arm = self._normalize_arm_name(arm)
        joints = self._left_joints if arm == "left" else self._right_joints
        for joint, value in zip(joints, qpos):
            lower, upper = self.model.joint_limits[:, self.model.get_joint_index(joint.name)]
            if value < lower - 1e-6 or value > upper + 1e-6:
                return False
        return True

    def _compute_lifter_target(self, torso_ref: np.ndarray) -> float:
        lifter_z = float(torso_ref[2] - self._shoulder_height_offset)
        return lifter_z

    def _candidate_lifter_targets(self, nominal_lifter_z: float, current_lifter_z: float) -> list[float]:
        step_size = self.config.lifter_search_step
        if step_size <= 0.0:
            raise RobotControllerError("lifter_search_step must be positive")

        clamped_nominal = float(np.clip(nominal_lifter_z, self.config.lifter_min, self.config.lifter_max))
        direction = 1.0 if clamped_nominal >= current_lifter_z else -1.0
        travel = abs(clamped_nominal - current_lifter_z)
        step_count = int(math.floor(travel / step_size))

        candidates = [
            float(np.clip(current_lifter_z + direction * step_size * i, self.config.lifter_min, self.config.lifter_max))
            for i in range(step_count + 1)
        ]
        candidates.append(clamped_nominal)

        unique_candidates: list[float] = []
        for candidate in candidates:
            if any(abs(candidate - existing) < 1e-9 for existing in unique_candidates):
                continue
            unique_candidates.append(candidate)

        if direction > 0.0:
            unique_candidates.sort()
        else:
            unique_candidates.sort(reverse=True)
        return unique_candidates

    def _command_lifter(self, data: SceneData, target_z: float) -> None:
        dt = float(self.model.options.timestep)
        max_step = self.config.max_lifter_speed * dt
        self._lifter_cmd += float(np.clip(target_z - self._lifter_cmd, -max_step, max_step))
        self._platform_actuator.set_ctrl(data, self._lifter_cmd)

    def _command_arms(self, data: SceneData, left_qpos: np.ndarray, right_qpos: np.ndarray) -> None:
        for actuator, target in zip(self._left_actuators, left_qpos):
            actuator.set_ctrl(data, float(target))
        for actuator, target in zip(self._right_actuators, right_qpos):
            actuator.set_ctrl(data, float(target))

    def _is_lifter_close(self, data: SceneData, plan: ArmMovePlan) -> bool:
        current_z = float(self._platform_joint.get_dof_pos(data)[0])
        return abs(plan.lifter_target_z - current_z) <= self.config.lifter_tolerance

    def _are_arms_close(self, data: SceneData, plan: ArmMovePlan) -> bool:
        if plan.mode == RobotMode.SINGLE_ARM:
            arm = plan.active_arm
            error = np.max(np.abs(self._current_arm_qpos(data, arm) - (plan.left_arm_qpos if arm == "left" else plan.right_arm_qpos)))
            return error <= self.config.arm_tolerance
        else:
            left_error = np.max(np.abs(self._current_arm_qpos(data, "left") - plan.left_arm_qpos))
            right_error = np.max(np.abs(self._current_arm_qpos(data, "right") - plan.right_arm_qpos))
            return left_error <= self.config.arm_tolerance and right_error <= self.config.arm_tolerance

    def _check_arm_safety(self, plan: ArmMovePlan) -> None:
        if not self._within_joint_limits("left", plan.left_arm_qpos):
            raise RobotControllerError("left arm target exceeds joint limits")
        if not self._within_joint_limits("right", plan.right_arm_qpos):
            raise RobotControllerError("right arm target exceeds joint limits")


__all__ = [
    "ArmMovePlan",
    "BaseMovePlan",
    "BaseTarget",
    "RobotController",
    "RobotControllerConfig",
    "RobotControllerError",
    "RobotMode",
    "RobotPhase",
    "RobotStatus",
]
