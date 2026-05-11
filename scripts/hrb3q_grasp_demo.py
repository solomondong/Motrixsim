"""HRB3Q-LC pick-and-place demo (fixed base).

Scenario: a bottle sits on the upper shelf deck in front (+Y) of the robot.
The left arm picks it up, the torso (Joint_Body) rotates ~180 deg so the arm
swings behind the robot, and the bottle is placed on the table at -Y.

The default grasp path uses the simulated Dahuan parallel gripper actuators and
real contact/friction.  A `--fake-grasp` fallback is kept for debugging only.

Motion plan (all stages use the LEFT arm only):
  1. READY            - open-loop tuck arm in front so IK has a good seed
  2. MOVE_TO_GRASP    - closed-loop IK to the bottle
  3. CLOSE_GRASP      - activate grasp, wait for settle
  4. LIFT             - closed-loop IK straight up
  5. TUCK_1           - open-loop retract arm close to torso before rotating
  6. ROTATE_BODY      - open-loop command Joint_Body -> pi (bottle held by gripper contact)
  7. MOVE_ABOVE_TABLE - closed-loop IK above the place pose
  8. LOWER_TO_PLACE   - closed-loop IK down to table
  9. RELEASE          - release grasp, let bottle settle on table
 10. RETREAT          - closed-loop IK lift arm
 11. TUCK_2           - open-loop retract arm
 12. ROTATE_BACK      - open-loop Joint_Body -> 0
 13. HOME             - open-loop zero-pose
 14. DONE

Run:
  DISPLAY=:1 conda run --live-stream -n motrix-lerobot python scripts/hrb3q_grasp_demo.py --debug
"""
from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path

import numpy as np

from motrixsim import SceneData, ik, load_model, run as render_run, step
from motrixsim.render import Color, RenderApp


# ---------------------------------------------------------------------------
# Quaternion / pose helpers (motrixsim uses [x, y, z, w])
# ---------------------------------------------------------------------------

def _normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def pose_compose(parent: np.ndarray, child_in_parent: np.ndarray) -> np.ndarray:
    p_pos, p_quat = parent[:3], _normalize(parent[3:7])
    c_pos, c_quat = child_in_parent[:3], _normalize(child_in_parent[3:7])
    return np.concatenate([
        p_pos + quat_rotate(p_quat, c_pos),
        _normalize(quat_mul(p_quat, c_quat)),
    ])


def pose_relative(world: np.ndarray, frame: np.ndarray) -> np.ndarray:
    f_pos, f_quat = frame[:3], _normalize(frame[3:7])
    inv_q = quat_conj(f_quat)
    return np.concatenate([
        quat_rotate(inv_q, world[:3] - f_pos),
        _normalize(quat_mul(inv_q, _normalize(world[3:7]))),
    ])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTUP_STEPS = 400
EEF_OFFSET = np.array([0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

# Poses for the bottle / table are in world frame and must match the XML layout.
BOTTLE_WORLD = np.array([-0.20, 0.58, 1.396], dtype=np.float64)
BOTTLE_HEIGHT = 0.22

TABLE_SURFACE_Z = 0.816
PLACE_WORLD_XY = np.array([0.20, -0.42], dtype=np.float64)

LIFT_DELTA_Z = 0.14
RETREAT_DELTA_Z = 0.15

# Gripper ctrl values (must match actuator_l_gripper ctrlrange in XML).
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
GRIPPER_PINCH = 0.012
GRIPPER_PAD_CENTER_Z = 0.055

# Joint targets for open-loop shaping stages.  Convention: q = [body, l1..l7].
READY_QPOS   = np.array([0.0, 0.0, 1.1, 0.0, 1.6, 0.0, 0.0, 0.0], dtype=np.float64)
TUCK_QPOS    = np.array([0.0, 0.0, 1.6, 0.0, 2.4, 0.0, 0.0, 0.0], dtype=np.float64)
TUCK_PI_QPOS = np.array([np.pi, 0.0, 1.6, 0.0, 2.4, 0.0, 0.0, 0.0], dtype=np.float64)
HOME_QPOS    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Stage completion thresholds
EE_POS_TOL = 0.03
QPOS_TOL = 0.12
CLOSE_GRASP_STEPS = 60
RELEASE_STEPS = 100

# Closed-loop IK parameters (iterations per physics step)
IK_MAX_ITER_PER_STEP = 20
IK_STEP_SIZE = 0.4
IK_DAMPING = 1e-3
IK_TOL = 5e-4


# ---------------------------------------------------------------------------
# Stage machine
# ---------------------------------------------------------------------------

class Mode(str, Enum):
    OPEN_LOOP_QPOS = "open_loop_qpos"   # drive actuators toward a fixed q target
    CLOSED_LOOP_IK = "closed_loop_ik"   # re-solve IK every step to track an EE target
    WAIT = "wait"                        # hold current ctrls for N steps


class Stage(str, Enum):
    STARTUP = "startup"
    READY = "ready"
    MOVE_TO_GRASP = "move_to_grasp"
    CLOSE_GRASP = "close_grasp"
    LIFT = "lift"
    TUCK_1 = "tuck_1"
    ROTATE_BODY = "rotate_body"
    MOVE_ABOVE_TABLE = "move_above_table"
    LOWER_TO_PLACE = "lower_to_place"
    RELEASE = "release"
    RETREAT = "retreat"
    TUCK_2 = "tuck_2"
    ROTATE_BACK = "rotate_back"
    HOME = "home"
    DONE = "done"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

class HRB3QGraspDemo:
    def __init__(self, model_path: str, debug: bool = False, fake_grasp: bool = False) -> None:
        self.debug = debug
        self.fake_grasp = fake_grasp
        abs_path = str(Path(model_path).resolve())
        print(f"loading: {abs_path}")
        print(f"grasp mode: {'FAKE (teleport)' if fake_grasp else 'REAL (gripper actuator)'}")
        self.model = load_model(abs_path)
        self.data = SceneData(self.model)

        # IK chain: Body -> arml7 (8 DOFs including Joint_Body)
        self.chain = ik.IkChain(
            self.model, end_link="arml7", start_link="Body",
            end_effector_offset=EEF_OFFSET,
        )
        self.solver = ik.DlsSolver(
            max_iter=IK_MAX_ITER_PER_STEP,
            step_size=IK_STEP_SIZE,
            tolerance=IK_TOL,
            damping=IK_DAMPING,
        )

        # Joints and actuators
        self.joint_body = self._require("joint", "Joint_Body")
        self.joint_waist = self._require("joint", "Joint_Waist")
        self.joints_l = [self._require("joint", f"Joint_l{i}") for i in range(1, 8)]

        self.act_waist = self._require("actuator", "actuator_waist")
        self.act_body = self._require("actuator", "actuator_body")
        self.acts_l = [self._require("actuator", f"actuator_l{i}") for i in range(1, 8)]
        self.acts_r = [self._require("actuator", f"actuator_r{i}") for i in range(1, 8)]

        # Gripper actuators (real grasping).  Each visible segment has its own
        # actuator; command all segment joints together so the outer pads cannot lag.
        self.l_grippers = self._require_actuator_group([
            "actuator_l_gripper",
            "actuator_l_gripper_r",
            "actuator_l_gripper_l_pad",
            "actuator_l_gripper_r_pad",
        ])
        self.r_grippers = self._require_actuator_group([
            "actuator_r_gripper",
            "actuator_r_gripper_r",
            "actuator_r_gripper_l_pad",
            "actuator_r_gripper_r_pad",
        ])
        self.l_gripper_target = GRIPPER_OPEN
        self.r_gripper_target = GRIPPER_OPEN

        self.bottle_body = self.model.get_body("Bt_/001_bottle_base2")
        if self.bottle_body is None:
            raise RuntimeError("bottle body not found: Bt_/001_bottle_base2")

        # Teleport (fake-grasp) state — only used when --fake-grasp is set.
        self.grasp_active = False
        self.bottle_in_ee: np.ndarray | None = None

        # Stage state
        self.stage = Stage.STARTUP
        self.stage_step = 0
        self.mode = Mode.WAIT
        self.qpos_target: np.ndarray | None = None
        self.ee_target: np.ndarray | None = None
        self.wait_steps = 0

    # ------------------------------------------------------------------
    def _require(self, kind: str, name: str):
        getter = getattr(self.model, f"get_{kind}")
        obj = getter(name)
        if obj is None:
            raise RuntimeError(f"{kind} not found: {name}")
        return obj

    def _require_actuator_group(self, names: list[str]):
        actuators = []
        missing = []
        for name in names:
            actuator = self.model.get_actuator(name)
            if actuator is None:
                missing.append(name)
            else:
                actuators.append(actuator)
        if missing:
            raise RuntimeError(f"gripper actuators not found: {missing}")
        return actuators

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------
    def _ee_pose(self) -> np.ndarray:
        return np.array(self.chain.get_end_effector_pose(self.data), dtype=np.float64)

    def _bottle_pose(self) -> np.ndarray:
        return np.array(self.bottle_body.get_pose(self.data), dtype=np.float64)

    def _current_qpos(self) -> np.ndarray:
        vals = [self.joint_body.get_dof_pos(self.data)[0]]
        vals.extend(j.get_dof_pos(self.data)[0] for j in self.joints_l)
        return np.array(vals, dtype=np.float64)

    # ------------------------------------------------------------------
    # Commanding
    # ------------------------------------------------------------------
    def _command_qpos(self, qpos: np.ndarray) -> None:
        self.act_body.set_ctrl(self.data, float(qpos[0]))
        for act, q in zip(self.acts_l, qpos[1:8]):
            act.set_ctrl(self.data, float(q))

    def _command_right_arm_zero(self) -> None:
        for act in self.acts_r:
            act.set_ctrl(self.data, 0.0)

    def _command_waist_zero(self) -> None:
        self.act_waist.set_ctrl(self.data, 0.0)

    # ------------------------------------------------------------------
    # IK (closed-loop)
    # ------------------------------------------------------------------
    def _solve_ik_step(self, ee_target: np.ndarray) -> np.ndarray:
        desired = ee_target.astype(np.float64).copy()
        # Keep current orientation (position-only IK)
        desired[3:7] = self._ee_pose()[3:7]
        result = np.asarray(
            self.solver.solve(self.chain, self.data, desired), dtype=np.float64
        )
        return result[2:2 + self.chain.num_dof_pos]

    # ------------------------------------------------------------------
    # Grasp (real via gripper actuator; fake/teleport only when --fake-grasp)
    # ------------------------------------------------------------------
    def _command_gripper(self) -> None:
        """Re-issue the current gripper ctrl target (called every phys step)."""
        for actuator in self.l_grippers:
            actuator.set_ctrl(self.data, float(self.l_gripper_target))
        for actuator in self.r_grippers:
            actuator.set_ctrl(self.data, float(self.r_gripper_target))

    def _activate_grasp(self) -> None:
        # Do not command fully closed: leave a small target gap so the gripper
        # keeps squeezing the bottle instead of numerically tunnelling through it.
        self.l_gripper_target = GRIPPER_PINCH
        if self.fake_grasp:
            ee = self._ee_pose()
            bottle = self._bottle_pose()
            self.bottle_in_ee = pose_relative(bottle, ee)
            self.grasp_active = True
            print(f"  [fake] teleport grasp activated. bottle@EE = {self.bottle_in_ee[:3]}")
        else:
            print(f"  [real] closing gripper group -> {GRIPPER_PINCH}")

    def _deactivate_grasp(self) -> None:
        self.l_gripper_target = GRIPPER_OPEN
        if self.fake_grasp:
            self.grasp_active = False
            self.bottle_in_ee = None
            print("  [fake] teleport released")
        else:
            print(f"  [real] opening gripper: actuator_l_gripper -> {GRIPPER_OPEN}")

    def _update_grasp_follow(self) -> None:
        """Teleport-style bottle follow. Only used when --fake-grasp is set."""
        if not self.fake_grasp or not self.grasp_active or self.bottle_in_ee is None:
            return
        ee = self._ee_pose()
        pose = pose_compose(ee, self.bottle_in_ee).astype(np.float64)
        self.bottle_body.set_dof_pos(self.data, pose, include_floatingbase=True)
        self.bottle_body.set_dof_vel(
            self.data, np.zeros(6, dtype=np.float64), include_floatingbase=True
        )

    # ------------------------------------------------------------------
    # Stage control
    # ------------------------------------------------------------------
    def _enter_open_loop(self, stage: Stage, qpos_target: np.ndarray) -> None:
        self.stage = stage
        self.stage_step = 0
        self.mode = Mode.OPEN_LOOP_QPOS
        self.qpos_target = qpos_target.copy()
        self.ee_target = None
        self._command_qpos(qpos_target)
        current = self._current_qpos()
        err = float(np.max(np.abs(current - qpos_target)))
        print(f"[{stage.value}] open-loop qpos target ({qpos_target.round(2).tolist()}), initial err={err:.3f}")

    def _enter_closed_loop(self, stage: Stage, ee_target: np.ndarray) -> None:
        self.stage = stage
        self.stage_step = 0
        self.mode = Mode.CLOSED_LOOP_IK
        self.qpos_target = None
        self.ee_target = ee_target.copy()
        ee_pos = self._ee_pose()[:3]
        dist = float(np.linalg.norm(ee_pos - ee_target[:3]))
        print(f"[{stage.value}] closed-loop IK target pos={ee_target[:3]}, initial dist={dist:.3f}")

    def _enter_wait(self, stage: Stage, steps: int) -> None:
        self.stage = stage
        self.stage_step = 0
        self.mode = Mode.WAIT
        self.wait_steps = steps
        self.qpos_target = None
        self.ee_target = None
        print(f"[{stage.value}] wait for {steps} steps")

    # ------------------------------------------------------------------
    def _stage_done(self) -> bool:
        if self.mode == Mode.OPEN_LOOP_QPOS and self.qpos_target is not None:
            err = float(np.max(np.abs(self._current_qpos() - self.qpos_target)))
            if self.stage_step > 2000:
                print(f"  !! {self.stage.value} timed out at err={err:.3f}; advancing")
                return True
            return err < QPOS_TOL
        if self.mode == Mode.CLOSED_LOOP_IK and self.ee_target is not None:
            dist = float(np.linalg.norm(self._ee_pose()[:3] - self.ee_target[:3]))
            if self.stage_step > 3000:
                print(f"  !! {self.stage.value} timed out at dist={dist:.3f}; advancing")
                return True
            return dist < EE_POS_TOL
        if self.mode == Mode.WAIT:
            return self.stage_step >= self.wait_steps
        return True

    # ------------------------------------------------------------------
    # EE targets derived from scene state
    # ------------------------------------------------------------------
    @staticmethod
    def _pose(pos: np.ndarray) -> np.ndarray:
        return np.concatenate([pos, [0.0, 0.0, 0.0, 1.0]])

    def _grasp_pose(self) -> np.ndarray:
        b = self._bottle_pose()
        # The IK target is the gripper base.  Align the pad center with the
        # bottle's mid-height so the two pads close around the bottle body.
        return self._pose(np.array([b[0], b[1], b[2] + BOTTLE_HEIGHT * 0.5 - GRIPPER_PAD_CENTER_Z]))

    def _lift_pose(self) -> np.ndarray:
        p = self._grasp_pose()
        p[2] += LIFT_DELTA_Z
        return p

    def _above_table_pose(self) -> np.ndarray:
        return self._pose(np.array([
            PLACE_WORLD_XY[0], PLACE_WORLD_XY[1],
            TABLE_SURFACE_Z + BOTTLE_HEIGHT * 0.5 + 0.15,
        ]))

    def _place_pose(self) -> np.ndarray:
        return self._pose(np.array([
            PLACE_WORLD_XY[0], PLACE_WORLD_XY[1],
            TABLE_SURFACE_Z + BOTTLE_HEIGHT * 0.5 + 0.02,
        ]))

    def _retreat_pose(self) -> np.ndarray:
        p = self._place_pose()
        p[2] += RETREAT_DELTA_Z
        return p

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def setup(self) -> None:
        # Fully zero everything while physics settles.
        self._command_waist_zero()
        self._command_right_arm_zero()
        self._command_qpos(HOME_QPOS)
        # Start with both grippers open.
        self.l_gripper_target = GRIPPER_OPEN
        self.r_gripper_target = GRIPPER_OPEN
        self._command_gripper()
        for _ in range(STARTUP_STEPS):
            step(self.model, self.data)
        print(f"startup done. left EE = {self._ee_pose()[:3]}")
        print(f"        bottle pose = {self._bottle_pose()[:3]}")
        self._enter_open_loop(Stage.READY, READY_QPOS)

    def _advance_stage(self) -> None:
        if not self._stage_done():
            return
        s = self.stage
        if s == Stage.READY:
            self._enter_closed_loop(Stage.MOVE_TO_GRASP, self._grasp_pose())
        elif s == Stage.MOVE_TO_GRASP:
            self._activate_grasp()
            self._enter_wait(Stage.CLOSE_GRASP, CLOSE_GRASP_STEPS)
        elif s == Stage.CLOSE_GRASP:
            self._enter_closed_loop(Stage.LIFT, self._lift_pose())
        elif s == Stage.LIFT:
            self._enter_open_loop(Stage.TUCK_1, TUCK_QPOS)
        elif s == Stage.TUCK_1:
            self._enter_open_loop(Stage.ROTATE_BODY, TUCK_PI_QPOS)
        elif s == Stage.ROTATE_BODY:
            self._enter_closed_loop(Stage.MOVE_ABOVE_TABLE, self._above_table_pose())
        elif s == Stage.MOVE_ABOVE_TABLE:
            self._enter_closed_loop(Stage.LOWER_TO_PLACE, self._place_pose())
        elif s == Stage.LOWER_TO_PLACE:
            self._deactivate_grasp()
            self._enter_wait(Stage.RELEASE, RELEASE_STEPS)
        elif s == Stage.RELEASE:
            self._enter_closed_loop(Stage.RETREAT, self._retreat_pose())
        elif s == Stage.RETREAT:
            self._enter_open_loop(Stage.TUCK_2, TUCK_PI_QPOS)
        elif s == Stage.TUCK_2:
            self._enter_open_loop(Stage.ROTATE_BACK, TUCK_QPOS)
        elif s == Stage.ROTATE_BACK:
            self._enter_open_loop(Stage.HOME, HOME_QPOS)
        elif s == Stage.HOME:
            self.stage = Stage.DONE
            print("[done] task finished")

    def phys_step(self) -> None:
        # Re-issue appropriate commands every step
        self._command_waist_zero()
        self._command_right_arm_zero()
        # Gripper ctrl must be re-issued every step so the contact / hold force is stable.
        self._command_gripper()

        if self.mode == Mode.OPEN_LOOP_QPOS and self.qpos_target is not None:
            self._command_qpos(self.qpos_target)
        elif self.mode == Mode.CLOSED_LOOP_IK and self.ee_target is not None:
            qpos = self._solve_ik_step(self.ee_target)
            self._command_qpos(qpos)
        # WAIT: keep previous ctrl values (don't re-issue arm)

        self.stage_step += 1
        self._advance_stage()
        self._update_grasp_follow()  # no-op unless --fake-grasp
        step(self.model, self.data)

    def render_step(self, render: RenderApp) -> None:
        if self.debug:
            gz = render.gizmos
            ee = self._ee_pose()
            gz.draw_sphere(0.035, ee[:3], color=Color.rgb(1.0, 0.4, 0.0))
            gz.draw_axes(ee[:3], ee[3:7], length=0.12)
            if self.ee_target is not None:
                t = self.ee_target
                gz.draw_sphere(0.04, t[:3], color=Color.rgb(0.2, 0.8, 1.0))
                gz.draw_line(ee[:3], t[:3], color=Color.rgb(1.0, 1.0, 0.0))
            b = self._bottle_pose()
            gz.draw_sphere(0.03, b[:3], color=Color.rgb(0.4, 1.0, 0.3))
        render.sync(self.data)

    def run(self) -> None:
        with RenderApp() as render:
            render.launch(self.model)
            self.setup()
            render_run.render_loop(
                self.model.options.timestep, 60,
                self.phys_step, lambda: self.render_step(render),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HRB3Q-LC grasp demo: pick bottle from shelf, rotate torso, place on table."
    )
    parser.add_argument("--model", default="xmls/hrb3q_grasp_demo.xml")
    parser.add_argument("--debug", action="store_true", help="draw EE and target gizmos")
    parser.add_argument(
        "--fake-grasp", action="store_true",
        help="fallback: teleport the bottle to follow the EE instead of using real gripper actuator",
    )
    args = parser.parse_args()
    HRB3QGraspDemo(
        model_path=args.model,
        debug=args.debug,
        fake_grasp=args.fake_grasp,
    ).run()


if __name__ == "__main__":
    main()

# Example:
#   DISPLAY=:1 conda run --live-stream -n motrix-lerobot python scripts/hrb3q_grasp_demo.py --debug
