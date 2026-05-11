import argparse
from pathlib import Path
import sys

import numpy as np

from motrixsim import SceneData, load_model, run, step
from motrixsim.render import Color, Layout, RenderApp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shelves_demo.controller import (
    RobotController,
    RobotControllerError,
    RobotPhase,
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


def _setup_camera_viewports(model, *, width: int = 320, height: int = 240) -> None:
    """Configure model cameras for rendering (call before render.launch)."""
    cameras = [c for c in model.cameras if "eye" not in c.name]
    if not cameras:
        print("No cameras found in model - skipping camera viewports")
        return
    for cam in cameras:
        cam.set_render_target("image", width, height)
    print(f"Camera viewports: {[c.name for c in cameras]}")


def _create_camera_viewports(model, render, *, width: int = 320, height: int = 240) -> None:
    """Create on-screen camera viewport widgets (call after render.launch)."""
    cameras = [c for c in model.cameras if "eye" not in c.name]
    if not cameras:
        return
    for i, cam in enumerate(cameras):
        layout = Layout(left=10 + i * (width + 10), top=10, width=width, height=height)
        render.widgets.create_camera_viewport(cam, layout=layout, sim_world_index=0)


def make_random_base_target(
    current_pose: np.ndarray,
    rng: np.random.Generator,
    *,
    xy_half_range: float = 5.0,
    min_distance: float = 0.5,
    max_attempts: int = 200,
) -> tuple[float, float, float]:
    current_xy = np.array(current_pose[:2], dtype=np.float64)
    for _ in range(max_attempts):
        target_xy = rng.uniform(-xy_half_range, xy_half_range, size=2)
        if np.linalg.norm(target_xy - current_xy) < min_distance:
            continue
        target_heading = float(rng.uniform(-np.pi, np.pi))
        return float(target_xy[0]), float(target_xy[1]), target_heading
    current_heading = _heading_from_body_yaw(_yaw_from_quat(current_pose[3:7]))
    return float(current_xy[0]), float(current_xy[1]), current_heading


def make_dual_arm_target_candidates(
    controller: RobotController,
    data: SceneData,
    rng: np.random.Generator,
    *,
    range_x: float,
    range_y: float,
    range_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_pose = controller.get_end_effector_pose(data, "left")
    right_pose = controller.get_end_effector_pose(data, "right")

    offset_x = rng.uniform(-range_x, range_x)
    offset_y = rng.uniform(-range_y, range_y)
    offset_z = rng.uniform(-range_z, range_z)

    left_target = left_pose.copy()
    right_target = right_pose.copy()
    left_target[:3] += np.array([offset_x, offset_y, offset_z], dtype=np.float64)
    right_target[:3] += np.array([offset_x, -offset_y, offset_z], dtype=np.float64)
    return left_target, right_target


def make_single_arm_target_candidate(
    controller: RobotController,
    data: SceneData,
    arm: str,
    rng: np.random.Generator,
    *,
    range_x: float,
    range_y: float,
    range_z: float,
) -> np.ndarray:
    target = controller.get_end_effector_pose(data, arm).copy()
    target[:3] += rng.uniform(
        low=np.array([-range_x, -range_y, -range_z], dtype=np.float64),
        high=np.array([range_x, range_y, range_z], dtype=np.float64),
    )
    return target


def run_base_demo(model_path: str, max_loop: int) -> None:
    model = load_model(model_path)
    _setup_camera_viewports(model)

    with RenderApp() as render:
        render.launch(model)
        _create_camera_viewports(model, render)
        data = SceneData(model)
        controller = RobotController(model)
        rng = np.random.default_rng(0)
        controller.initialize(data)

        for _ in range(400):
            step(model, data)

        completed_loops = 0
        active_target: tuple[float, float, float] | None = None
        last_phase = None
        last_detail = None

        def schedule_next_target() -> None:
            nonlocal completed_loops, active_target
            if completed_loops >= max_loop:
                raise SystemExit(0)
            active_target = make_random_base_target(controller.get_base_pose(data), rng)
            plan = controller.move_base_world(data, *active_target, drive_direction="forward")
            completed_loops += 1
            print("=" * 60)
            print(f"Base target {completed_loops}/{max_loop}")
            print(f"target xy:      ({plan.target.x:.3f}, {plan.target.y:.3f})")
            print(f"path heading:   {plan.path_heading:.3f}")
            print(f"target heading: {plan.target.heading:.3f}")
            print("sequence: rotate -> move -> final rotate")
            print("=" * 60)

        schedule_next_target()
        gizmos = render.gizmos

        def phys_step() -> None:
            nonlocal last_phase, last_detail
            status = controller.update(data)
            if status.phase != last_phase or status.detail != last_detail:
                detail = f" ({status.detail})" if status.detail else ""
                print(f"phase -> {status.phase.value}{detail}")
                last_phase = status.phase
                last_detail = status.detail
            if status.phase == RobotPhase.ERROR:
                print(f"controller error: {status.error or 'base demo entered error state'}")
                raise SystemExit(1)
            if status.phase == RobotPhase.DONE:
                last_phase = None
                last_detail = None
                schedule_next_target()
            step(model, data)

        def render_step() -> None:
            if active_target is not None:
                base_pose = controller.get_base_pose(data)
                current_pos = np.array(base_pose[:3], dtype=np.float64)
                target_pos = np.array([active_target[0], active_target[1], current_pos[2]], dtype=np.float64)
                target_quat = _quat_from_yaw(_body_yaw_from_heading(active_target[2]))
                gizmos.draw_sphere(0.04, target_pos, color=Color.rgb(1.0, 0.0, 0.0))
                gizmos.draw_axes(target_pos, target_quat, length=0.18)
                gizmos.draw_line(current_pos, target_pos, color=Color.rgb(1.0, 1.0, 0.0))
            render.sync(data)

        run.render_loop(model.options.timestep, 60, phys_step, render_step)


def run_dual_arm_demo(model_path: str, max_loop: int, range_x: float, range_y: float, range_z: float) -> None:
    model = load_model(model_path)
    _setup_camera_viewports(model)

    with RenderApp() as render:
        render.launch(model)
        _create_camera_viewports(model, render)
        data = SceneData(model)
        controller = RobotController(model)
        rng = np.random.default_rng(0)
        controller.initialize(data)

        for _ in range(400):
            step(model, data)

        left_target = None
        right_target = None
        completed_loops = 0
        last_phase = None
        last_detail = None

        def schedule_next_target() -> None:
            nonlocal left_target, right_target, completed_loops
            if completed_loops >= max_loop:
                raise SystemExit(0)

            for _ in range(200):
                candidate_left, candidate_right = make_dual_arm_target_candidates(
                    controller,
                    data,
                    rng,
                    range_x=range_x,
                    range_y=range_y,
                    range_z=range_z,
                )
                try:
                    plan = controller.move_dual_arm(
                        data,
                        candidate_left,
                        candidate_right,
                        constrain_orientation=False,
                    )
                    left_target = candidate_left
                    right_target = candidate_right
                    completed_loops += 1
                    print("=" * 60)
                    print(f"Dual-arm target {completed_loops}/{max_loop}")
                    print(f"lifter target z: {plan.lifter_target_z:.3f}")
                    print(f"left residual:   {plan.left_residual:.6f}")
                    print(f"right residual:  {plan.right_residual:.6f}")
                    print(f"sample range:    +/-({range_x:.3f}, {range_y:.3f}, {range_z:.3f}) m")
                    print("=" * 60)
                    return
                except RobotControllerError:
                    continue

            raise RuntimeError("failed to sample a reachable dual-arm target")

        schedule_next_target()
        gizmos = render.gizmos

        def phys_step() -> None:
            nonlocal last_phase, last_detail
            status = controller.update(data)
            if status.phase != last_phase or status.detail != last_detail:
                detail = f" ({status.detail})" if status.detail else ""
                print(f"phase -> {status.phase.value}{detail}")
                last_phase = status.phase
                last_detail = status.detail
            if status.phase == RobotPhase.ERROR:
                print(f"controller error: {status.error or 'dual-arm demo entered error state'}")
                raise SystemExit(1)
            if status.phase == RobotPhase.DONE:
                last_phase = None
                last_detail = None
                schedule_next_target()
            step(model, data)

        def render_step() -> None:
            if left_target is not None and right_target is not None:
                left_current = controller.get_end_effector_pose(data, "left")
                right_current = controller.get_end_effector_pose(data, "right")
                gizmos.draw_cuboid(np.array([0.06, 0.06, 0.06]), left_target[:3], rot=left_target[3:7])
                gizmos.draw_axes(left_target[:3], left_target[3:7], length=0.12)
                gizmos.draw_cuboid(np.array([0.06, 0.06, 0.06]), right_target[:3], rot=right_target[3:7])
                gizmos.draw_axes(right_target[:3], right_target[3:7], length=0.12)
                gizmos.draw_sphere(0.03, left_current[:3], color=Color.rgb(0.0, 1.0, 0.0))
                gizmos.draw_sphere(0.03, right_current[:3], color=Color.rgb(0.0, 1.0, 0.0))
                gizmos.draw_sphere(0.025, left_target[:3], color=Color.rgb(1.0, 0.0, 0.0))
                gizmos.draw_sphere(0.025, right_target[:3], color=Color.rgb(1.0, 0.0, 0.0))
                gizmos.draw_line(left_current[:3], left_target[:3], color=Color.rgb(1.0, 1.0, 0.0))
                gizmos.draw_line(right_current[:3], right_target[:3], color=Color.rgb(1.0, 1.0, 0.0))
            render.sync(data)

        run.render_loop(model.options.timestep, 60, phys_step, render_step)


def run_single_arm_demo(
    model_path: str,
    max_loop: int,
    range_x: float,
    range_y: float,
    range_z: float,
    arm_mode: str,
) -> None:
    model = load_model(model_path)
    _setup_camera_viewports(model)

    with RenderApp() as render:
        render.launch(model)
        _create_camera_viewports(model, render)
        data = SceneData(model)
        controller = RobotController(model)
        rng = np.random.default_rng(0)
        controller.initialize(data)

        for _ in range(400):
            step(model, data)

        active_arm = None
        target_pose = None
        completed_loops = 0
        last_phase = None
        last_detail = None

        def pick_arm(loop_index: int) -> str:
            if arm_mode in {"left", "right"}:
                return arm_mode
            return "left" if loop_index % 2 == 0 else "right"

        def schedule_next_target() -> None:
            nonlocal active_arm, target_pose, completed_loops
            if completed_loops >= max_loop:
                raise SystemExit(0)

            next_arm = pick_arm(completed_loops)
            for _ in range(200):
                candidate = make_single_arm_target_candidate(
                    controller,
                    data,
                    next_arm,
                    rng,
                    range_x=range_x,
                    range_y=range_y,
                    range_z=range_z,
                )
                try:
                    plan = controller.move_single_arm(
                        data,
                        arm=next_arm,
                        target_pose=candidate,
                        constrain_orientation=False,
                    )
                    active_arm = next_arm
                    target_pose = candidate
                    completed_loops += 1
                    residual = plan.left_residual if next_arm == "left" else plan.right_residual
                    print("=" * 60)
                    print(f"Single-arm target {completed_loops}/{max_loop}")
                    print(f"arm:             {next_arm}")
                    print(f"lifter target z: {plan.lifter_target_z:.3f}")
                    print(f"residual:        {residual:.6f}")
                    print(f"sample range:    +/-({range_x:.3f}, {range_y:.3f}, {range_z:.3f}) m")
                    print("=" * 60)
                    return
                except RobotControllerError:
                    continue

            raise RuntimeError(f"failed to sample a reachable single-arm target for {next_arm}")

        schedule_next_target()
        gizmos = render.gizmos

        def phys_step() -> None:
            nonlocal last_phase, last_detail
            status = controller.update(data)
            if status.phase != last_phase or status.detail != last_detail:
                detail = f" ({status.detail})" if status.detail else ""
                print(f"phase -> {status.phase.value}{detail}")
                last_phase = status.phase
                last_detail = status.detail
            if status.phase == RobotPhase.ERROR:
                print(f"controller error: {status.error or 'single-arm demo entered error state'}")
                raise SystemExit(1)
            if status.phase == RobotPhase.DONE:
                last_phase = None
                last_detail = None
                schedule_next_target()
            step(model, data)

        def render_step() -> None:
            if active_arm is not None and target_pose is not None:
                current_pose = controller.get_end_effector_pose(data, active_arm)
                color = Color.rgb(1.0, 0.4, 0.0) if active_arm == "left" else Color.rgb(0.2, 0.6, 1.0)
                gizmos.draw_cuboid(np.array([0.06, 0.06, 0.06]), target_pose[:3], rot=target_pose[3:7])
                gizmos.draw_axes(target_pose[:3], target_pose[3:7], length=0.12)
                gizmos.draw_sphere(0.03, current_pose[:3], color=Color.rgb(0.0, 1.0, 0.0))
                gizmos.draw_sphere(0.025, target_pose[:3], color=color)
                gizmos.draw_line(current_pose[:3], target_pose[:3], color=Color.rgb(1.0, 1.0, 0.0))
            render.sync(data)

        run.render_loop(model.options.timestep, 60, phys_step, render_step)


def main(mode: str, max_loop: int, range_x: float, range_y: float, range_z: float, model_path: str, arm: str) -> None:
    if mode == "base":
        run_base_demo(model_path, max_loop=max_loop)
        return
    if mode == "dual_arm":
        run_dual_arm_demo(model_path, max_loop=max_loop, range_x=range_x, range_y=range_y, range_z=range_z)
        return
    if mode == "single_arm":
        run_single_arm_demo(
            model_path,
            max_loop=max_loop,
            range_x=range_x,
            range_y=range_y,
            range_z=range_z,
            arm_mode=arm,
        )
        return
    raise ValueError(f"unsupported mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shelf handling unified controller demos")
    parser.add_argument("--mode", choices=["base", "dual_arm", "single_arm"], default="base", help="which controller mode to demo")
    parser.add_argument("--arm", choices=["left", "right", "alternate"], default="alternate", help="single-arm demo arm selection")
    parser.add_argument("--model", default="xmls/clean_world.xml", help="model xml path")
    parser.add_argument("--max-loop", type=int, default=20, help="number of targets to execute")
    parser.add_argument("--range-x", type=float, default=0.20, help="arm demo random target half-range along x in meters")
    parser.add_argument("--range-y", type=float, default=0.15, help="arm demo random target half-range along y in meters")
    parser.add_argument("--range-z", type=float, default=0.15, help="arm demo random target half-range along z in meters")
    args = parser.parse_args()
    main(
        mode=args.mode,
        max_loop=args.max_loop,
        range_x=args.range_x,
        range_y=args.range_y,
        range_z=args.range_z,
        model_path=args.model,
        arm=args.arm,
    )
    

# 示例命令
# DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/controller_demo.py --mode base
# DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/controller_demo.py --mode dual_arm
# DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/controller_demo.py --mode single_arm --arm alternate
# DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/controller_demo.py --mode single_arm --arm left
