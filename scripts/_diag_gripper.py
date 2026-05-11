#!/usr/bin/env python3
"""Diagnose HRB3Q gripper controllability in Motrix simulation.

This script does NOT need a renderer / DISPLAY. It loads the given model,
lists all actuators / joints, probes common gripper actuator names, and
runs a closed-loop control test on the left gripper actuator group to verify:

  1. Whether the gripper actuator exists in the loaded model.
  2. Whether the actuator actually moves the finger joint(s).
  3. Whether the two remaining inner/root finger structures follow the same
     ctrl target.

Usage:
    conda run --live-stream -n motrix-lerobot \
        python scripts/_diag_gripper.py --model xmls/hrb3q_gripper_test.xml

    # 想检查 demo 用的 XML 是否带夹爪：
    conda run --live-stream -n motrix-lerobot \
        python scripts/_diag_gripper.py --model xmls/hrb3q_grasp_demo.xml

    # 想直接检查 URDF：
    conda run --live-stream -n motrix-lerobot \
        python scripts/_diag_gripper.py --model HRB3Q-LC/urdf/HRB3Q-LC.urdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

from motrixsim import SceneData, load_model, step


GRIPPER_ACTUATOR_CANDIDATES = [
    "actuator_l_gripper",
    "actuator_l_gripper_r",
    "actuator_r_gripper",
    "actuator_r_gripper_r",
    "mimic_relation_l",
    "mimic_relation_r",
    "l_gripper",
    "r_gripper",
    "gripper_l",
    "gripper_r",
    "L_gripper",
    "R_gripper",
]

GRIPPER_JOINT_CANDIDATES = [
    "Joint_l_gripper_l",
    "Joint_l_gripper_r",
    "Joint_r_gripper_l",
    "Joint_r_gripper_r",
    "l_gripper_finger_l",
    "l_gripper_finger_r",
]


def _safe_iter(model, attr_name: str):
    """Best-effort enumeration: try .actuators / .joints, fall back to indexed access."""
    container = getattr(model, attr_name, None)
    if container is None:
        return []
    try:
        return list(container)
    except TypeError:
        return []


def _list_items(model) -> None:
    print("\n=== All actuators ===")
    actuators = _safe_iter(model, "actuators")
    if not actuators:
        print("  (model.actuators not iterable; falling back to candidate probing only)")
    for a in actuators:
        name = getattr(a, "name", repr(a))
        print(f"  - {name}")

    print("\n=== All joints ===")
    joints = _safe_iter(model, "joints")
    if not joints:
        print("  (model.joints not iterable; falling back to candidate probing only)")
    for j in joints:
        name = getattr(j, "name", repr(j))
        print(f"  - {name}")


def _probe_names(model) -> None:
    print("\n=== Gripper actuator name probing ===")
    found_any = False
    for name in GRIPPER_ACTUATOR_CANDIDATES:
        a = model.get_actuator(name)
        mark = "FOUND" if a is not None else "None"
        if a is not None:
            found_any = True
        print(f"  actuator {name:<24} -> {mark}")

    print("\n=== Gripper joint name probing ===")
    for name in GRIPPER_JOINT_CANDIDATES:
        j = model.get_joint(name)
        mark = "FOUND" if j is not None else "None"
        print(f"  joint    {name:<24} -> {mark}")

    if not found_any:
        print("\n[!] 当前模型未发现任何已知的夹爪 actuator 名称。")
        print("    程序员调用的 set_ctrl 不会有任何效果（actuator 是 None）。")


def _run_control_test(model, data) -> None:
    l_actuators = [
        model.get_actuator(name)
        for name in (
            "actuator_l_gripper",
            "actuator_l_gripper_r",
        )
    ]
    l_actuators = [actuator for actuator in l_actuators if actuator is not None]
    if not l_actuators:
        print("\n[skip] actuator_l_gripper not present, control test skipped.")
        return

    tracked = {
        "left_inner": model.get_joint("Joint_l_gripper_l"),
        "right_inner": model.get_joint("Joint_l_gripper_r"),
    }

    print("\n=== Control response test on left gripper actuator group ===")
    print(f"  (driving {len(l_actuators)} actuators through 0 -> 0.04 -> 0 -> 0.04, 500 steps each)")
    header = "  ctrl     " + " ".join(f"{k:>10}" for k in tracked.keys())
    print(header)
    sequence = [0.0, 0.04, 0.0, 0.04]
    for target in sequence:
        for actuator in l_actuators:
            actuator.set_ctrl(data, float(target))
        for _ in range(500):
            step(model, data)
        readings = []
        for k, j in tracked.items():
            val = j.get_dof_pos(data)[0] if j is not None else float("nan")
            readings.append((k, val))
        row = f"  {target:.3f}   " + " ".join(f"{v:+10.4f}" for _, v in readings)
        print(row)

    print("\n判读规则：")
    print("  - 两个数值都跟随 ctrl 变化              -> 左右内部夹指结构同步正常")
    print("  - 只有一个数值跟随 ctrl                 -> 对侧 actuator 缺失或没有被脚本同步下发")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="xmls/hrb3q_gripper_test.xml",
        help="Path to the MJCF/URDF model file to diagnose.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise SystemExit(f"model file not found: {model_path}")

    print(f"[load] {model_path}")
    model = load_model(str(model_path))
    data = SceneData(model)

    _list_items(model)
    _probe_names(model)
    _run_control_test(model, data)

    print("\n[done]")


if __name__ == "__main__":
    main()
