#!/usr/bin/env python3
"""
Test HRB3Q-LC DH parallel gripper movement in Motrix simulator.

This script demonstrates the DH parallel gripper open/close motion on both arms.
The grippers cycle through: fully open -> fully closed -> partially open patterns.

Usage:
    DISPLAY=:1 conda run --live-stream -n motrix-lerobot python scripts/test_gripper.py
"""

import argparse
import math
from pathlib import Path

from motrixsim import SceneData, load_model, run, step
from motrixsim.render import RenderApp


# Gripper parameters (DH parallel gripper)
GRIPPER_OPEN = 0.04       # 40mm per finger = 80mm total opening
GRIPPER_CLOSED = 0.0      # Fully closed
GRIPPER_HALF_OPEN = 0.02  # 20mm per finger = 40mm total opening


def test_gripper(model_path: str) -> None:
    """Load model and test DH parallel gripper open/close movement."""
    abs_model_path = str(Path(model_path).resolve())
    
    print(f"Loading model from: {abs_model_path}")
    model = load_model(abs_model_path)
    data = SceneData(model)
    
    print("Model loaded successfully")
    print(f"Bodies: {len(list(model.bodies))}")
    print(f"Joints: {len(list(model.joints))}")
    
    # Get arm actuators (to hold a fixed pose)
    arm_actuator_names = [
        "actuator_waist", "actuator_body",
        "actuator_l1", "actuator_l2", "actuator_l3", "actuator_l4",
        "actuator_l5", "actuator_l6", "actuator_l7",
        "actuator_r1", "actuator_r2", "actuator_r3", "actuator_r4",
        "actuator_r5", "actuator_r6", "actuator_r7",
    ]
    
    arm_actuators = []
    for name in arm_actuator_names:
        actuator = model.get_actuator(name)
        if actuator is not None:
            arm_actuators.append((name, actuator))
        else:
            print(f"  Warning: actuator '{name}' not found")
    
    # Get gripper actuators
    l_gripper = model.get_actuator("actuator_l_gripper")
    r_gripper = model.get_actuator("actuator_r_gripper")
    
    if l_gripper is None or r_gripper is None:
        print("ERROR: Gripper actuators not found!")
        return
    
    print(f"\nGripper actuators found:")
    print(f"  Left gripper:  actuator_l_gripper")
    print(f"  Right gripper: actuator_r_gripper")
    print(f"\nDH Parallel Gripper specs:")
    print(f"  Stroke per finger: 40mm")
    print(f"  Total opening:     80mm")
    print(f"  Control range:     [{GRIPPER_CLOSED}, {GRIPPER_OPEN}] m")
    
    # Arm pose: extend arms forward for visibility
    # Left arm: reach forward (-Y direction in local frame)
    arm_targets = {
        "actuator_waist": 0.0,
        "actuator_body": 0.0,
        "actuator_l1": 0.0,
        "actuator_l2": -1.2,
        "actuator_l3": 0.0,
        "actuator_l4": -1.0,
        "actuator_l5": 0.0,
        "actuator_l6": -0.5,
        "actuator_l7": 0.0,
        "actuator_r1": 0.0,
        "actuator_r2": 1.2,
        "actuator_r3": 0.0,
        "actuator_r4": 1.0,
        "actuator_r5": 0.0,
        "actuator_r6": -0.5,
        "actuator_r7": 0.0,
    }
    
    # Gripper test sequence: (description, left_target, right_target, duration_sec)
    test_sequence = [
        ("Both grippers FULLY OPEN", GRIPPER_OPEN, GRIPPER_OPEN, 2.0),
        ("Both grippers FULLY CLOSED", GRIPPER_CLOSED, GRIPPER_CLOSED, 2.0),
        ("Both grippers HALF OPEN", GRIPPER_HALF_OPEN, GRIPPER_HALF_OPEN, 2.0),
        ("Left OPEN, Right CLOSED", GRIPPER_OPEN, GRIPPER_CLOSED, 2.0),
        ("Left CLOSED, Right OPEN", GRIPPER_CLOSED, GRIPPER_OPEN, 2.0),
        ("Sinusoidal cycling (both)", None, None, 6.0),  # Special: continuous cycling
    ]
    
    with RenderApp() as render:
        render.launch(model)
        
        # Set arm pose and run stabilization
        print("\nSetting arm pose and stabilizing...")
        for name, actuator in arm_actuators:
            target = arm_targets.get(name, 0.0)
            actuator.set_ctrl(data, target)
        
        # Start with grippers open
        l_gripper.set_ctrl(data, GRIPPER_OPEN)
        r_gripper.set_ctrl(data, GRIPPER_OPEN)
        
        for _ in range(1000):
            step(model, data)
        
        print("\n=== DH Parallel Gripper Test ===")
        print("Testing gripper open/close movement on both arms.")
        print("Each gripper has two parallel fingers controlled by slide joints.\n")
        
        dt = model.options.timestep
        sim_time = [0.0]
        seq_idx = [0]
        seq_start_time = [0.0]
        last_seq_idx = [-1]
        
        # Compute total duration
        total_duration = sum(s[3] for s in test_sequence)
        
        def phys_step() -> None:
            """Physics step: drive gripper test sequence."""
            sim_time[0] += dt
            
            # Determine which sequence step we're in
            cycle_time = sim_time[0] % total_duration
            elapsed = 0.0
            current_idx = 0
            for i, (_, _, _, dur) in enumerate(test_sequence):
                if cycle_time < elapsed + dur:
                    current_idx = i
                    break
                elapsed += dur
            else:
                current_idx = len(test_sequence) - 1
            
            seq_idx[0] = current_idx
            t_in_step = cycle_time - elapsed
            
            # Print when switching steps
            if seq_idx[0] != last_seq_idx[0]:
                last_seq_idx[0] = seq_idx[0]
                desc = test_sequence[current_idx][0]
                if current_idx == 0 and sim_time[0] > total_duration:
                    print("\n--- Restarting test cycle ---")
                print(f"  [{current_idx + 1}/{len(test_sequence)}] {desc}")
            
            # Hold arm pose
            for name, actuator in arm_actuators:
                target = arm_targets.get(name, 0.0)
                actuator.set_ctrl(data, target)
            
            # Set gripper commands
            _, l_target, r_target, duration = test_sequence[current_idx]
            
            if l_target is not None:
                # Fixed target: smoothly interpolate
                l_gripper.set_ctrl(data, l_target)
                r_gripper.set_ctrl(data, r_target)
            else:
                # Sinusoidal cycling mode
                phase = t_in_step / duration * 2 * math.pi * 2  # 2 full cycles
                l_val = (math.sin(phase) + 1.0) * 0.5 * GRIPPER_OPEN
                r_val = (math.cos(phase) + 1.0) * 0.5 * GRIPPER_OPEN
                l_gripper.set_ctrl(data, l_val)
                r_gripper.set_ctrl(data, r_val)
            
            step(model, data)
        
        def render_step() -> None:
            """Render step callback."""
            render.sync(data)
        
        print(f"\nRunning gripper test. Close the window to exit.")
        print(f"Total cycle duration: {total_duration:.1f}s\n")
        run.render_loop(dt, 60, phys_step, render_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test HRB3Q-LC DH parallel gripper movement")
    parser.add_argument(
        "--model",
        default="xmls/hrb3q_gripper_test.xml",
        help="Path to the gripper test model XML file"
    )
    args = parser.parse_args()
    
    test_gripper(args.model)


if __name__ == "__main__":
    main()
