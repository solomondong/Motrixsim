#!/usr/bin/env python3
"""
Display HRB3Q-LC robot degrees of freedom in Motrix simulator.

This script demonstrates each joint's range of motion with the robot base on the ground.

Usage:
    DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/display_hrb3q_dof.py
"""

import argparse
import math
from pathlib import Path

from motrixsim import SceneData, load_model, run, step
from motrixsim.render import RenderApp


def display_dof_demo(model_path: str) -> None:
    """Load and display HRB3Q-LC model with DOF demonstration."""
    abs_model_path = str(Path(model_path).resolve())
    
    # Load model
    print(f"Loading model from: {abs_model_path}")
    model = load_model(abs_model_path)
    data = SceneData(model)
    
    print("Model loaded successfully")
    print(f"Bodies: {len(list(model.bodies))}")
    print(f"Joints: {len(list(model.joints))}")
    
    # Get actuators
    actuator_names = [
        "actuator_waist", "actuator_body",
        "actuator_l1", "actuator_l2", "actuator_l3", "actuator_l4", "actuator_l5", "actuator_l6", "actuator_l7",
        "actuator_r1", "actuator_r2", "actuator_r3", "actuator_r4", "actuator_r5", "actuator_r6", "actuator_r7",
    ]
    
    actuators = []
    for name in actuator_names:
        actuator = model.get_actuator(name)
        if actuator is not None:
            actuators.append(actuator)
            print(f"  Found actuator: {name}")
        else:
            print(f"  Warning: actuator '{name}' not found")
    
    print(f"\nTotal actuators: {len(actuators)}")
    
    # Joint descriptions for display
    joint_descriptions = [
        "Waist (slide joint - torso lift)",
        "Body (hinge joint - torso rotation)",
        "Left Arm Joint 1 (shoulder rotation)",
        "Left Arm Joint 2 (shoulder pitch)",
        "Left Arm Joint 3 (shoulder roll)",
        "Left Arm Joint 4 (elbow)",
        "Left Arm Joint 5 (forearm rotation)",
        "Left Arm Joint 6 (wrist pitch)",
        "Left Arm Joint 7 (wrist roll)",
        "Right Arm Joint 1 (shoulder rotation)",
        "Right Arm Joint 2 (shoulder pitch)",
        "Right Arm Joint 3 (shoulder roll)",
        "Right Arm Joint 4 (elbow)",
        "Right Arm Joint 5 (forearm rotation)",
        "Right Arm Joint 6 (wrist pitch)",
        "Right Arm Joint 7 (wrist roll)",
    ]
    
    # Animation parameters
    JOINT_DISPLAY_DURATION = 3.0  # seconds per joint
    NUM_ACTUATORS = len(actuators)
    TOTAL_CYCLE_TIME = JOINT_DISPLAY_DURATION * NUM_ACTUATORS

    # Run simulation with rendering
    with RenderApp() as render:
        render.launch(model)
        
        # Initial stabilization steps
        print("\nRunning initial simulation steps...")
        for _ in range(500):
            step(model, data)
        
        print("\n=== Robot Degrees of Freedom Demo ===")
        print("The robot base is fixed on the ground.")
        print("Each joint will demonstrate its range of motion.\n")
        
        sim_time = [0.0]
        last_joint_idx = [-1]
        dt = model.options.timestep
        
        def phys_step() -> None:
            """Physics step callback - drives joint animation based on simulation time."""
            sim_time[0] += dt
            t_in_cycle = sim_time[0] % TOTAL_CYCLE_TIME
            joint_idx = int(t_in_cycle / JOINT_DISPLAY_DURATION)
            
            if joint_idx >= NUM_ACTUATORS:
                joint_idx = NUM_ACTUATORS - 1
            
            # Print when switching to a new joint
            if joint_idx != last_joint_idx[0]:
                last_joint_idx[0] = joint_idx
                if joint_idx == 0 and sim_time[0] > JOINT_DISPLAY_DURATION:
                    print("\n=== Starting new cycle ===")
                desc = joint_descriptions[joint_idx] if joint_idx < len(joint_descriptions) else f"Joint {joint_idx}"
                print(f"Demonstrating: {desc}")
            
            # Time within current joint's display window
            t_in_joint = t_in_cycle - joint_idx * JOINT_DISPLAY_DURATION
            
            # Sinusoidal motion: two full oscillations per display window
            # Waist (slide joint) gets smaller amplitude
            if joint_idx == 0:
                amplitude = 0.25
            else:
                amplitude = math.pi * 0.5
            angle = math.sin(t_in_joint / JOINT_DISPLAY_DURATION * 2 * math.pi * 2) * amplitude
            
            # Set control for the target joint, keep others at zero
            for i, actuator in enumerate(actuators):
                if i == joint_idx:
                    actuator.set_ctrl(data, angle)
                else:
                    actuator.set_ctrl(data, 0.0)
            
            # Step simulation after setting controls
            step(model, data)
        
        def render_step() -> None:
            """Render step callback."""
            render.sync(data)
        
        print(f"\nModel displayed. Close the window to exit.")
        run.render_loop(dt, 60, phys_step, render_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display HRB3Q-LC robot degrees of freedom")
    parser.add_argument(
        "--model",
        default="xmls/hrb3q_dof_demo.xml",
        help="Path to the HRB3Q-LC DOF model XML file"
    )
    args = parser.parse_args()
    
    display_dof_demo(args.model)


if __name__ == "__main__":
    main()
