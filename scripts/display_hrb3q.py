#!/usr/bin/env python3
"""
Display HRB3Q-LC robot model in Motrix simulator.

Usage:
    DISPLAY=:1 conda run --live-stream -n MotrixLab python scripts/display_hrb3q.py
"""

import argparse
from pathlib import Path

from motrixsim import SceneData, load_model, run, step
from motrixsim.render import RenderApp


def display_hrb3q(model_path: str) -> None:
    """Load and display HRB3Q-LC model in the simulator."""
    # Convert to absolute path
    abs_model_path = str(Path(model_path).resolve())
    
    # Load model
    print(f"Loading model from: {abs_model_path}")
    model = load_model(abs_model_path)
    data = SceneData(model)
    
    print("Model loaded successfully")
    print(f"Bodies: {len(list(model.bodies))}")
    print(f"Joints: {len(list(model.joints))}")
    
    # Run simulation with rendering
    with RenderApp() as render:
        render.launch(model)
        
        # Initial stabilization steps
        print("Running initial simulation steps...")
        for _ in range(100):
            step(model, data)
        
        print("Model displayed. Close the window to exit.")
        
        def phys_step() -> None:
            step(model, data)
        
        def render_step() -> None:
            render.sync(data)
        
        run.render_loop(model.options.timestep, 60, phys_step, render_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display HRB3Q-LC robot in Motrix simulator")
    parser.add_argument(
        "--model",
        default="xmls/hrb3q_demo.xml",
        help="Path to the HRB3Q-LC model XML file"
    )
    args = parser.parse_args()
    
    display_hrb3q(args.model)


if __name__ == "__main__":
    main()
