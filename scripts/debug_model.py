#!/usr/bin/env python3
"""Debug script to check when NaN first appears."""
from motrixsim import SceneData, load_model, step
import math

xml_path = "xmls/hrb3q_dof_demo.xml"
print(f"Model: {xml_path}")
model = load_model(xml_path)
data = SceneData(model)

# Set all actuators to 0 (hold position)
for a in model.actuators:
    a.set_ctrl(data, 0.0)

print("Checking for NaN step by step...")
for i in range(500):
    step(model, data)
    if (i + 1) % 10 == 0:
        positions = [j.get_dof_pos(data) for j in model.joints]
        has_nan = any(math.isnan(float(p)) for pos in positions for p in pos)
        if has_nan:
            print(f"  Step {i+1}: NaN detected!")
            for j in model.joints:
                pos = j.get_dof_pos(data)
                print(f"    {j.name}: {pos}")
            break
        else:
            max_pos = max(abs(float(p)) for pos in positions for p in pos)
            print(f"  Step {i+1}: OK (max |pos| = {max_pos:.6f})")

print("\nDone.")
