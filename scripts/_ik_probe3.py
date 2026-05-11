"""Find a reasonable 'ready' pose for HRB3Q-LC left arm and test IK from it."""
from pathlib import Path
import numpy as np
from motrixsim import SceneData, load_model, ik, step

model = load_model(str(Path(__file__).resolve().parents[1] / "xmls/hrb3q_grasp_demo.xml"))
data = SceneData(model)

body_act = model.get_actuator("actuator_body")
acts_l = [model.get_actuator(f"actuator_l{i}") for i in range(1, 8)]
acts_r = [model.get_actuator(f"actuator_r{i}") for i in range(1, 8)]
waist_act = model.get_actuator("actuator_waist")

chain = ik.IkChain(model, end_link="arml7", start_link="Body",
                   end_effector_offset=np.array([0, 0, 0.08, 0, 0, 0, 1.0]))

# Settle at zero first
for _ in range(200):
    step(model, data)

# Try various ready poses
candidates = [
    ("zero",          [0.0] * 7),
    ("elbow_bend",    [0.0, -1.2, 0.0, -1.5, 0.0, 0.0, 0.0]),
    ("elbow_mirror",  [0.0,  1.2, 0.0,  1.5, 0.0, 0.0, 0.0]),
    ("tuck_fwd",      [1.0, -1.2, 0.0, -1.5, 0.0, 0.0, 0.0]),
    ("tuck_down",     [1.57, -1.0, 0.0, -1.2, 0.0, 0.0, 0.0]),
    ("shoulder_down", [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]

for name, qpos in candidates:
    # Reset to zero first
    for a in acts_l + acts_r:
        a.set_ctrl(data, 0.0)
    body_act.set_ctrl(data, 0.0)
    waist_act.set_ctrl(data, 0.0)
    for _ in range(400):
        step(model, data)
    # Now command target
    for a, q in zip(acts_l, qpos):
        a.set_ctrl(data, float(q))
    for _ in range(1500):
        step(model, data)
    ee = np.array(chain.get_end_effector_pose(data))
    print(f"{name:16s} -> EE pos = ({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})")
