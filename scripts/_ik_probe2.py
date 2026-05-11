"""Verify IK convergence for -Y side targets AFTER Body is manually rotated."""
from pathlib import Path
import numpy as np
from motrixsim import SceneData, load_model, ik, step

model = load_model(str(Path(__file__).resolve().parents[1] / "xmls/hrb3q_grasp_demo.xml"))
data = SceneData(model)

body_act = model.get_actuator("actuator_body")
act_l = [model.get_actuator(f"actuator_l{i}") for i in range(1, 8)]
act_r = [model.get_actuator(f"actuator_r{i}") for i in range(1, 8)]
waist_act = model.get_actuator("actuator_waist")

# Initial stabilize
for _ in range(200):
    step(model, data)

chain = ik.IkChain(model, end_link="arml7", start_link="Body",
                   end_effector_offset=np.array([0, 0, 0.08, 0, 0, 0, 1.0]))
solver = ik.DlsSolver(max_iter=400, step_size=0.5, tolerance=1e-3, damping=1e-3)

# Rotate body to pi
print("commanding Joint_Body -> pi ...")
waist_act.set_ctrl(data, 0.0)
body_act.set_ctrl(data, np.pi)
for a in act_l + act_r:
    a.set_ctrl(data, 0.0)
for _ in range(1500):
    step(model, data)

ee = np.array(chain.get_end_effector_pose(data))
print(f"EE after body rotated: pos={ee[:3]}, quat={ee[3:7]}")

targets = [
    ("above_table", np.array([ 0.2, -0.42, 1.08, ee[3], ee[4], ee[5], ee[6]])),
    ("place",       np.array([ 0.2, -0.42, 0.95, ee[3], ee[4], ee[5], ee[6]])),
    ("retreat",     np.array([ 0.2, -0.42, 1.10, ee[3], ee[4], ee[5], ee[6]])),
]
for name, tgt in targets:
    res = np.asarray(solver.solve(chain, data, tgt))
    print(f"  {name:12s}: iters={int(res[0]):4d}, residual={float(res[1]):.4f}")
