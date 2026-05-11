"""Test IK convergence for HRB3Q-LC left arm."""
from pathlib import Path
import numpy as np
from motrixsim import SceneData, load_model, ik, step

model = load_model(str(Path(__file__).resolve().parents[1] / "xmls/hrb3q_grasp_demo.xml"))
data = SceneData(model)
for _ in range(200):
    step(model, data)

chain = ik.IkChain(model, end_link="arml7", start_link="Body",
                   end_effector_offset=np.array([0, 0, 0.08, 0, 0, 0, 1.0]))

ee0 = np.array(chain.get_end_effector_pose(data))
print(f"initial EE: pos={ee0[:3]}, quat={ee0[3:7]}")

targets = [
    ("pregrasp",       np.array([-0.2, 0.46, 1.45,  ee0[3], ee0[4], ee0[5], ee0[6]])),
    ("grasp",          np.array([-0.2, 0.58, 1.451, ee0[3], ee0[4], ee0[5], ee0[6]])),
    ("lift",           np.array([-0.2, 0.58, 1.591, ee0[3], ee0[4], ee0[5], ee0[6]])),
    ("above_table",    np.array([ 0.2,-0.42, 1.08,  ee0[3], ee0[4], ee0[5], ee0[6]])),
    ("place",          np.array([ 0.2,-0.42, 0.95,  ee0[3], ee0[4], ee0[5], ee0[6]])),
]

for step_size in [0.3, 0.5, 0.8]:
    for max_iter in [200, 800]:
        solver = ik.DlsSolver(max_iter=max_iter, step_size=step_size, tolerance=1e-3, damping=1e-3)
        print(f"\n--- step_size={step_size}, max_iter={max_iter} ---")
        for name, tgt in targets:
            res = np.asarray(solver.solve(chain, data, tgt))
            print(f"  {name:16s}: iters={int(res[0]):4d}, residual={float(res[1]):.4f}")
