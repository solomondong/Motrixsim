"""One-off probe: load the HRB3Q-LC grasp scene, print key body poses and EE pose."""
from pathlib import Path
import numpy as np
from motrixsim import SceneData, load_model, ik, step

path = str(Path(__file__).resolve().parents[1] / "xmls/hrb3q_grasp_demo.xml")
print("loading", path)
model = load_model(path)
data = SceneData(model)

for _ in range(200):
    step(model, data)

for name in ["Origin_base", "Waist", "Body", "arml1", "arml7", "armr7", "Sh_/Shelf_Single", "Tb_/table", "Bt_/001_bottle_base2"]:
    b = model.get_body(name)
    if b is None:
        print(f"  body '{name}': NOT FOUND")
        continue
    pose = b.get_pose(data)
    print(f"  body '{name}': pos=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) quat=({pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f},{pose[6]:.3f})")

print("\nActuators:")
for name in ["actuator_waist", "actuator_body", "actuator_l1", "actuator_r1"]:
    a = model.get_actuator(name)
    print(f"  {name}: {a is not None}")

print("\nIK chain left (Body -> arml7, offset (0,0,0.08)):")
try:
    chain = ik.IkChain(model, end_link="arml7", start_link="Body",
                       end_effector_offset=np.array([0, 0, 0.08, 0, 0, 0, 1.0]))
    ee = chain.get_end_effector_pose(data)
    print(f"  EE pose: pos=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) quat=({ee[3]:.3f},{ee[4]:.3f},{ee[5]:.3f},{ee[6]:.3f})")
    print(f"  num_dof_pos: {chain.num_dof_pos}")
except Exception as exc:
    print(f"  left chain error: {exc}")

print("\nIK chain right (Body -> armr7):")
try:
    chain = ik.IkChain(model, end_link="armr7", start_link="Body",
                       end_effector_offset=np.array([0, 0, 0.08, 0, 0, 0, 1.0]))
    ee = chain.get_end_effector_pose(data)
    print(f"  EE pose: pos=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) quat=({ee[3]:.3f},{ee[4]:.3f},{ee[5]:.3f},{ee[6]:.3f})")
except Exception as exc:
    print(f"  right chain error: {exc}")
