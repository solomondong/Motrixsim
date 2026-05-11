"""Headless driver for the grasp demo (no rendering) to validate the stage logic."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.hrb3q_grasp_demo import HRB3QGraspDemo, Stage

demo = HRB3QGraspDemo(model_path="xmls/hrb3q_grasp_demo.xml", debug=False)
demo.setup()

from motrixsim import step
MAX_STEPS = 60000
for i in range(MAX_STEPS):
    demo.phys_step()
    if demo.stage == Stage.DONE:
        print(f"finished in {i} steps")
        break
else:
    print(f"did not finish after {MAX_STEPS} steps, stage={demo.stage.value}")
