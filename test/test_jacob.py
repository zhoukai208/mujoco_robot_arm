
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from control.arm_base import ArmBaseViewer
from utils import *


class TestJacob(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path


    def runBefore(self):
        pass

    def runFunc(self):
        q = self.data.qpos[:7].copy()
        print(self.kinematics.J(q))

if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_pos.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
    robot = TestJacob(SCENE_XML_PATH, YAML_PATH, YAML_PATH)
    robot.run_loop()