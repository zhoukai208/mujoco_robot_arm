import yaml
import numpy as np
import mujoco
import cv2
import pinocchio
from scipy.spatial.transform import Rotation as R
from arm_base import ArmBaseViewer
from utils import *
from pupil_apriltags import Detector


# ====================== 主程序：机械臂IBVS伺服控制 ======================
class ArmIBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)

    def runBefore(self):
        super().runBefore()

    def runFunc(self):
        frame = self.get_camera_image(show=True)


if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_with_apriltag2.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
    
    robot = ArmIBVS(SCENE_XML_PATH, SCENE_XML_PATH, YAML_PATH)
    robot.run_loop()