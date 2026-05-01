import yaml
import numpy as np
import mujoco
from pynput.keyboard import Key, Listener
import pynput.keyboard as pkb
from arm_base import ArmBaseViewer
from utils import *


class ArmPBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 多点轨迹（原有）
        self.target_poses_list = []
        self.trajectory_list = []
        self.current_target_idx = 0
        self.traj_idx = 0

        # ===================== AprilTag 控制参数 =====================
        self.tag_pos = np.array([0.5, 0.0, 0.05])
        self.tag_rot = np.array([0.0, 0.0, 0.0])
        self.STEP_TRANS = 0.005
        self.STEP_ROT = 0.015
        self.tag_body_id = None

    def runBefore(self):
        super().runBefore()
        self.data.qpos[:7] = self.initial_pos[:7]

    def runFunc(self):

        self.mocaps.set_pose("apriltag_0", [0.6, -0.4, 0.0])
        self.get_camera_image(show=True)
        self.print_counter += 1
        if self.print_counter % 50 == 0:
            print(f"执行点: {self.current_target_idx+1}/{len(self.target_poses_list)} | 标签位置: {self.tag_pos.round(3)}")


if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_with_apriltag.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
    
    # 修复构造函数参数
    robot = ArmMoveController(SCENE_XML_PATH, SCENE_XML_PATH, YAML_PATH)
    robot.run_loop()