import sys
import mujoco
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from mujoco_viewer import CustomViewer
from traj_planner import MoveLPlanner, MoveJPlanner
from mocap_manager import MocapManager
from pinocchio_kinematic import PandaKinematics
from utils import *

class ArmBaseViewer(CustomViewer):
    def __init__(self, render_path, arm_path):
        super().__init__(render_path, 3, azimuth=-45, elevation=-30)
        self.arm_path = arm_path

        # 相机
        self.camera_name = "eye_in_hand"
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self.width = 640
        self.height = 480

        # 末端
        self.flange_name = "link7"
        self.flange_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.flange_name)

        # 运动学
        self.kinematics = PandaKinematics()
        self.movej_planner = MoveJPlanner(self.kinematics, num_steps=3)
        self.movel_planner = MoveLPlanner(self.kinematics)

        # 通用变量
        self.print_counter = 0
        self.initial_pos = [0, 0, 0, -1.44, 0, 1.5, 0.9]
        self.joint_cmd = None
        self.gripper_cmd = 0.04
        self.control_mode = None

        # Mocap
        self.mocaps = MocapManager(self.model, self.data)

    def calc_intrinsics(self):
        import math
        fovy = self.model.cam_fovy[self.camera_id]
        fy = 0.5 * self.height / math.tan(fovy * math.pi / 360)
        fx = fy * (self.width / self.height)
        cx = self.width / 2
        cy = self.height / 2
        print(f'camera intrinsics: {fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}')

    def runBefore(self):
        self.model.opt.timestep = 0.005

    def get_camera_image(self, show=True):
        return self.getFixedCameraImage(self.camera_name, show=show)