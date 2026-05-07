import mujoco
import numpy as np
from mujoco_viewer import ArmBaseViewer, ROOT_DIR
from utils import euler2rotmat

class JointImpedanceControlViewer(ArmBaseViewer):
    def __init__(self, render_path, arm_path):
        super().__init__(render_path, arm_path)
        self.Kp = np.diag([10] * 7)  # 刚度矩阵
        self.Kd = np.diag([100] * 7)  # 阻尼矩阵
        self.joint_cmd = None

    def runBefore(self):
        self.data.qpos[:7] = [0, 0.377, 0, -1, 0, 1.44, 1]


    def runFunc(self):
        if self.joint_cmd is None:
            self.joint_cmd = self.data.qpos[:7].copy()

        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()

        error = self.joint_cmd - q


        tau_impedance = self.Kp @ error - self.Kd @ dq

        tau_gravity = self.data.qfrc_bias[:7].copy()

        total_torque = tau_impedance + tau_gravity
        print(tau_impedance)

        self.data.ctrl[:7] = total_torque
        
        

if __name__ == '__main__':
    SCENE_XML_PATH = str(ROOT_DIR / 'model/franka_emika_panda/scene_tau.xml')
    robot = JointImpedanceControlViewer(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
