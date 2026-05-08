import mujoco
import numpy as np
from mujoco_viewer import ArmBaseViewer
from xml_paths import PANDA_TAU_SCENE_XML

class JointAdmittanceControlViewer(ArmBaseViewer):
    def __init__(self, render_path, arm_path):
        super().__init__(render_path, arm_path)

        # ================= 导纳控制参数 =================
        self.Md = np.diag([0.5] * 7)  # 虚拟质量
        self.Kd = np.diag([10.0] * 7) # 虚拟刚度
        self.Dd = np.diag([20.0] * 7) # 虚拟阻尼

        # 导纳内部状态：期望 关节位置/速度
        self.q_des = None
        self.dq_des = np.zeros(7)

        # 内环PD控制器（跟踪导纳给出的轨迹）
        self.Kp = 10
        self.Kd_torque = 50

    def runBefore(self):
        self.data.qpos[:7] = [0, 0.377, 0, -1, 0, 1.44, 1]

    def runFunc(self):
        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()
        dt = self.model.opt.timestep

        # 初始化目标位置
        if self.q_des is None:
            self.q_des = q.copy()

        tau_ext = np.asarray(self.data.qfrc_applied[:7]).ravel()


        # ================ 2. 导纳控制算法 ================
        e_q = self.q_des - q
        e_dq = self.dq_des - dq

        # 【修复】确保所有参与矩阵乘法的变量都是 (7,)
        e_q = e_q.ravel()
        e_dq = e_dq.ravel()

        # 虚拟加速度
        ddq_des = np.linalg.inv(self.Md) @ (tau_ext - self.Dd @ e_dq - self.Kd @ e_q)

        # 积分得到期望速度、位置
        self.dq_des += ddq_des * dt
        self.q_des += self.dq_des * dt

        # ================ 3. 内环PD控制（跟踪） ================
        tau_pd = self.Kp * (self.q_des - q) - self.Kd_torque * dq

        # ================ 4. 重力补偿 ================
        tau_gravity = self.data.qfrc_bias[:7].copy()

        # ================ 5. 输出力矩 ================
        total_torque = tau_pd + tau_gravity
        self.data.ctrl[:7] = total_torque
        print(f"tau_pd: {tau_pd}, q: {q}")

if __name__ == '__main__':
    SCENE_XML_PATH = PANDA_TAU_SCENE_XML
    robot = JointAdmittanceControlViewer(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
