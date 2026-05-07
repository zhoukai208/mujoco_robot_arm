import mujoco
import numpy as np
from mujoco_viewer import ArmBaseViewer, ROOT_DIR


# ===================== 姿态误差 =====================
def get_orientation_error(desired_quat, current_quat):
    q_err = np.zeros(4)
    q_curr_inv = np.zeros(4)

    mujoco.mju_negQuat(q_curr_inv, current_quat)
    mujoco.mju_mulQuat(q_err, desired_quat, q_curr_inv)

    orient_error = np.zeros(3)
    mujoco.mju_quat2Vel(orient_error, q_err, 1.0)
    return orient_error


class EndEffectorImpedanceControlViewer(ArmBaseViewer):
    def __init__(self, render_path, arm_path):
        super().__init__(render_path, arm_path)

        # ===== 任务空间阻抗 =====
        self.Kp = np.diag([60, 60, 60, 20, 20, 20])
        self.Kd = np.diag([150, 150, 150, 40, 40, 40])

        # ===== null-space（加强）=====
        self.Kp_null = 100
        self.Kd_null = 20


        self.ee_target_pos = None
        self.ee_target_quat = None
        self.q_init = None

    def runBefore(self):
        self.data.qpos[:7] = [0, 0.377, 0, -1, 0, 1.44, 1]

    def runFunc(self):

        # ===== 初始化 =====
        if self.ee_target_pos is None:
            self.ee_target_pos = self.data.body(self.ee_id).xpos.copy()
            self.ee_target_quat = self.data.body(self.ee_id).xquat.copy()
            self.q_init = self.data.qpos[:7].copy()

        # ===== 状态 =====
        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()

        ee_pos = self.data.body(self.ee_id).xpos.copy()
        ee_quat = self.data.body(self.ee_id).xquat.copy()

        # ===== 误差 =====
        pos_err = self.ee_target_pos - ee_pos
        rot_err = get_orientation_error(self.ee_target_quat, ee_quat)
        ee_err = np.hstack([pos_err, rot_err])

        # ===== Jacobian =====
        J = self.kinematics.J(q)  # 6x7

        # ===== 末端速度 =====
        ee_vel = J @ dq
        print("ee_vel:", ee_vel)   
        print("ee_err:", ee_err)

        # ===== 动力学矩阵 =====
        nv = self.model.nv
        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)

        M = M_full[:7, :7]
        Minv = np.linalg.inv(M)

        # ===== Lambda（带阻尼，防奇异）=====
        Lambda = np.linalg.inv(J @ Minv @ J.T + 1e-6 * np.eye(6))

        # ===== 任务空间阻抗 =====
        F = self.Kp @ ee_err - self.Kd @ ee_vel

        # ===== 主任务 =====
        tau_task = J.T @ (Lambda @ F)
        

        # ===== 动力学一致伪逆 =====
        J_pinv = Minv @ J.T @ Lambda

        # ===== null-space =====
        N = np.eye(7) - J_pinv @ J

        tau_null = N @ (
            -self.Kp_null * (q - self.q_init)
            -self.Kd_null * dq
        )

        # ===== 重力补偿 =====
        tau_bias = self.data.qfrc_bias[:7].copy()

        # ===== 总力矩 =====

        tau = tau_task + tau_null + tau_bias

        self.data.ctrl[:7] = tau

        # ===== Debug =====
        # print("pos_err:", np.linalg.norm(pos_err), "rot_err:", np.linalg.norm(rot_err), "F:", F)
        # print("tau:", tau, "tau_task:", tau_task, "tau_null:", tau_null, "tau_joint:", tau_joint, "tau_bias:", tau_bias)
        print("J:", J)

if __name__ == '__main__':
    SCENE_XML_PATH = str(ROOT_DIR / 'model/franka_emika_panda/scene_tau.xml')
    robot = EndEffectorImpedanceControlViewer(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
