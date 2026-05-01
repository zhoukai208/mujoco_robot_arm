import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
import src.mujoco_viewer as mujoco_viewer
import os

# 防止Qt冲突
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path

    def runBefore(self):
        self.nj = self.model.nu
        self.dt = self.model.opt.timestep

        # ===================== 真正的导纳参数 =====================
        self.M = 1.0    # 虚拟质量
        self.B = 10.0   # 虚拟阻尼（让运动平滑）

        # 初始状态
        self.q_desired = self.data.qpos[:self.nj].copy()  # 目标角度
        self.dq_desired = np.zeros(self.nj)               # 目标速度

        # 记录数据
        self.q_history = []
        self.qdot_history = []
        self.torque_history = []

        # PD跟踪参数（只负责跟踪目标，不产生回弹）
        self.Kp = 80
        self.Kd = 8

    def runFunc(self):
        q = self.data.qpos[:self.nj]
        qdot = self.data.qvel[:self.nj]

        # ===================== 核心：导纳控制 =====================
        # 外部力矩（你拖拽产生的力）
        tau_ext = self.data.qfrc_applied[:self.nj]

        # 导纳公式：外力 → 加速度
        # ddq = F_ext / M  - B/M * dq_desired
        ddq_desired = (tau_ext - self.B * self.dq_desired) / self.M

        # 积分：加速度 → 速度
        self.dq_desired += ddq_desired * self.dt

        # 积分：速度 → 目标位置（关键！目标位置被外力改变了）
        self.q_desired += self.dq_desired * self.dt

        # ===================== 底层PD跟踪目标位置 =====================
        tau = self.Kp * (self.q_desired - q) - self.Kd * qdot

        # 控制输出
        self.data.ctrl[:] = tau

        # 记录
        self.q_history.append(q.copy())
        self.qdot_history.append(qdot.copy())
        self.torque_history.append(tau.copy())

test = Test("./model/trs_so_arm100/scene_without_position.xml")
test.run_loop()