import yaml
import mujoco
import random
import numpy as np
from arm_base import ArmBaseViewer, ROOT_DIR
from utils import *
import pickle

class ArmReachCollector(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 安全工作空间
        self.workspace = {
            'x': [-0.7, 0.7],
            'y': [-0.7, 0.7],
            'z': [0.02, 0.1]
        }

        # 状态机
        self.CONTROL_STATE = {
            'IDLE': 0,       # 重置
            'PLANNING': 1,   # 解IK得到目标关节
            'MOVING': 2,     # 机械臂运动（真的动）
            'COLLECT': 3     # 采集1条数据并重置
        }
        self.current_state = self.CONTROL_STATE['IDLE']

        # 固定向下姿态
        self.target_ori = {
            "roll": 180,
            "pitch": 0,
            "yaw": 0
        }

        # BC 数据
        self.dataset = []
        self.collect_steps = 0
        self.max_collect = 5000
        self.save_path = ROOT_DIR / "artifacts/bc_reach/bc_reach_dataset.pkl"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_target = None
        self.cube_pos = np.zeros(3)

    def get_target_rot(self):
        r = np.deg2rad(self.target_ori["roll"])
        p = np.deg2rad(self.target_ori["pitch"])
        y = np.deg2rad(self.target_ori["yaw"])
        return euler2rotmat(r, p, y)

    def get_random_pos(self):
        x = random.uniform(*self.workspace['x'])
        y = random.uniform(*self.workspace['y'])
        z = random.uniform(*self.workspace['z'])
        return np.array([x, y, z])

    def set_cube(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_body")
        adr = self.model.jnt_qposadr[jid]
        self.cube_pos = self.get_random_pos()
        self.data.qpos[adr:adr+3] = self.cube_pos
        self.data.qpos[adr+3:adr+7] = [1,0,0,0]
        target_pos = self.cube_pos + np.array([0,0,0.2])
        return target_pos

    def plan_ik(self):
        q_current = self.data.qpos[:7]
        target_pos = self.set_cube()
        target_rot = self.get_target_rot()
        success, self.q_target = self.kinematics.ik(q_current, target_rot, target_pos)
        return success

    def collect_single_data(self):
        # 观测：当前7关节 + 方块3位置
        obs = np.concatenate([self.data.qpos[:7], self.cube_pos])
        # 动作：目标关节角（BC学习的正确标签）
        act = self.q_target.copy()
        self.dataset.append({"obs": obs, "act": act})
        self.collect_steps += 1

        if self.collect_steps % 1000 == 0:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.dataset, f)
            print(f"✅ 已保存 {self.collect_steps}/{self.max_collect}")

    def runBefore(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def runFunc(self):
        # ===================== IDLE: 重置 =====================
        if self.current_state == self.CONTROL_STATE['IDLE']:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.current_state = self.CONTROL_STATE['PLANNING']

        # ===================== PLANNING: 解IK =====================
        elif self.current_state == self.CONTROL_STATE['PLANNING']:
            if self.plan_ik():
                self.current_state = self.CONTROL_STATE['MOVING']
            else:
                print("❌ IK失败")
                self.current_state = self.CONTROL_STATE['IDLE']

        # ===================== MOVING: 机械臂真的运动！=====================
        elif self.current_state == self.CONTROL_STATE['MOVING']:
            q_current = self.data.qpos[:7]
            # 简单PD控制，让机械臂走向目标
            self.data.ctrl[:7] = self.q_target

            # 判断是否到达
            if np.mean(np.abs(q_current - self.q_target)) < 0.05:
                self.current_state = self.CONTROL_STATE['COLLECT']

        # ===================== COLLECT: 采1条数据 =====================
        elif self.current_state == self.CONTROL_STATE['COLLECT']:
            self.collect_single_data()
            if self.collect_steps >= self.max_collect:
                print("\n🎉 采集完成！")
                return
            self.current_state = self.CONTROL_STATE['IDLE']

        # 打印
        self.print_counter += 1
        if self.print_counter % 50 == 0:
            print(f"采集: {self.collect_steps}/{self.max_collect} | 方块: {self.cube_pos.round(2)}")

if __name__ == '__main__':
    SCENE_XML_PATH = str(ROOT_DIR / 'model/franka_emika_panda/scene_pos.xml')
    YAML_PATH = str(ROOT_DIR / 'config/target_pos.yaml')
    robot = ArmReachCollector(SCENE_XML_PATH, YAML_PATH, YAML_PATH)
    robot.run_loop()
