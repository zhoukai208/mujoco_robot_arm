import mujoco
import random
import numpy as np
from arm_base import ArmBaseViewer
from utils import *
import pickle

class BCDataVisualizer(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path, dataset_path="bc_reach_dataset.pkl"):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 加载BC数据集（核心：读取pkl文件）
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()
        print(f"✅ 成功加载数据集，总样本数: {len(self.dataset)}")

        # 验证状态机
        self.CONTROL_STATE = {
            'IDLE': 0,       # 空闲：随机选数据样本
            'SETUP': 1,      # 设置方块位置+目标关节
            'MOVING': 2,     # 机械臂运动验证
        }
        self.current_state = self.CONTROL_STATE['IDLE']

        # 固定向下姿态（仅保留，不使用）
        self.target_ori = {
            "roll": 180,
            "pitch": 0,
            "yaw": 0
        }

        # 数据样本缓存
        self.current_sample = None
        self.q_target = None
        self.cube_pos = None

    def load_dataset(self):
        """读取pkl数据集"""
        try:
            with open(self.dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            return []

    def get_random_sample(self):
        """随机抽取一个数据样本"""
        if not self.dataset:
            return None
        return random.choice(self.dataset)

    def set_cube_from_data(self):
        """从数据样本中设置方块位置"""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_body")
        adr = self.model.jnt_qposadr[jid]
        # 从数据集读取方块位置
        self.data.qpos[adr:adr+3] = self.cube_pos
        self.data.qpos[adr+3:adr+7] = [1,0,0,0]

    def set_q_start(self):
        """设置初始关节角度"""
        self.data.qpos[:7] = self.q_start

    def runBefore(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def runFunc(self):
        # ===================== IDLE: 随机选取数据样本 =====================
        if self.current_state == self.CONTROL_STATE['IDLE']:
            # 重置仿真
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            # 随机取一个样本
            self.current_sample = self.get_random_sample()
            if self.current_sample is None:
                return
            
            # 解析样本数据（核心验证）
            obs = self.current_sample["obs"]   # 10维: [7关节, 3方块位置]
            act = self.current_sample["act"]   # 7维: 目标关节角q_target
            
            self.cube_pos = obs[7:10]          # 提取方块位置
            self.q_start = obs[:7]
            self.q_target = act                # 提取目标关节
            print(f"\n🎯 验证样本 | 方块位置: {self.cube_pos.round(3)}, obs: {obs}, act: {act}")
            
            self.current_state = self.CONTROL_STATE['SETUP']

        # ===================== SETUP: 设置场景 =====================
        elif self.current_state == self.CONTROL_STATE['SETUP']:
            # 设置方块到数据集对应的位置
            self.set_cube_from_data()
            # 设置初始关节角度
            self.set_q_start()
            self.current_state = self.CONTROL_STATE['MOVING']

        # ===================== MOVING: 运动到目标，可视化验证 =====================
        elif self.current_state == self.CONTROL_STATE['MOVING']:
            q_current = self.data.qpos[:7]
            # 下发目标关节指令
            self.data.ctrl[:7] = self.q_target

            # 判断是否到达目标
            if np.mean(np.abs(q_current - self.q_target)) < 0.05:
                print("✅ 到达目标！数据验证正常 ✅")
                # 停留1秒后切换下一个样本
                mujoco.mj_step(self.model, self.data)
                import time
                time.sleep(1)
                self.current_state = self.CONTROL_STATE['IDLE']

        # 打印状态
        self.print_counter += 1
        if self.print_counter % 50 == 0:
            print(f"🎥 可视化验证中 | 方块位置: {self.cube_pos.round(2)}")


if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_pos.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
    
    # 初始化可视化验证器
    robot = BCDataVisualizer(SCENE_XML_PATH, YAML_PATH, YAML_PATH)
    robot.run_loop()