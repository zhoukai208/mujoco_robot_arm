import mujoco
import random
import numpy as np
import torch
import torch.nn as nn
from arm_base import ArmBaseViewer, ROOT_DIR
from utils import *
import time

# ===================== 【关键】和训练完全一致的 BC 模型 =====================
class BCModel(nn.Module):
    def __init__(self, obs_dim=10, act_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class BCReachInfer(ArmBaseViewer):
    def __init__(self, render_path, model_path=None):
        super().__init__(render_path, render_path)

        # 加载 BC 模型
        self.model_path = model_path or ROOT_DIR / "artifacts/bc_reach/bc_reach_best_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bc_model = BCModel().to(self.device)
        self.bc_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.bc_model.eval()
        print(f"✅ BC 最优模型加载成功！设备: {self.device}")

        # 状态机
        self.CONTROL_STATE = {
            'IDLE': 0,
            'INFER': 1,
            'MOVING': 2,
        }
        self.current_state = self.CONTROL_STATE['IDLE']

        # 工作空间
        self.workspace = {
            'x': [-0.7, 0.7],
            'y': [-0.7, 0.7],
            'z': [0.02, 0.1]
        }

        # 缓存变量
        self.cube_pos = None
        self.q_target = None
        self.target_pos = None

        # ===================== 新增：超时机制配置 =====================
        self.max_timeout_steps = 600    # 最大运动步数（超时判定失败）
        self.current_moving_steps = 0   # 当前运动步数
        # ===================== 新增：成功/失败统计 =====================
        self.success_count = 0
        self.fail_count = 0

        # 末端执行器配置
        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        except:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    def get_random_pos(self):
        x = random.uniform(*self.workspace['x'])
        y = random.uniform(*self.workspace['y'])
        z = random.uniform(*self.workspace['z'])
        return np.array([x, y, z], dtype=np.float32)

    def set_cube(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_body")
        adr = self.model.jnt_qposadr[jid]
        self.cube_pos = self.get_random_pos()
        self.data.qpos[adr:adr+3] = self.cube_pos
        self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
        
        self.target_pos = self.cube_pos + np.array([0, 0, 0.2])
        print(f"\n🎲 方块位置: {self.cube_pos.round(3)}")

    def infer_action(self):
        obs = np.concatenate([self.data.qpos[:7].astype(np.float32), self.cube_pos])
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.q_target = self.bc_model(obs_tensor).cpu().numpy()[0]

    def get_pose_error(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        pos_error = np.linalg.norm(ee_pos - self.target_pos)
        return pos_error

    def runBefore(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def runFunc(self):
        # 1. 重置 + 生成方块
        if self.current_state == self.CONTROL_STATE['IDLE']:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.current_moving_steps = 0  # 重置运动步数
            self.set_cube()
            self.current_state = self.CONTROL_STATE['INFER']

        # 2. BC 模型推理
        elif self.current_state == self.CONTROL_STATE['INFER']:
            self.infer_action()
            self.current_state = self.CONTROL_STATE['MOVING']

        # 3. 运动控制 + 超时判断 + 统计
        elif self.current_state == self.CONTROL_STATE['MOVING']:
            # 下发控制指令
            self.data.ctrl[:7] = self.q_target
            q_current = self.data.qpos[:7].copy()
            self.current_moving_steps += 1  # 步数累加

            # ============== 核心：关节到位判断 ==============
            if np.allclose(q_current, self.q_target, atol=1e-1):
                # 成功
                self.success_count += 1
                pos_error = self.get_pose_error()
                print(f"✅ 成功！| 末端误差: {pos_error*1000:.1f}mm")
                self.current_state = self.CONTROL_STATE['IDLE']

            # ============== 新增：超时失败判断 ==============
            elif self.current_moving_steps >= self.max_timeout_steps:
                # 超时失败
                self.fail_count += 1
                print(f"❌ 失败！| 运动超时({self.max_timeout_steps}步)")
                self.current_state = self.CONTROL_STATE['IDLE']

            # ============== 运动中，实时打印误差 ==============
            else:
                pos_error = self.get_pose_error()
                remaining_steps = self.max_timeout_steps - self.current_moving_steps
                print(f"🎥 运动中 | 误差:{pos_error*1000:.1f}mm | 剩余步数:{remaining_steps}")

        print(f"📊 统计：成功={self.success_count} | 失败={self.fail_count} | 总计={self.success_count+self.fail_count}")
        print("="*60)

if __name__ == '__main__':
    SCENE_XML_PATH = str(ROOT_DIR / 'model/franka_emika_panda/scene_pos.xml')
    MODEL_PATH = ROOT_DIR / 'artifacts/bc_reach/bc_reach_best_model.pth'
    
    robot = BCReachInfer(SCENE_XML_PATH, MODEL_PATH)
    robot.run_loop()
