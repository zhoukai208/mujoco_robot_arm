import mujoco
import random
import numpy as np
import pickle
from utils import euler2rotmat
from pinocchio_kinematic import PandaKinematics

# ===================== 固定配置 =====================
MODEL_XML_PATH = "/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_pos.xml"
SAVE_PATH = "bc_reach_dataset.pkl"
MAX_COLLECT = 5000  # 数据量

# 工作空间
WORKSPACE = {
    'x': [-0.7, 0.7],
    'y': [-0.7, 0.7],
    'z': [0.02, 0.1]
}

# 固定末端姿态
TARGET_ORI = {
    "roll": 180,
    "pitch": 0,
    "yaw": 0
}

# ===================== 纯后台加载模型 =====================
# 无渲染、无窗口、纯计算
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)
kin = PandaKinematics(arm_path='/home/ethan/work/mujoco-learning-main/model/franka_panda_urdf/robots/panda_arm.urdf')  # 初始化运动学

# ===================== 全局变量 =====================
dataset = []
collect_steps = 0

# ===================== 核心函数（完全复用你的逻辑） =====================
def get_target_rot():
    r = np.deg2rad(TARGET_ORI["roll"])
    p = np.deg2rad(TARGET_ORI["pitch"])
    y = np.deg2rad(TARGET_ORI["yaw"])
    return euler2rotmat(r, p, y)

def get_random_pos():
    x = random.uniform(*WORKSPACE['x'])
    y = random.uniform(*WORKSPACE['y'])
    z = random.uniform(*WORKSPACE['z'])
    return np.array([x, y, z])

def set_cube():
    """设置随机方块位置"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_body")
    adr = model.jnt_qposadr[jid]
    cube_pos = get_random_pos()
    data.qpos[adr:adr+3] = cube_pos
    data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    target_pos = cube_pos + np.array([0, 0, 0.2])
    return cube_pos, target_pos

def plan_ik():
    """求解IK"""
    q_current = data.qpos[:7].copy()
    cube_pos, target_pos = set_cube()
    target_rot = get_target_rot()
    success, q_target = kin.ik(q_current, target_rot, target_pos)
    return success, q_current, cube_pos, q_target

def collect_single_data(obs, act):
    """采集单条数据并保存"""
    global collect_steps
    dataset.append({"obs": obs, "act": act})
    collect_steps += 1

    # 每1000条保存一次（覆盖式保存完整数据集，安全）
    if collect_steps % 1000 == 0:
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"✅ 已保存 {collect_steps}/{MAX_COLLECT}")

# ===================== 纯后台主循环 =====================
print("🚀 开始 无渲染 后台生成BC数据...")
while collect_steps < MAX_COLLECT:
    # 1. 重置机器人
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 2. 求解IK
    success, q_start, cube_pos, q_target = plan_ik()
    if not success:
        print("❌ IK求解失败，跳过")
        continue

    # 3. 后台运动（无渲染，仅计算）直到到达目标
    while True:
        q_current = data.qpos[:7].copy()
        data.ctrl[:7] = q_target  # 下发控制指令
        mujoco.mj_step(model, data)  # 后台仿真步进
        
        # 判断到达
        if np.mean(np.abs(q_current - q_target)) < 0.05:
            break

    # 4. 采集数据（和你原来格式完全一致）
    obs = np.concatenate([q_start, cube_pos])
    act = q_target.copy()
    collect_single_data(obs, act)

# 最终保存
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)

print(f"\n🎉 数据生成完成！")
print(f"📊 总样本数：{len(dataset)}")
print(f"💾 保存路径：{SAVE_PATH}")