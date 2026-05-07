
import pinocchio
from utils import *

class MoveLPlanner:
    def __init__(self, kinematics, num_steps=300):
        self.kin = kinematics          # 直接使用你的正逆解
        self.num_steps = num_steps      # 轨迹平滑度
        self.max_error = 1e-3           # 直线精度

    def plan(self, q_start, target_pos, target_rot):
        """
        笛卡尔空间直线运动
        :param q_start:    起始关节角
        :param target_pos: 目标位置 [x,y,z]
        :param target_rot: 目标旋转矩阵 3x3
        :return: 轨迹 [q0, q1, q2 ...]
        """
        q_start = np.array(q_start).flatten()
        target_pos = np.array(target_pos)
        target_rot = np.array(target_rot)

        # 1. 正解得到起始位姿
        start_pos, start_quat = self.kin.fk(q_start)
        start_rot = self._quat2rot(start_quat)

        trajectory = [q_start.copy()]
        q_current = q_start.copy()

        # 2. 笛卡尔空间线性插补
        for t in np.linspace(0, 1, self.num_steps)[1:]:
            # 位置插补
            interp_pos = (1 - t) * start_pos + t * target_pos

            # 姿态插补（球面线性插值）
            interp_rot = self._slerp_rot(start_rot, target_rot, t)

            # 3. 每一步都调用 IK 求解关节角
            success, q_current = self.kin.ik(
                current_q=q_current,
                target_rot=interp_rot,
                target_pos=interp_pos,
                eps=self.max_error,
                DT=0.2,
                damp=1e-6
            )
            trajectory.append(np.array(q_current))

        return trajectory

    def _quat2rot(self, quat):
        """四元数 [w,x,y,z] → 旋转矩阵"""
        w, x, y, z = quat
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])

    def _slerp_rot(self, R1, R2, t):
        """旋转矩阵球面插值，保证姿态平滑转动"""
        q1 = pinocchio.Quaternion(R1)
        q2 = pinocchio.Quaternion(R2)
        q = q1.slerp(t, q2)
        return q.matrix()
    
    
class MoveJPlanner:
    def __init__(self, kinematics, num_steps=300):
        self.kin = kinematics  # 正逆解工具
        self.num_steps = num_steps

    def plan(self, q_start, target_pos, target_rot):
        """
        统一接口：和 MoveL 完全一样
        功能：关节空间平滑运动 + 精准到达目标位姿（pos+rot）
        """
        # 1. 转 numpy 数组（修复报错核心）
        q_start = np.array(q_start, dtype=np.float64).flatten()
        
        # 2. 逆解求解目标关节角（保证末端到达目标位姿）
        success, q_target = self.kin.ik(q_start, target_rot, target_pos)
        if not success:
            print("❌ MoveJ 规划失败：IK 无解")
            return []
        
        q_target = np.array(q_target, dtype=np.float64).flatten()

        # 3. 关节空间线性插值
        trajectory = []
        for t in np.linspace(0, 1, self.num_steps):
            q = (1 - t) * q_start + t * q_target
            trajectory.append(q.copy())

        return trajectory
