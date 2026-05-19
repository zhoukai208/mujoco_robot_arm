"""Jacobian / Singularity 可视化结果说明。

这个脚本用于观察机械臂在不同姿态下的雅可比矩阵性质。窗口左侧是
eye-in-hand 相机图像，右侧 HUD 分三列显示数值指标。

左列:Jacobian 基础指标

- q:
  当前 7 个关节角，单位 rad。

- qdot:
  当前 7 个关节速度，单位 rad/s。

- 6D Jacobian J [linear; angular]:
  完整末端雅可比，满足:
      [vx, vy, vz, wx, wy, wz] = J(q) qdot
  前三维是末端线速度，后三维是末端角速度。

- sigma:
  Jacobian 的奇异值。每个奇异值表示关节速度映射到某个末端运动方向
  的能力；越大表示该方向越容易动，越小表示越难动。

- rank:
  Jacobian 的秩。6D Jacobian 满秩是 6/6，位置 Jacobian 满秩是 3/3。
  如果 rank 下降，说明末端某些方向已经失去或近似失去运动能力。

- min sigma:
  最小奇异值，是判断奇异点最直观的指标。越接近 0，越接近奇异姿态。

- condition:
  条件数，约等于 最大奇异值 / 最小奇异值。值越大，表示运动能力越
  不均匀；接近奇异点时通常会变得很大。

- manipulability:
  可操作度，这里用奇异值乘积表示。越大表示当前姿态附近越灵活；
  越接近 0，越接近奇异或运动能力退化。

- worst twist:
  6D 末端最差运动方向，也就是最小奇异值对应的方向，包含线速度和
  角速度分量。

- Position Jacobian Jv:
  只看末端位置运动的雅可比，即 J 的前三行:
      [vx, vy, vz] = Jv(q) qdot
  它更适合理解 reach、轨迹跟踪、抓取接近等只关心末端位置的任务。

- worst xyz:
  末端位置最差运动方向。例如该方向接近 [0, 0, 1] 时，说明当前姿态
  下沿 Z 方向移动比较困难。

中列:末端速度、Null Space、DLS

- End-effector velocity:
  当前真实仿真状态下，由 J(q) qdot 算出来的末端速度。

- linear m/s:
  末端线速度 [vx, vy, vz]，单位 m/s。

- angular rad/s:
  末端角速度 [wx, wy, wz]，单位 rad/s。

- |linear| / |angular|:
  线速度大小和角速度大小。

- 6D null-space direction:
  6D 任务的零空间方向。

- n:
  一个 7 维关节速度方向。理论上如果 qdot = n，则:
      J(q) n ~= 0
  也就是关节在动，但末端 6D 位姿几乎不动。这是 7 自由度机械臂冗余性
  的来源。

- ||J n||:
  验证 null-space 的残差。越接近 0，说明这个方向越是真正的零空间方向。

- Pseudo-inverse vs DLS:
  普通伪逆和阻尼最小二乘（Damped Least Squares）的对比。

- lambda:
  DLS 的阻尼系数。阻尼越大，越不容易在奇异点附近产生巨大关节速度，
  但末端跟踪误差也会变大。

- target twist:
  用于测试的目标末端速度方向。这里选择 worst twist，也就是最难实现
  的方向。

- ||qdot_pinv||:
  用普通伪逆求出的关节速度大小。接近奇异点时，这个值可能暴涨。

- ||qdot_dls||:
  用 DLS 求出的关节速度大小。通常比伪逆更小、更稳定。

- twist err pinv / twist err dls:
  伪逆和 DLS 方法产生的末端速度误差。DLS 通常牺牲一点精度，换取
  更小的关节速度和更稳定的控制。

右列:Manipulability Ellipsoid

- Manipulability ellipsoid:
  位置可操作椭球的 XY 投影。它表示当前姿态下，如果关节速度大小受限:
      ||qdot|| <= 1
  末端线速度大概能覆盖哪些方向。

  椭圆越圆，说明各方向运动能力越均匀；椭圆越扁，说明某些方向很好动、
  某些方向很难动；某个轴很短时，说明对应方向接近退化。

- 3D radii:
  位置可操作椭球三个主轴长度，对应位置 Jacobian 的 3 个奇异值。

- major=cyan:
  最长主轴方向，即当前最容易产生末端线速度的方向。

- minor=orange:
  较短主轴方向，即较困难的运动方向之一。

- Green outline:
  XY 投影下的速度椭圆轮廓。

快速判断规则:

- min sigma 越小，越接近奇异。
- condition 越大，越不稳定。
- manipulability 越小，整体运动能力越差。
- ellipsoid 越扁，某些方向越难动。
- ||qdot_pinv|| 远大于 ||qdot_dls||，说明伪逆在当前姿态下容易放大关节速度。
"""

import argparse
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from mujoco_viewer import ArmBaseViewer, CAMERA_WINDOW_NAME
from utils import format_vec
from xml_paths import PANDA_POS_SCENE_XML


HUD_WINDOW_NAME = "Jacobian Singularity"
KEY_NONE = -1


@dataclass
class DlsComparison:
    """伪逆和 DLS 在同一个目标 twist 下的对比结果。"""

    target_twist: np.ndarray
    qdot_pinv: np.ndarray
    qdot_dls: np.ndarray
    achieved_twist_pinv: np.ndarray
    achieved_twist_dls: np.ndarray
    error_pinv: float
    error_dls: float
    qdot_norm_pinv: float
    qdot_norm_dls: float


@dataclass
class ManipulabilityEllipsoid:
    """位置可操作椭球。

    对位置雅可比 Jv 做 SVD:
        Jv = U * S * V^T

    当关节速度满足 ||qdot|| <= 1 时，末端可达线速度集合近似为
    一个椭球；U 的列向量是椭球主轴方向，S 是主轴长度。
    """

    axes_world: np.ndarray
    radii: np.ndarray


@dataclass
class JacobianMetrics:
    """一次 Jacobian 分析的结果。

    这里同时保留 6D Jacobian 和位置 Jacobian 的指标:
    - 6D Jacobian: 同时考虑末端线速度和角速度，维度是 6x7。
    - Position Jacobian: 只看末端位置速度，维度是 3x7。

    两者都值得看:位置控制更关心 Jv，完整任务空间控制更关心 6D J。
    """

    singular_values_6d: np.ndarray
    singular_values_pos: np.ndarray
    rank_6d: int
    rank_pos: int
    manipulability_6d: float
    manipulability_pos: float
    min_sigma_6d: float
    min_sigma_pos: float
    condition_6d: float
    condition_pos: float
    worst_twist_direction: np.ndarray
    worst_pos_direction: np.ndarray
    nullspace_direction: np.ndarray
    nullspace_residual: float
    dls_lambda: float
    dls_comparison: DlsComparison
    ellipsoid: ManipulabilityEllipsoid
    ee_twist: np.ndarray


class JacobianAnalyzer:
    """从几何雅可比矩阵中计算奇异点相关指标。"""

    def __init__(self, kinematics, rank_tol=1e-4, dls_lambda=0.05):
        self.kinematics = kinematics
        self.rank_tol = rank_tol
        self.dls_lambda = dls_lambda

    def compute(self, q, qdot) -> JacobianMetrics:
        q = np.asarray(q, dtype=np.float64).flatten()
        qdot = np.asarray(qdot, dtype=np.float64).flatten()

        # PandaKinematics.J(q) 返回末端几何雅可比:
        #   twist = [linear_velocity, angular_velocity] = J(q) @ qdot
        # 对 7 自由度机械臂来说，J 的形状是 6x7。
        J = self.kinematics.J(q)

        # 只取前三行可以得到位置雅可比 Jv，用来观察末端位置速度方向
        # 是否退化。比如某些姿态下，机械臂可能很难沿某个方向移动。
        J_pos = J[:3, :]

        # 奇异值可以理解为“关节速度映射到末端速度”的放大能力。
        # 最小奇异值越接近 0，说明至少有一个末端运动方向很难产生，
        # 这就是接近奇异点的典型信号。
        U_6d, s_6d, Vh_6d = np.linalg.svd(J, full_matrices=True)
        U_pos, s_pos, _ = np.linalg.svd(J_pos, full_matrices=True)

        rank_6d = self._rank(s_6d)
        rank_pos = self._rank(s_pos)

        # U 的最后一列对应最小奇异值方向，也就是“最难产生的末端速度方向”。
        # 对 6D Jacobian 来说它是 twist 方向；对 Jv 来说它是 3D 线速度方向。
        worst_twist_direction = U_6d[:, -1].copy()
        worst_pos_direction = U_pos[:, -1].copy()

        # Panda 是 7 自由度，完整 6D 任务通常有 1 维 null space。
        # Vh 的最后一行就是一个关节空间方向 n，满足 J @ n ~= 0。
        nullspace_direction = Vh_6d[-1, :].copy()
        nullspace_direction = self._normalize(nullspace_direction)
        nullspace_residual = float(np.linalg.norm(J @ nullspace_direction))

        # DLS 对比:用最差 6D 方向作为目标 twist。
        # 伪逆在这个方向上最容易放大关节速度；DLS 会牺牲一点精度换稳定。
        dls_comparison = self._compare_dls(J, worst_twist_direction)

        ellipsoid = ManipulabilityEllipsoid(
            axes_world=U_pos.copy(),
            radii=s_pos.copy(),
        )

        # 直接验证核心公式:末端速度 = 雅可比 * 关节速度。
        # 前 3 维是线速度，后 3 维是角速度。
        ee_twist = J @ qdot

        return JacobianMetrics(
            singular_values_6d=s_6d,
            singular_values_pos=s_pos,
            rank_6d=rank_6d,
            rank_pos=rank_pos,
            manipulability_6d=self._product(s_6d),
            manipulability_pos=self._product(s_pos),
            min_sigma_6d=self._min_sigma(s_6d),
            min_sigma_pos=self._min_sigma(s_pos),
            condition_6d=self._condition(s_6d),
            condition_pos=self._condition(s_pos),
            worst_twist_direction=worst_twist_direction,
            worst_pos_direction=worst_pos_direction,
            nullspace_direction=nullspace_direction,
            nullspace_residual=nullspace_residual,
            dls_lambda=self.dls_lambda,
            dls_comparison=dls_comparison,
            ellipsoid=ellipsoid,
            ee_twist=ee_twist,
        )

    def _rank(self, values):
        return int(np.sum(np.asarray(values) > self.rank_tol))

    @staticmethod
    def _normalize(values):
        values = np.asarray(values, dtype=np.float64)
        norm = np.linalg.norm(values)
        if norm < 1e-12:
            return values.copy()
        return values / norm

    def _compare_dls(self, J, target_twist):
        target_twist = self._normalize(target_twist)

        # Moore-Penrose 伪逆:奇异值小时会产生很大的关节速度。
        qdot_pinv = np.linalg.pinv(J) @ target_twist

        # Damped Least Squares:
        #   qdot = J^T (J J^T + lambda^2 I)^-1 v
        # lambda 越大越稳定，但末端速度跟踪误差也会更大。
        lhs = J @ J.T + (self.dls_lambda ** 2) * np.eye(J.shape[0])
        qdot_dls = J.T @ np.linalg.solve(lhs, target_twist)

        achieved_pinv = J @ qdot_pinv
        achieved_dls = J @ qdot_dls
        return DlsComparison(
            target_twist=target_twist,
            qdot_pinv=qdot_pinv,
            qdot_dls=qdot_dls,
            achieved_twist_pinv=achieved_pinv,
            achieved_twist_dls=achieved_dls,
            error_pinv=float(np.linalg.norm(achieved_pinv - target_twist)),
            error_dls=float(np.linalg.norm(achieved_dls - target_twist)),
            qdot_norm_pinv=float(np.linalg.norm(qdot_pinv)),
            qdot_norm_dls=float(np.linalg.norm(qdot_dls)),
        )

    @staticmethod
    def _product(values):
        # Yoshikawa manipulability 常用形式是奇异值乘积。
        # 值越小，局部运动能力越差；接近 0 通常表示接近奇异。
        return float(np.prod(np.maximum(values, 0.0)))

    @staticmethod
    def _min_sigma(values):
        if len(values) == 0:
            return 0.0
        return float(np.min(values))

    @staticmethod
    def _condition(values):
        # 条件数 = 最大奇异值 / 最小奇异值。
        # 条件数越大，速度映射越“扁”，小的关节误差越容易放大成
        # 某些方向上的末端控制困难。
        if len(values) == 0:
            return float("inf")
        min_value = float(np.min(values))
        if min_value < 1e-9:
            return float("inf")
        return float(np.max(values) / min_value)


class AutoJointMotion:
    """生成平滑的关节目标，让 Jacobian 指标随姿态变化而变化。"""

    def __init__(self, home_q, model_timestep):
        self.home_q = np.asarray(home_q, dtype=np.float64).copy()
        self.timestep = float(model_timestep)
        self.step_count = 0
        self.enabled = True

        # 每个关节使用不同的幅值、频率和相位。
        # 这样机械臂不会只是做单一周期动作，Jacobian 会经过更多不同姿态，
        # 更容易观察最小奇异值和可操作度的变化。
        self.amplitude = np.array([0.35, 0.35, 0.45, 0.35, 0.45, 0.30, 0.45])
        self.frequency = np.array([0.20, 0.17, 0.13, 0.11, 0.19, 0.15, 0.09])
        self.phase = np.array([0.0, 0.6, 1.1, 0.3, 1.7, 0.9, 2.2])

    def next_target(self):
        if not self.enabled:
            return self.home_q.copy()
        t = self.step_count * self.timestep
        self.step_count += 1

        # 用正弦波生成目标关节角。这里不是为了做精确轨迹规划，
        # 而是为了让 demo 自动扫过一批姿态，方便观察指标。
        return self.home_q + self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + self.phase)

    def reset(self, home_q):
        self.home_q = np.asarray(home_q, dtype=np.float64).copy()
        self.step_count = 0

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled


class JacobianHud:
    """把 Jacobian 指标画到 OpenCV 窗口上。"""

    def __init__(self, window_name=HUD_WINDOW_NAME):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1820, 820)

    def draw(self, frame, q, qdot, metrics: JacobianMetrics, auto_enabled):
        # 左侧保留相机图像，右侧拼接一个深色信息面板。
        # 这样既能观察机械臂/环境，也能实时看数值变化。
        target_height = max(frame.shape[0], 820)
        if frame.shape[0] < target_height:
            pad = target_height - frame.shape[0]
            camera_panel = cv2.copyMakeBorder(
                frame,
                0,
                pad,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        else:
            camera_panel = frame.copy()

        panel_width = 1180
        panel = np.zeros((target_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (28, 31, 34)
        canvas = cv2.hconcat([camera_panel, panel])

        panel_x = frame.shape[1]
        x1 = panel_x + 24
        x2 = panel_x + 410
        x3 = panel_x + 790

        y = 34
        self._put(canvas, "Jacobian Singularity Demo", x1, y, scale=0.78, color=(255, 255, 255))
        self._put(canvas, f"mode: {'AUTO' if auto_enabled else 'HOLD'}  |  Space: toggle  R: reset", x2, y, scale=0.52)
        y += 34
        y = self._put_vector(canvas, "q", q, x1, y, precision=2, chunk_size=4)
        y = self._put_vector(canvas, "qdot", qdot, x1, y, precision=2, chunk_size=4)
        y += 12

        # 6D Jacobian 用于完整末端 twist 映射:
        # [vx, vy, vz, wx, wy, wz]^T = J(q) qdot
        self._put(canvas, "6D Jacobian J [linear; angular]", x1, y, color=(210, 230, 255))
        y += 26
        y = self._put_vector(canvas, "sigma", metrics.singular_values_6d, x1, y, precision=4, chunk_size=3)
        self._put(
            canvas,
            f"rank: {metrics.rank_6d}/6  min sigma: {metrics.min_sigma_6d:.6f}",
            x1,
            y,
            color=self._metric_color(metrics.min_sigma_6d),
        )
        y += 24
        self._put(canvas, f"condition: {self._format_condition(metrics.condition_6d)}", x1, y)
        y += 24
        self._put(canvas, f"manipulability: {metrics.manipulability_6d:.8f}", x1, y)
        y += 24
        y = self._put_vector(canvas, "worst twist", metrics.worst_twist_direction, x1, y, precision=3, chunk_size=3)
        y += 16

        # 位置 Jacobian 只关心末端位置线速度，适合理解 reach、轨迹跟踪
        # 这类任务中的“某个方向是否不好动”。
        self._put(canvas, "Position Jacobian Jv", x1, y, color=(210, 230, 255))
        y += 26
        y = self._put_vector(canvas, "sigma", metrics.singular_values_pos, x1, y, precision=4, chunk_size=3)
        self._put(
            canvas,
            f"rank: {metrics.rank_pos}/3  min sigma: {metrics.min_sigma_pos:.6f}",
            x1,
            y,
            color=self._metric_color(metrics.min_sigma_pos),
        )
        y += 24
        self._put(canvas, f"condition: {self._format_condition(metrics.condition_pos)}", x1, y)
        y += 24
        self._put(canvas, f"manipulability: {metrics.manipulability_pos:.8f}", x1, y)
        y += 24
        y = self._put_vector(canvas, "worst xyz", metrics.worst_pos_direction, x1, y, precision=3, chunk_size=3)

        # 这里显示的是用当前 qdot 计算出的末端速度。
        # 如果开启自动运动，qdot 来自 MuJoCo 仿真的实际关节速度；
        # 如果 HOLD，则速度会逐渐接近 0。
        linear = metrics.ee_twist[:3]
        angular = metrics.ee_twist[3:]
        y2 = 84
        self._put(canvas, "End-effector velocity", x2, y2, color=(210, 230, 255))
        y2 += 26
        y2 = self._put_vector(canvas, "linear m/s", linear, x2, y2, precision=4, chunk_size=3)
        y2 = self._put_vector(canvas, "angular rad/s", angular, x2, y2, precision=4, chunk_size=3)
        self._put(canvas, f"|linear|={np.linalg.norm(linear):.5f}", x2, y2)
        y2 += 24
        self._put(canvas, f"|angular|={np.linalg.norm(angular):.5f}", x2, y2)
        y2 += 42

        self._put(canvas, "6D null-space direction", x2, y2, color=(210, 230, 255))
        y2 += 26
        y2 = self._put_vector(canvas, "n", metrics.nullspace_direction, x2, y2, precision=3, chunk_size=4)
        self._put(canvas, f"||J n||: {metrics.nullspace_residual:.8f}", x2, y2)
        y2 += 42

        dls = metrics.dls_comparison
        self._put(canvas, f"Pseudo-inverse vs DLS (lambda={metrics.dls_lambda:.3f})", x2, y2, color=(210, 230, 255))
        y2 += 26
        y2 = self._put_vector(canvas, "target twist", dls.target_twist, x2, y2, precision=3, chunk_size=3)
        self._put(canvas, f"||qdot_pinv||: {dls.qdot_norm_pinv:.4f}", x2, y2)
        y2 += 24
        self._put(canvas, f"||qdot_dls ||: {dls.qdot_norm_dls:.4f}", x2, y2)
        y2 += 24
        self._put(canvas, f"twist err pinv: {dls.error_pinv:.6f}", x2, y2)
        y2 += 24
        self._put(canvas, f"twist err dls : {dls.error_dls:.6f}", x2, y2)
        y2 += 34

        self._put(canvas, "Manipulability ellipsoid", x3, 84, color=(210, 230, 255))
        self._put(canvas, "Jv XY projection", x3, 110, color=(210, 230, 255))
        self._draw_ellipsoid_xy(canvas, metrics.ellipsoid, origin=(x3 + 170, 320))
        self._put(canvas, "3D radii:", x3, 520, color=(210, 230, 255))
        self._put(canvas, format_vec(metrics.ellipsoid.radii, precision=4), x3, 548)
        self._put(canvas, "Axis colors: major=cyan, minor=orange", x3, 586, scale=0.48)
        self._put(canvas, "Green outline: velocity ellipsoid", x3, 612, scale=0.48)

        cv2.imshow(self.window_name, canvas)
        return cv2.waitKeyEx(1)

    @staticmethod
    def _put(canvas, text, x, y, scale=0.56, color=(225, 225, 225)):
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _put_vector(canvas, label, values, x, y, precision=3, chunk_size=3):
        values = np.asarray(values).flatten()
        for start in range(0, len(values), chunk_size):
            chunk = values[start : start + chunk_size]
            prefix = f"{label}: " if start == 0 else " " * (len(label) + 2)
            text = prefix + format_vec(chunk, precision=precision)
            JacobianHud._put(canvas, text, x, y, scale=0.50)
            y += 22
        return y

    @staticmethod
    def _metric_color(value):
        # 简单用颜色标出接近奇异的程度:
        # 红色:最小奇异值很小；黄色:需要注意；绿色:相对健康。
        if value < 0.02:
            return (80, 80, 255)
        if value < 0.06:
            return (80, 210, 255)
        return (120, 240, 120)

    @staticmethod
    def _format_condition(value):
        if not np.isfinite(value):
            return "inf"
        return f"{value:.3f}"

    @staticmethod
    def _draw_ellipsoid_xy(canvas, ellipsoid: ManipulabilityEllipsoid, origin):
        # 位置可操作椭球是 3D 的。HUD 里画 XY 投影:
        #   covariance = U diag(sigma^2) U^T
        # 取 covariance 的 XY 子块，再做一次 2D 特征分解，就能得到
        # 屏幕上的椭圆长短轴和旋转角。
        center = tuple(int(v) for v in origin)
        axes = ellipsoid.axes_world
        radii = ellipsoid.radii
        cov_3d = axes @ np.diag(radii ** 2) @ axes.T
        cov_xy = cov_3d[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        max_radius = max(float(np.sqrt(np.max(eigvals))), 1e-6)
        scale = 120.0 / max_radius
        ellipse_axes = np.maximum(np.sqrt(eigvals) * scale, 1.0).astype(int)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        cv2.circle(canvas, center, 3, (255, 255, 255), -1)
        cv2.ellipse(canvas, center, tuple(ellipse_axes), angle, 0, 360, (120, 240, 120), 2, cv2.LINE_AA)
        cv2.line(canvas, (center[0] - 140, center[1]), (center[0] + 140, center[1]), (90, 90, 90), 1)
        cv2.line(canvas, (center[0], center[1] - 140), (center[0], center[1] + 140), (90, 90, 90), 1)
        cv2.putText(canvas, "+X", (center[0] + 128, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, "+Y", (center[0] + 8, center[1] - 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        colors = [(80, 200, 255), (255, 180, 80)]
        for i in range(2):
            vec = eigvecs[:, i] * ellipse_axes[i]
            start = (int(center[0] - vec[0]), int(center[1] + vec[1]))
            end = (int(center[0] + vec[0]), int(center[1] - vec[1]))
            cv2.line(canvas, start, end, colors[i], 2, cv2.LINE_AA)


class JacobianSingularityDemo(ArmBaseViewer):
    """雅可比与奇异点可视化 Demo 主类。

    这个类只负责连接 MuJoCo 仿真循环:
    1. 给机械臂一个关节目标；
    2. 读取当前 q / qdot；
    3. 调 JacobianAnalyzer 计算指标；
    4. 调 JacobianHud 显示结果。
    """

    def __init__(self, render_path, arm_path, start_hold=False):
        super().__init__(render_path, arm_path)
        self.start_hold = start_hold
        self.analyzer = JacobianAnalyzer(self.kinematics)
        self.hud = None
        self.motion = None
        self.home_q = None
        self.last_metrics = None

    def position_windows(self):
        # CustomViewer.run_loop() 默认会创建 "MuJoCo Camera Output"。
        # 这个 demo 已经把相机图像和指标画在 JacobianHud 里，默认窗口会变成
        # 空白/无用窗口，所以这里覆盖父类方法并确保它被关闭。
        try:
            cv2.destroyWindow(CAMERA_WINDOW_NAME)
        except cv2.error:
            pass

    def runBefore(self):
        super().runBefore()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # 以 keyframe 的初始姿态作为自动运动中心点。
        # --hold 模式下也会用这个姿态作为保持目标。
        self.home_q = self.data.qpos[:7].copy()
        self.motion = AutoJointMotion(self.home_q, self.model.opt.timestep)
        self.motion.enabled = not self.start_hold
        self.hud = JacobianHud()
        print("\nJacobian / Singularity Demo")
        print("Space: toggle AUTO/HOLD")
        print("R: reset to home")
        print("Close MuJoCo viewer or press Ctrl+C to exit.\n")

    def runFunc(self):
        self._apply_joint_motion()

        # MuJoCo 当前状态:q 是关节角，qdot 是关节速度。
        # 用真实仿真速度而不是目标速度，可以看到位置执行器动态带来的影响。
        q = self.data.qpos[:7].copy()
        qdot = self.data.qvel[:7].copy()
        metrics = self.analyzer.compute(q, qdot)
        self.last_metrics = metrics

        frame = self.get_camera_image(show=False)
        key = self.hud.draw(frame, q, qdot, metrics, self.motion.enabled)
        self._process_key(key)
        self._print_periodic_status(metrics)

    def _apply_joint_motion(self):
        # 位置执行器控制:写入前 7 个 ctrl，对应 joint1_pos 到 joint7_pos。
        q_target = self.motion.next_target()
        self.data.ctrl[:7] = q_target

        # demo 不研究夹爪，保持张开避免干扰观察。
        if self.model.nu > 7:
            self.data.ctrl[7:] = 255

    def _process_key(self, key):
        if key == KEY_NONE:
            return
        ascii_key = key & 0xFF
        if ascii_key == ord(" "):
            enabled = self.motion.toggle()
            print(f"[Mode] {'AUTO' if enabled else 'HOLD'}")
        elif ascii_key in (ord("r"), ord("R")):
            # 重置后重新把当前姿态设为自动运动中心点，避免旧轨迹目标
            # 和重置姿态之间产生突然跳变。
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.home_q = self.data.qpos[:7].copy()
            self.motion.reset(self.home_q)
            print("[Reset] home")

    def _print_periodic_status(self, metrics):
        self.print_counter += 1
        if self.print_counter % 120 != 0:
            return
        print(
            "[Jacobian] "
            f"min_sigma_6d={metrics.min_sigma_6d:.6f} "
            f"cond_6d={JacobianHud._format_condition(metrics.condition_6d)} "
            f"manip_6d={metrics.manipulability_6d:.8f} "
            f"linear_speed={np.linalg.norm(metrics.ee_twist[:3]):.5f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Jacobian singularity visualization demo.")
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Start in HOLD mode instead of automatic joint motion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    robot = JacobianSingularityDemo(
        PANDA_POS_SCENE_XML,
        PANDA_POS_SCENE_XML,
        start_hold=args.hold,
    )
    robot.run_loop()
