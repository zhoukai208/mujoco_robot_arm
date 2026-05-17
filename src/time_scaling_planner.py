from dataclasses import dataclass

import numpy as np


@dataclass
class TimeScalingConfig:
    """时间参数化配置。

    method: linear、trapezoidal、quintic、s_curve 之一。
    duration: 轨迹总时长，单位秒。
    dt: 采样周期，单位秒。
    accel_fraction: 梯形速度曲线中加速段/减速段各占总时长的比例。
    auto_duration: 是否根据关节速度/加速度限制自动拉长轨迹时间。
    """

    method: str = "quintic"
    duration: float = 3.0
    dt: float = 0.01
    accel_fraction: float = 0.25
    auto_duration: bool = False


@dataclass
class JointLimitConfig:
    """关节空间约束。

    q_min/q_max 可以为 None，表示不检查位置上下限。
    qd_max/qdd_max 可以传标量或与关节数相同的数组。
    """

    q_min: object = None
    q_max: object = None
    qd_max: object = 1.0
    qdd_max: object = 2.0


@dataclass
class JointTrajectory:
    """最终给控制器或画图使用的关节轨迹。"""

    method: str
    time: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray

    @property
    def duration(self):
        """轨迹总时长。"""

        return float(self.time[-1])

    @property
    def dof(self):
        """关节自由度数量。"""

        return int(self.q.shape[1])

    def sample_at(self, t):
        """按控制器当前时间插值采样 q/qd/qdd。

        这里使用逐关节线性插值，便于控制循环的 dt 与规划 dt 不完全一致时使用。
        """

        t = np.clip(float(t), self.time[0], self.time[-1])
        q = np.array([np.interp(t, self.time, self.q[:, i]) for i in range(self.dof)])
        qd = np.array([np.interp(t, self.time, self.qd[:, i]) for i in range(self.dof)])
        qdd = np.array([np.interp(t, self.time, self.qdd[:, i]) for i in range(self.dof)])
        return q, qd, qdd


class JointLinearPath:
    """MoveJ 风格的关节空间直线路径 q(s)。

    这里的 s 是路径进度，不是时间:
        s = 0 表示起点
        s = 1 表示终点

    这个类只描述路径形状，不决定速度和加速度。
    """

    def __init__(self, q_start, q_goal):
        self.q_start = np.asarray(q_start, dtype=np.float64).flatten()
        self.q_goal = np.asarray(q_goal, dtype=np.float64).flatten()
        if self.q_start.shape != self.q_goal.shape:
            raise ValueError("q_start and q_goal must have the same shape")
        self.dq = self.q_goal - self.q_start

    def sample(self, s):
        """根据路径进度 s 采样关节位置 q(s)。"""

        s = np.asarray(s, dtype=np.float64).reshape(-1, 1)
        return self.q_start[None, :] + s * self.dq[None, :]

    def derivative(self, s):
        """返回 dq/ds。直线路径的一阶导是常数。"""

        s = np.asarray(s, dtype=np.float64).reshape(-1)
        return np.repeat(self.dq[None, :], len(s), axis=0)

    def second_derivative(self, s):
        """返回 d2q/ds2。直线路径的二阶导为 0。"""

        s = np.asarray(s, dtype=np.float64).reshape(-1)
        return np.zeros((len(s), self.q_start.size), dtype=np.float64)


def _as_limit_array(value, dof, name, allow_none=False):
    """把标量或数组形式的限制统一成长度为 dof 的数组。"""

    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(dof, float(arr), dtype=np.float64)
    else:
        arr = arr.flatten()
        if arr.size != dof:
            raise ValueError(f"{name} must be a scalar or have length {dof}")
    return arr


class TimeScaling:
    """生成 s(t)、ds/dt、d2s/dt2。"""

    @staticmethod
    def normalized_peaks(method, accel_fraction=0.25):
        """返回归一化曲线在 duration=1 时的最大 |ds/dtau| 和 |d2s/dtau2|。

        自动定时时用它把路径长度换算成实际关节速度/加速度峰值。
        """

        method = method.lower()
        if method in ("linear", "joint_linear"):
            # 线性曲线采样点上加速度为 0，但端点速度突变，实际系统不建议用它执行。
            return 1.0, 0.0
        if method in ("trapezoidal", "trap"):
            tb = float(accel_fraction)
            if not 0.0 < tb < 0.5:
                raise ValueError("accel_fraction must be in (0, 0.5)")
            v = 1.0 / (1.0 - tb)
            a = v / tb
            return v, a
        if method in ("quintic", "minimum_jerk", "minjerk"):
            return 1.875, 10.0 / np.sqrt(3.0)
        if method in ("s_curve", "scurve", "septic"):
            # 七次多项式的峰值用密集采样求，避免手写复杂解析式。
            tau = np.linspace(0.0, 1.0, 10001)
            sd_tau = 140 * tau**3 - 420 * tau**4 + 420 * tau**5 - 140 * tau**6
            sdd_tau = (
                420 * tau**2
                - 1680 * tau**3
                + 2100 * tau**4
                - 840 * tau**5
            )
            return float(np.max(np.abs(sd_tau))), float(np.max(np.abs(sdd_tau)))
        raise ValueError(f"Unsupported time scaling method: {method}")

    @staticmethod
    def sample(config):
        """采样时间缩放曲线。

        返回:
            time: 时间戳。
            s: 路径进度。
            sd: ds/dt。
            sdd: d2s/dt2。
        """

        duration = float(config.duration)
        dt = float(config.dt)
        if duration <= 0:
            raise ValueError("duration must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")

        steps = max(2, int(np.ceil(duration / dt)) + 1)
        time = np.linspace(0.0, duration, steps)
        tau = time / duration
        method = config.method.lower()

        if method in ("linear", "joint_linear"):
            s = tau
            sd = np.full_like(tau, 1.0 / duration)
            sdd = np.zeros_like(tau)
        elif method in ("trapezoidal", "trap"):
            s, sd, sdd = TimeScaling._trapezoidal(tau, duration, config.accel_fraction)
        elif method in ("quintic", "minimum_jerk", "minjerk"):
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
            sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
        elif method in ("s_curve", "scurve", "septic"):
            s = 35 * tau**4 - 84 * tau**5 + 70 * tau**6 - 20 * tau**7
            sd = (140 * tau**3 - 420 * tau**4 + 420 * tau**5 - 140 * tau**6) / duration
            sdd = (
                420 * tau**2
                - 1680 * tau**3
                + 2100 * tau**4
                - 840 * tau**5
            ) / duration**2
        else:
            raise ValueError(f"Unsupported time scaling method: {config.method}")

        return time, s, sd, sdd

    @staticmethod
    def _trapezoidal(tau, duration, accel_fraction):
        """梯形速度曲线: 加速、匀速、减速。"""

        tb = float(accel_fraction)
        if not 0.0 < tb < 0.5:
            raise ValueError("accel_fraction must be in (0, 0.5)")

        v = 1.0 / (1.0 - tb)
        a = v / tb
        s = np.empty_like(tau)
        sd_tau = np.empty_like(tau)
        sdd_tau = np.empty_like(tau)

        accel = tau < tb
        cruise = (tau >= tb) & (tau <= 1.0 - tb)
        decel = tau > 1.0 - tb

        s[accel] = 0.5 * a * tau[accel] ** 2
        sd_tau[accel] = a * tau[accel]
        sdd_tau[accel] = a

        s[cruise] = v * (tau[cruise] - 0.5 * tb)
        sd_tau[cruise] = v
        sdd_tau[cruise] = 0.0

        remaining = 1.0 - tau[decel]
        s[decel] = 1.0 - 0.5 * a * remaining**2
        sd_tau[decel] = a * remaining
        sdd_tau[decel] = -a

        return s, sd_tau / duration, sdd_tau / duration**2


class JointTrajectoryPlanner:
    """简单的关节空间轨迹规划器。

    逻辑只有两步:
        1. 用 JointLinearPath 定义 MoveJ 路径 q(s)。
        2. 用 TimeScaling 定义 s(t)，再用链式法则得到 q(t)、qd(t)、qdd(t)。
    """

    def __init__(self, config=None, limits=None):
        self.config = config or TimeScalingConfig()
        self.limits = limits

    def plan(self, q_start, q_goal, config=None, limits=None):
        """规划从 q_start 到 q_goal 的关节轨迹。"""

        config = config or self.config
        limits = limits or self.limits
        path = JointLinearPath(q_start, q_goal)
        config = self._resolve_config(path, config, limits)
        time, s, sd, sdd = TimeScaling.sample(config)

        q = path.sample(s)
        dq_ds = path.derivative(s)
        d2q_ds2 = path.second_derivative(s)

        # 链式法则: qd = dq/ds * ds/dt
        qd = dq_ds * sd[:, None]
        # 链式法则: qdd = d2q/ds2 * (ds/dt)^2 + dq/ds * d2s/dt2
        qdd = d2q_ds2 * sd[:, None] ** 2 + dq_ds * sdd[:, None]
        return JointTrajectory(config.method, time, q, qd, qdd)

    def _resolve_config(self, path, config, limits):
        """在规划前检查约束，并在需要时自动拉长 duration。"""

        if limits is None:
            return config

        self._check_position_limits(path, limits)
        if not config.auto_duration:
            return config

        duration = self._estimate_duration(path.dq, config, limits)
        return TimeScalingConfig(
            method=config.method,
            duration=max(float(config.duration), duration),
            dt=config.dt,
            accel_fraction=config.accel_fraction,
            auto_duration=config.auto_duration,
        )

    def _check_position_limits(self, path, limits):
        """检查起终点是否超出关节位置上下限。"""

        dof = path.q_start.size
        q_min = _as_limit_array(limits.q_min, dof, "q_min", allow_none=True)
        q_max = _as_limit_array(limits.q_max, dof, "q_max", allow_none=True)
        below_min = q_min is not None and (
            np.any(path.q_start < q_min) or np.any(path.q_goal < q_min)
        )
        above_max = q_max is not None and (
            np.any(path.q_start > q_max) or np.any(path.q_goal > q_max)
        )
        if below_min:
            raise ValueError("q_start or q_goal is below q_min")
        if above_max:
            raise ValueError("q_start or q_goal is above q_max")

    def _estimate_duration(self, dq, config, limits):
        """根据每个关节的速度/加速度上限估算最短可行时长。"""

        dof = dq.size
        qd_max = _as_limit_array(limits.qd_max, dof, "qd_max")
        qdd_max = _as_limit_array(limits.qdd_max, dof, "qdd_max")
        if np.any(qd_max <= 0.0):
            raise ValueError("qd_max must be positive")
        if np.any(qdd_max <= 0.0):
            raise ValueError("qdd_max must be positive")

        sd_peak, sdd_peak = TimeScaling.normalized_peaks(
            config.method,
            config.accel_fraction,
        )
        abs_dq = np.abs(dq)
        duration_from_speed = np.max(abs_dq * sd_peak / qd_max)
        if sdd_peak > 0.0:
            duration_from_accel = np.max(np.sqrt(abs_dq * sdd_peak / qdd_max))
        else:
            duration_from_accel = 0.0
        return float(max(duration_from_speed, duration_from_accel))


# 兼容之前 demo 中的旧名字，本质上就是 JointTrajectoryPlanner。
ConfigurableJointPlanner = JointTrajectoryPlanner
