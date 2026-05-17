import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from time_scaling_planner import JointLimitConfig, JointTrajectoryPlanner, TimeScalingConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "trajectory_time_scaling"


def parse_args():
    """解析命令行参数，便于切换时间参数化方式和采样参数。"""

    parser = argparse.ArgumentParser(
        description="Compare time parameterization methods on one joint-space path."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["linear", "trapezoidal", "quintic", "s_curve"],
        help="Time parameterization methods to compare.",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--fixed-duration",
        action="store_true",
        help="Use duration directly instead of stretching it to satisfy joint limits.",
    )
    parser.add_argument(
        "--qd-max",
        type=float,
        default=1.0,
        help="Scalar joint velocity limit used by automatic duration.",
    )
    parser.add_argument(
        "--qdd-max",
        type=float,
        default=2.0,
        help="Scalar joint acceleration limit used by automatic duration.",
    )
    parser.add_argument(
        "--accel-fraction",
        type=float,
        default=0.25,
        help="Acceleration/deceleration phase ratio for trapezoidal scaling.",
    )
    parser.add_argument(
        "--joint-index",
        type=int,
        default=3,
        help="Joint index shown in the compact comparison plot.",
    )
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures and trajectory records.",
    )
    return parser.parse_args()


def make_joint_request():
    """构造关节空间规划起点和终点。

    这里故意不通过 IK 生成目标点，目的是让 demo 只关注:
        关节空间路径 q(s) + 不同时间参数化 s(t)
    """

    q_start = np.array([0.0, 0.0, 0.0, -1.44, 0.0, 1.5, 0.9], dtype=np.float64)
    q_goal = np.array([0.35, -0.45, 0.25, -1.85, 0.2, 1.95, 0.55], dtype=np.float64)
    return q_start, q_goal


def make_default_limits(qd_max, qdd_max):
    """构造 Panda 七轴的基础位置/速度/加速度限制。

    这里的速度和加速度上限用于教学 demo，可通过命令行调松或调紧。
    """

    q_min = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        dtype=np.float64,
    )
    q_max = np.array(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        dtype=np.float64,
    )
    return JointLimitConfig(q_min=q_min, q_max=q_max, qd_max=qd_max, qdd_max=qdd_max)


def plan_with_methods(
    q_start,
    q_goal,
    methods,
    duration,
    dt,
    accel_fraction,
    auto_duration,
    limits,
):
    """同一条关节直线路径上分别应用多种时间参数化方法。"""

    trajectories = []
    for method in methods:
        config = TimeScalingConfig(
            method=method,
            duration=duration,
            dt=dt,
            accel_fraction=accel_fraction,
            auto_duration=auto_duration,
        )
        planner = JointTrajectoryPlanner(config, limits=limits)
        trajectories.append(planner.plan(q_start, q_goal))
    return trajectories


def save_trajectory_csv(trajectory, output_dir):
    """保存单条轨迹的 time、q、dq、ddq，便于后续分析或导入控制器。"""

    dof = trajectory.q.shape[1]
    output_path = output_dir / f"{trajectory.method}_joint_trajectory.csv"
    header = (
        ["time"]
        + [f"q{i + 1}" for i in range(dof)]
        + [f"dq{i + 1}" for i in range(dof)]
        + [f"ddq{i + 1}" for i in range(dof)]
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t in enumerate(trajectory.time):
            writer.writerow(
                [t]
                + trajectory.q[i].tolist()
                + trajectory.qd[i].tolist()
                + trajectory.qdd[i].tolist()
            )
    return output_path


def save_records(trajectories, output_dir):
    """记录所有方法的完整轨迹数据和摘要指标。"""

    csv_paths = [save_trajectory_csv(traj, output_dir) for traj in trajectories]
    npz_path = output_dir / "joint_trajectory_comparison.npz"
    np.savez(
        npz_path,
        **{f"{traj.method}_time": traj.time for traj in trajectories},
        **{f"{traj.method}_q": traj.q for traj in trajectories},
        **{f"{traj.method}_dq": traj.qd for traj in trajectories},
        **{f"{traj.method}_ddq": traj.qdd for traj in trajectories},
    )

    summary_path = output_dir / "trajectory_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "duration", "max_abs_dq", "max_abs_ddq"])
        for traj in trajectories:
            writer.writerow(
                [
                    traj.method,
                    float(traj.duration),
                    float(np.max(np.abs(traj.qd))),
                    float(np.max(np.abs(traj.qdd))),
                ]
            )
    return csv_paths, npz_path, summary_path


def plot_selected_joint(trajectories, joint_index, output_path):
    """绘制单个关节的 q、dq、ddq，对比不同时间参数化的曲线形状。"""

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for traj in trajectories:
        axes[0].plot(traj.time, traj.q[:, joint_index], label=traj.method)
        axes[1].plot(traj.time, traj.qd[:, joint_index], label=traj.method)
        axes[2].plot(traj.time, traj.qdd[:, joint_index], label=traj.method)

    axes[0].set_ylabel("q (rad)")
    axes[1].set_ylabel("dq (rad/s)")
    axes[2].set_ylabel("ddq (rad/s^2)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title(f"Joint {joint_index + 1}: q / dq / ddq")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def plot_all_joints(trajectories, field_name, ylabel, output_path):
    """按关节分别绘制 q、dq 或 ddq，完整对比七个关节的轨迹。"""

    dof = trajectories[0].q.shape[1]
    rows = 4
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for joint_index in range(dof):
        ax = axes[joint_index]
        for traj in trajectories:
            values = getattr(traj, field_name)
            ax.plot(traj.time, values[:, joint_index], label=traj.method)
        ax.set_title(f"joint {joint_index + 1}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes[dof:]:
        ax.axis("off")
    axes[-2].set_xlabel("time (s)")
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def print_summary(trajectories, csv_paths, npz_path, summary_path):
    """打印每种规划方法的最大关节速度和最大关节加速度。"""

    print("\nTrajectory summary")
    print("method        duration   max|dq|    max|ddq|")
    for traj in trajectories:
        max_dq = np.max(np.abs(traj.qd))
        max_ddq = np.max(np.abs(traj.qdd))
        print(f"{traj.method:<12} {traj.duration:8.3f}  {max_dq:8.4f}  {max_ddq:9.4f}")

    print("\nSaved records:")
    for path in csv_paths:
        print(f"- {path}")
    print(f"- {npz_path}")
    print(f"- {summary_path}")


def main():
    """关节空间路径规划 demo: 同一路径 + 多种时间参数化。"""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q_start, q_goal = make_joint_request()
    limits = make_default_limits(args.qd_max, args.qdd_max)
    trajectories = plan_with_methods(
        q_start=q_start,
        q_goal=q_goal,
        methods=args.methods,
        duration=args.duration,
        dt=args.dt,
        accel_fraction=args.accel_fraction,
        auto_duration=not args.fixed_duration,
        limits=limits,
    )

    joint_index = int(np.clip(args.joint_index, 0, len(q_start) - 1))
    plot_selected_joint(
        trajectories,
        joint_index,
        args.output_dir / "selected_joint_q_dq_ddq.png",
    )
    plot_all_joints(trajectories, "q", "q (rad)", args.output_dir / "all_joints_q.png")
    plot_all_joints(
        trajectories,
        "qd",
        "dq (rad/s)",
        args.output_dir / "all_joints_dq.png",
    )
    plot_all_joints(
        trajectories,
        "qdd",
        "ddq (rad/s^2)",
        args.output_dir / "all_joints_ddq.png",
    )

    csv_paths, npz_path, summary_path = save_records(trajectories, args.output_dir)
    print_summary(trajectories, csv_paths, npz_path, summary_path)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
