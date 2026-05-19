import argparse
import contextlib
import io
import math
from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

from mujoco_viewer import ArmBaseViewer
from utils import euler2rotmat, format_vec, rot_to_quat
from xml_paths import PANDA_ONLINE_SERVO_SCENE_XML


GRIPPER_OPEN = 255


def clamp_array(value, lower, upper):
    return np.minimum(np.maximum(value, lower), upper)


@dataclass
class ServoState:
    target_q: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    target_vel: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    cmd_time: float = 0.02
    lookahead_time: float = 0.03
    gain: float = 1.2
    last_msg_time: float = 0.0
    has_command: bool = False


class OnlineJointServo:
    """Low-rate joint targets in, smooth high-rate joint position commands out."""

    def __init__(
        self,
        lower_limit,
        upper_limit,
        dt: float,
        max_joint_vel: float = 1.2,
        max_joint_acc: float = 4.0,
        damping_ratio: float = 1.4,
    ):
        self.lower_limit = np.asarray(lower_limit, dtype=np.float64)
        self.upper_limit = np.asarray(upper_limit, dtype=np.float64)
        self.dt = float(dt)
        self.max_joint_vel = float(max_joint_vel)
        self.max_joint_acc = float(max_joint_acc)
        self.damping_ratio = float(damping_ratio)
        self.state = ServoState()
        self.current_q = np.zeros(7, dtype=np.float64)
        self.vel = np.zeros(7, dtype=np.float64)

    def reset(self, current_q, now: float = 0.0):
        self.current_q = np.asarray(current_q, dtype=np.float64).copy()
        self.vel[:] = 0.0
        self.state = ServoState(
            target_q=self.current_q.copy(),
            target_vel=np.zeros(7, dtype=np.float64),
            last_msg_time=float(now),
        )

    def servo_j(
        self,
        target_q,
        now: float,
        cmd_time: float,
        lookahead_time: float,
        gain: float,
    ):
        target_q = clamp_array(np.asarray(target_q, dtype=np.float64), self.lower_limit, self.upper_limit)
        previous_q = self.state.target_q.copy()
        previous_time = self.state.last_msg_time

        self.state.target_q = target_q
        self.state.cmd_time = max(float(cmd_time), self.dt)
        self.state.lookahead_time = max(float(lookahead_time), self.dt)
        self.state.gain = max(float(gain), 0.1)

        if self.state.has_command:
            sample_dt = max(float(now) - previous_time, self.dt)
            raw_vel = (target_q - previous_q) / sample_dt
            self.state.target_vel = np.clip(raw_vel, -self.max_joint_vel, self.max_joint_vel)
        else:
            self.state.target_vel[:] = 0.0

        self.state.last_msg_time = float(now)
        self.state.has_command = True

    def step(self, now: float):
        command_age = float(now) - self.state.last_msg_time
        if command_age > self.state.cmd_time + 0.01:
            self._brake()
        else:
            self._track_target(command_age)

        self.current_q = clamp_array(self.current_q + self.vel * self.dt, self.lower_limit, self.upper_limit)
        return self.current_q.copy()

    def _brake(self):
        decel = self.max_joint_acc * self.dt
        self.vel = np.where(
            self.vel > 0.0,
            np.maximum(0.0, self.vel - decel),
            np.minimum(0.0, self.vel + decel),
        )

    def _track_target(self, command_age: float):
        horizon = min(max(command_age, 0.0) + self.state.lookahead_time, 2.0 * self.state.cmd_time)
        predicted_q = self.state.target_q + self.state.target_vel * horizon
        predicted_q = clamp_array(predicted_q, self.lower_limit, self.upper_limit)

        natural_freq = max(2.0 * math.pi * self.state.gain, 1.0)
        pos_error = predicted_q - self.current_q
        vel_error = self.state.target_vel - self.vel
        acc = natural_freq * natural_freq * pos_error + 2.0 * self.damping_ratio * natural_freq * vel_error
        acc = np.clip(acc, -self.max_joint_acc, self.max_joint_acc)
        self.vel = np.clip(self.vel + acc * self.dt, -self.max_joint_vel, self.max_joint_vel)


class MovingTarget:
    def __init__(self):
        self.center = np.array([0.45, 0.0, 0.28], dtype=np.float64)
        self.ee_rot_offset = np.eye(3, dtype=np.float64)

    def pose(self, t: float):
        pos = self.center + np.array(
            [
                0.08 * math.sin(2.0 * math.pi * 0.18 * t),
                0.10 * math.sin(2.0 * math.pi * 0.13 * t + 0.8),
                0.04 * math.sin(2.0 * math.pi * 0.21 * t + 1.4),
            ],
            dtype=np.float64,
        )
        yaw = 0.25 * math.sin(2.0 * math.pi * 0.10 * t)
        rot = euler2rotmat(math.pi, 0.0, yaw) @ self.ee_rot_offset
        return pos, rot


class ArmOnlineServoController(ArmBaseViewer):
    def __init__(
        self,
        render_path,
        arm_path,
        command_hz: float = 50.0,
        lookahead_time: float = 0.03,
        gain: float = 1.2,
        show_camera: bool = False,
    ):
        super().__init__(render_path, arm_path)
        self.command_period = 1.0 / command_hz
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.show_camera = show_camera
        self.window_name = "Online Servo Camera"

        self.target_name = "servo_target"
        self.target = MovingTarget()
        self.next_ik_time = 0.0
        self.last_ik_success = False
        self.last_ik_q = None
        self.link7_to_ee_pos = np.zeros(3, dtype=np.float64)
        self.link7_to_ee_rot = np.eye(3, dtype=np.float64)

        lower, upper = self._joint_limits()
        self.servo = OnlineJointServo(
            lower_limit=lower,
            upper_limit=upper,
            dt=self.model.opt.timestep,
        )

    def _joint_limits(self):
        lower = []
        upper = []
        for joint_idx in range(7):
            if self.model.jnt_limited[joint_idx]:
                lower.append(self.model.jnt_range[joint_idx, 0])
                upper.append(self.model.jnt_range[joint_idx, 1])
            else:
                lower.append(self.model.actuator_ctrlrange[joint_idx, 0])
                upper.append(self.model.actuator_ctrlrange[joint_idx, 1])
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def _update_link7_to_ee_transform(self):
        link7_pos = self.data.body(self.flange_id).xpos.copy()
        link7_rot = self.data.body(self.flange_id).xmat.reshape(3, 3).copy()
        ee_pos = self.data.body(self.ee_id).xpos.copy()
        ee_rot = self.data.body(self.ee_id).xmat.reshape(3, 3).copy()
        self.link7_to_ee_pos = link7_rot.T @ (ee_pos - link7_pos)
        self.link7_to_ee_rot = link7_rot.T @ ee_rot

    def _link7_target_from_ee_target(self, ee_pos, ee_rot):
        link7_rot = ee_rot @ self.link7_to_ee_rot.T
        link7_pos = ee_pos - link7_rot @ self.link7_to_ee_pos
        return link7_pos, link7_rot

    def _set_target_visual(self, pos, rot):
        quat = rot_to_quat(rot)
        self.mocaps.set_pose(self.target_name, pos, quat)

    def _compute_ik_command(self, now: float, target_pos, target_rot):
        link7_pos, link7_rot = self._link7_target_from_ee_target(target_pos, target_rot)
        q_seed = self.data.qpos[:7].copy() if self.last_ik_q is None else self.last_ik_q
        with contextlib.redirect_stdout(io.StringIO()):
            success, q_ik = self.kinematics.ik(
                q_seed,
                link7_rot,
                link7_pos,
                eps=5e-4,
                IT_MAX=250,
                DT=0.15,
                damp=1e-5,
            )

        self.last_ik_success = bool(success and np.all(np.isfinite(q_ik)))
        if self.last_ik_success:
            self.last_ik_q = np.asarray(q_ik, dtype=np.float64)
            self.servo.servo_j(
                self.last_ik_q,
                now=now,
                cmd_time=self.command_period,
                lookahead_time=self.lookahead_time,
                gain=self.gain,
            )

    def _update_servo_command_if_needed(self, now: float, target_pos, target_rot):
        if now + 1e-12 < self.next_ik_time:
            return

        self._compute_ik_command(now, target_pos, target_rot)
        while self.next_ik_time <= now + 1e-12:
            self.next_ik_time += self.command_period

    def _command_joints(self, q_cmd):
        q_cmd = clamp_array(q_cmd, self.servo.lower_limit, self.servo.upper_limit)
        self.data.ctrl[:7] = q_cmd
        if self.model.nu > 7:
            self.data.ctrl[7:] = GRIPPER_OPEN

    def _print_status(self, target_pos):
        ee_pos = self.data.body(self.ee_id).xpos.copy()
        err = target_pos - ee_pos
        print(
            "\n[Online Servo]\n"
            f"  target_pos: {format_vec(target_pos)}\n"
            f"  ee_pos: {format_vec(ee_pos)}\n"
            f"  pos_err_norm: {np.linalg.norm(err):.4f} m\n"
            f"  ik_success: {self.last_ik_success}\n"
            f"  q_cmd: {format_vec(self.servo.current_q)}"
        )

    def runBefore(self):
        super().runBefore()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.model.opt.timestep = 0.005
        mujoco.mj_forward(self.model, self.data)
        self._update_link7_to_ee_transform()
        self.servo.dt = self.model.opt.timestep
        self.servo.reset(self.data.qpos[:7].copy(), now=self.data.time)
        self.next_ik_time = self.data.time

        target_pos, target_rot = self.target.pose(self.data.time)
        self._set_target_visual(target_pos, target_rot)
        self._compute_ik_command(self.data.time, target_pos, target_rot)

        if self.show_camera:
            cv2.namedWindow(self.window_name)

        print("\nOnline servo demo")
        print("目标球在空间中自主运动，控制器周期读取目标位姿，IK 估计关节角，高频 servo 平滑跟随。")

    def runFunc(self):
        now = self.data.time
        target_pos, target_rot = self.target.pose(now)
        self._set_target_visual(target_pos, target_rot)
        self._update_servo_command_if_needed(now, target_pos, target_rot)
        self._command_joints(self.servo.step(now))

        self.print_counter += 1
        if self.print_counter % 100 == 0:
            self._print_status(target_pos)

    def run_headless(self, duration: float):
        self.runBefore()
        end_time = self.data.time + duration
        while self.data.time < end_time:
            mujoco.mj_forward(self.model, self.data)
            self.runFunc()
            mujoco.mj_step(self.model, self.data)


def parse_args():
    parser = argparse.ArgumentParser(description="Panda online servo target tracking demo.")
    parser.add_argument("--headless", action="store_true", help="run without opening the MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=5.0, help="headless run duration in seconds")
    parser.add_argument("--show-camera", action="store_true", help="show eye-in-hand camera window")
    parser.add_argument("--command-hz", type=float, default=50.0, help="IK command update rate")
    parser.add_argument("--lookahead-time", type=float, default=0.03, help="servo lookahead time")
    parser.add_argument("--gain", type=float, default=1.2, help="servo tracking gain")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    robot = ArmOnlineServoController(
        PANDA_ONLINE_SERVO_SCENE_XML,
        PANDA_ONLINE_SERVO_SCENE_XML,
        command_hz=args.command_hz,
        lookahead_time=args.lookahead_time,
        gain=args.gain,
        show_camera=args.show_camera,
    )
    if args.headless:
        robot.run_headless(args.duration)
    else:
        robot.run_loop()
