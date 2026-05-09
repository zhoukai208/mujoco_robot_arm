from dataclasses import dataclass

import mujoco
import numpy as np

from ibvs import (
    DLS_DAMPING,
    JOINT_SPEED_LIMIT,
    OBSERVATION_KP,
    OBSERVATION_REACHED_THRESHOLD,
    ArmIBVS,
)
from xml_paths import PANDA_IBVS_POS_SCENE_XML


POSITION_TARGET_TRACKING_LIMIT = 0.25


@dataclass(frozen=True)
class ServoPositionState:
    features: object
    error: np.ndarray
    v_cam_cv: np.ndarray
    v_ee: np.ndarray
    J_ee: np.ndarray
    q_dot: np.ndarray
    q_target: np.ndarray
    q_current: np.ndarray


class ArmIBVSPosition(ArmIBVS):
    """IBVS controller using MuJoCo position actuators for the Panda joints."""

    def __init__(self, render_path, arm_path):
        super().__init__(render_path, arm_path)
        self.window_name = "IBVS Position Servo"
        self.joint_actuator_ids = np.array(
            [self.model.actuator(f"joint{i}_pos").id for i in range(1, 8)],
            dtype=np.int32,
        )
        self.joint_ctrl_range = self.model.actuator_ctrlrange[self.joint_actuator_ids].copy()
        self.joint_position_target = None

    def runBefore(self):
        super().runBefore()
        q = self.current_arm_q()
        self.joint_position_target = q.copy()
        self.write_joint_position_target(q)
        mujoco.mj_forward(self.model, self.data)

    def current_arm_q(self):
        return self.data.qpos[:7].copy()

    def write_joint_position_target(self, q_target):
        q_target = np.asarray(q_target, dtype=np.float64)
        q_target = np.clip(
            q_target,
            self.joint_ctrl_range[:, 0],
            self.joint_ctrl_range[:, 1],
        )
        self.data.ctrl[self.joint_actuator_ids] = q_target
        self.joint_position_target = q_target.copy()
        return q_target

    def integrate_joint_velocity_command(self, q_dot):
        q = self.current_arm_q()
        if self.joint_position_target is None:
            self.joint_position_target = q.copy()

        tracking_error = self.joint_position_target - q
        if np.linalg.norm(tracking_error) > POSITION_TARGET_TRACKING_LIMIT:
            self.joint_position_target = q.copy()

        dt = float(self.model.opt.timestep)
        delta_q = np.clip(q_dot, -JOINT_SPEED_LIMIT, JOINT_SPEED_LIMIT) * dt
        q_target = self.joint_position_target + delta_q
        return self.write_joint_position_target(q_target)

    def compute_joint_velocity(self, v_ee):
        J_ee = self.end_effector_jacobian()
        damping_matrix = DLS_DAMPING * DLS_DAMPING * np.eye(6)
        q_dot = J_ee.T @ np.linalg.solve(J_ee @ J_ee.T + damping_matrix, v_ee)
        q_dot = np.clip(q_dot, -JOINT_SPEED_LIMIT, JOINT_SPEED_LIMIT)
        return q_dot, J_ee

    def move_to_observation_pose(self, frame):
        q = self.current_arm_q()
        q_err = self.ob_q - q
        q_dot = np.clip(
            OBSERVATION_KP * q_err,
            -JOINT_SPEED_LIMIT,
            JOINT_SPEED_LIMIT,
        )
        self.integrate_joint_velocity_command(q_dot)

        q_diff = np.linalg.norm(q_err)
        if q_diff >= OBSERVATION_REACHED_THRESHOLD:
            print(f"diff: {q_diff}")
            return

        success, _, detection_by_id, missing_ids = self.tag_estimator.detect(frame)
        if not success:
            print(f"Reach obstacle pose, but AprilTag detection is incomplete, missing={missing_ids}")
            return

        target_features = self.detection_image_features(detection_by_id)
        if target_features is None:
            print("Reach obstacle pose, but target feature depth is invalid")
            return

        self.ibvs.set_target(target_features)
        self.reach_ob = True
        self.write_joint_position_target(q)
        print(f"set U: {target_features.u}")
        print(f"set V: {target_features.v}")
        print("Reach obstacle")

    def run_visual_servo_step(self, detection_by_id):
        current_features = self.detection_image_features(detection_by_id)
        if current_features is None:
            return None

        v_cam_cv, error = self.ibvs.compute_camera_velocity(current_features)
        v_ee = self.camera_optical_velocity_to_ee(v_cam_cv)
        q_dot, J_ee = self.compute_joint_velocity(v_ee)
        q_target = self.integrate_joint_velocity_command(q_dot)

        return ServoPositionState(
            features=current_features,
            error=error,
            v_cam_cv=v_cam_cv,
            v_ee=v_ee,
            J_ee=J_ee,
            q_dot=q_dot,
            q_target=q_target,
            q_current=self.current_arm_q(),
        )

    def print_servo_state(self, servo_state):
        super().print_servo_state(servo_state)
        if self.print_counter % 50 == 0:
            print(f"[IBVS Position] Joint position target (7D): {servo_state.q_target}")
            print(f"[IBVS Position] Current joint position (7D): {servo_state.q_current}")
            print(
                "[IBVS Position] Position actuator tracking error norm: "
                f"{np.linalg.norm(servo_state.q_target - servo_state.q_current):.4f}"
            )


if __name__ == "__main__":
    SCENE_XML_PATH = PANDA_IBVS_POS_SCENE_XML

    robot = ArmIBVSPosition(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
