import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from apriltag_detector import AprilTagPoseEstimator
from ibvs import DLS_DAMPING, IBVSController, ImageFeatures, JOINT_SPEED_LIMIT, adjoint
from mujoco_viewer import CAMERA_WINDOW_NAME, CAMERA_WINDOW_POS, CAMERA_WINDOW_SIZE, ArmBaseViewer
from utils import euler2rotmat, format_vec

try:
    from .view_scene import make_loadable_scene_xml
except ImportError:
    from view_scene import make_loadable_scene_xml

JOINT_REACHED_TOL = 0.008
GRIPPER_OPEN = 255
GRIPPER_CLOSE = 0
GRIPPER_CLOSE_MIN_STEPS = 180
GRIPPER_CLOSE_TIMEOUT_STEPS = 500
APPROACH_STEPS_DOWN = 35
DESCEND_STEPS = 45
PATH_WAYPOINT_TOL = 0.006
POSITION_TARGET_TRACKING_LIMIT = 0.25
IBVS_ALIGNED_THRESHOLD_PX = 10.0
ENABLE_IBVS = True
INSERT_DESCEND_STEP = 0.001
INSERT_MAX_DESCEND = 0.45
RELEASE_OPEN_STEPS = 80
INSERT_FORCE_THRESHOLD = 5.0
INSERT_CONTACT_FORCE_THRESHOLD = 1.0
INSERT_CONTACT_DESCEND_SCALE = 0.2
ENABLE_INSERT_ADMITTANCE = True
INSERT_DEPTH_Z_TOL = 0.006
INSERT_DEPTH_XY_TOL = 0.005
INSERT_ADMITTANCE_M = np.array([0.5, 0.5], dtype=np.float64)
INSERT_ADMITTANCE_D = np.array([20.0, 20.0], dtype=np.float64)
INSERT_ADMITTANCE_K = np.array([10.0, 10.0], dtype=np.float64)
INSERT_ADMITTANCE_MAX_OFFSET = 0.006
INSERT_ADMITTANCE_MAX_VELOCITY = 0.03
INSERT_ADMITTANCE_FORCE_SIGN = -1.0
APPROACH_CLEARANCE = 0.14
SAFE_APPROACH_CLEARANCE = 0.28
HOLE_OBSERVATION_EXTRA_HEIGHT = 0.22
HOLE_OBSERVATION_OFFSET_XY = np.array([0.0, 0.0], dtype=np.float64)
HOLE_OBSERVATION_RANDOM_XY_RANGE = np.array(
    [[-0.245, 0.075], [-0.145, 0.145]],
    dtype=np.float64,
)
HOLE_OBSERVATION_RANDOM_RPY_RANGE = np.deg2rad(
    np.array([[-4.0, 4.0], [-4.0, 4.0], [-8.0, 8.0]], dtype=np.float64)
)
HOLE_APPROACH_LOCAL_Z = 0.20
INSERT_TEST_OFFSET_XY = np.array([0.0, 0.01], dtype=np.float64)
CAMERA_CLOCKWISE_YAW_OFFSET = -np.deg2rad(45.0)
TAG_IDS = (0, 1, 2, 3)
IBVS_TARGET_PIXELS_PATH = Path(__file__).resolve().parents[2] / "config/peg_in_hole_ibvs_target_pixels.yaml"


class GraspState(Enum):
    PLAN = auto()
    MOVE_ABOVE_PIN = auto()
    DESCEND = auto()
    CLOSE_GRIPPER = auto()
    LIFT = auto()
    CHECK_IBVS_ERROR = auto()
    INSERT_DESCEND = auto()
    INSERT_HOLD = auto()
    RELEASE_GRIPPER = auto()
    RETREAT_AFTER_RELEASE = auto()
    RETREAT_ABORT = auto()
    DONE = auto()


@dataclass
class JointPositionCommandResult:
    dq: np.ndarray
    q_err: np.ndarray
    q_target: np.ndarray
    q_ctrl: np.ndarray
    reached: bool

    @property
    def q_err_norm(self):
        return float(np.linalg.norm(self.q_err))

    @property
    def dq_norm(self):
        return float(np.linalg.norm(self.dq))


class PositionArmController:
    """Thin wrapper around MuJoCo Panda position actuators."""

    def __init__(self, model, data, gripper_command=None):
        self.model = model
        self.data = data
        self.gripper_command = gripper_command
        self.joint_actuator_ids = np.array(
            [self.model.actuator(f"joint{i}_pos").id for i in range(1, 8)],
            dtype=np.int32,
        )
        self.joint_ctrl_range = self.model.actuator_ctrlrange[self.joint_actuator_ids].copy()
        self.joint_kp = self.model.actuator_gainprm[self.joint_actuator_ids, 0].copy()

    def set_gripper(self, command):
        if self.model.nu > 7:
            self.data.ctrl[7:] = command

    def _apply_gripper_command(self):
        if self.gripper_command is not None:
            self.set_gripper(self.gripper_command())

    def move_to_joint(self, q_target, reached_tol=JOINT_REACHED_TOL):
        q_target = np.asarray(q_target, dtype=np.float64)
        q_target = np.clip(
            q_target,
            self.joint_ctrl_range[:, 0],
            self.joint_ctrl_range[:, 1],
        )

        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()
        q_err = q_target - q
        q_ctrl = self._gravity_compensated_position_target(q_target)
        self.data.ctrl[self.joint_actuator_ids] = q_ctrl
        self._apply_gripper_command()

        return JointPositionCommandResult(
            dq=dq,
            q_err=q_err,
            q_target=q_target.copy(),
            q_ctrl=q_ctrl.copy(),
            reached=np.linalg.norm(q_err) < reached_tol,
        )

    def _gravity_compensated_position_target(self, q_target):
        qfrc_bias = self.data.qfrc_bias[:7].copy()
        q_ctrl = q_target + qfrc_bias / self.joint_kp
        return np.clip(
            q_ctrl,
            self.joint_ctrl_range[:, 0],
            self.joint_ctrl_range[:, 1],
        )


class PositionJointPathFollower:
    """Replay a joint-position path and wait for the final waypoint to settle."""

    def __init__(self, controller, reached_tol):
        self.controller = controller
        self.reached_tol = reached_tol
        self.path = []
        self.index = 0

    def start(self, path):
        self.path = [np.asarray(q, dtype=np.float64).copy() for q in path]
        self.index = 0

    @property
    def count(self):
        return len(self.path)

    def step(self):
        if not self.path:
            return True, None

        target = self.path[min(self.index, len(self.path) - 1)]
        result = self.controller.move_to_joint(target, self.reached_tol)

        if self.index < len(self.path) - 1:
            self.index += 1
            return False, result

        if result.reached:
            self.index += 1
            return True, result
        return False, result


class PegInHoleGraspDemo(ArmBaseViewer):
    def __init__(self, render_path):
        super().__init__(render_path, render_path)
        self.state = GraspState.PLAN
        self.state_step = 0
        self.pose_targets = {}
        self.approach_q_path = []
        self.descend_q_path = []
        self.lift_q_path = []
        self.grasp_rot = euler2rotmat(np.pi, 0.0, CAMERA_CLOCKWISE_YAW_OFFSET)
        self.target_tag_pixels = self._load_target_tag_pixels(IBVS_TARGET_PIXELS_PATH)
        self.tag_detector = AprilTagPoseEstimator(TAG_IDS)
        self.fx, self.fy, self.cx, self.cy = self.calc_intrinsics()
        self.ibvs = IBVSController(self.fx, self.fy, self.cx, self.cy, gain=0.8, stop_threshold_px=8.0)
        self._set_ibvs_target_pixels()
        self.T_MJ_CAMERA_FROM_CV_CAMERA = np.eye(4, dtype=np.float64)
        self.T_MJ_CAMERA_FROM_CV_CAMERA[:3, :3] = np.diag([1.0, -1.0, -1.0])
        self.joint_position_target = None
        self.insert_start_q = None
        self.insert_start_pos = None
        self.insert_target_pos = None
        self.insert_force_bias = np.zeros(3, dtype=np.float64)
        self.insert_admittance_offset_xy = np.zeros(2, dtype=np.float64)
        self.insert_admittance_velocity_xy = np.zeros(2, dtype=np.float64)
        self.insert_admittance_active = False
        self.insert_succeeded = False
        self.insert_stop_reason = "未停止"
        self.observation_xy_offset = None
        self.observation_rpy_offset = np.zeros(3, dtype=np.float64)
        self.rng = np.random.default_rng()
        self.arm_controller = PositionArmController(
            self.model,
            self.data,
            gripper_command=self._gripper_command_for_state,
        )
        self.approach_follower = PositionJointPathFollower(self.arm_controller, PATH_WAYPOINT_TOL)
        self.descend_follower = PositionJointPathFollower(self.arm_controller, PATH_WAYPOINT_TOL)
        self.lift_follower = PositionJointPathFollower(self.arm_controller, PATH_WAYPOINT_TOL)
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg_body")
        self.hole_base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hole_base")
        self.peg_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_collision")
        self.peg_grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_grasp_site")
        self.peg_top_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_top_site")
        self.peg_bottom_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_bottom_site")
        self.hole_bottom_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "hole_bottom_site",
        )
        self.grasp_weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "peg_grasp_weld")
        self.ee_force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
        self.ee_force_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.grasp_failed_reported = False
        if self.kinematics.EE_FRAME_ID != self.kinematics.FRAME_ID:
            self.kinematics.FRAME_ID = self.kinematics.EE_FRAME_ID
            print("[Setup] IK target frame set to ee_center_body")

    def runBefore(self):
        super().runBefore()
        self.model.opt.gravity[:] = [0.0, 0.0, -9.81]
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        mujoco.mj_forward(self.model, self.data)
        self.arm_controller.move_to_joint(self.data.qpos[:7].copy())
        self.arm_controller.set_gripper(GRIPPER_OPEN)
        if self.grasp_weld_id >= 0:
            self.data.eq_active[self.grasp_weld_id] = 0
        mujoco.mj_forward(self.model, self.data)
        print("[Start] plan -> grasp -> random observation -> IBVS align -> insert")

    def _load_target_tag_pixels(self, path):
        if not path.exists():
            raise FileNotFoundError(f"IBVS target pixel yaml not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        tag_pixels = payload.get("tag_pixels")
        if tag_pixels is None:
            raise ValueError(f"Missing 'tag_pixels' in {path}")

        targets = {}
        if isinstance(tag_pixels, dict):
            items = [{"id": tag_id, "pixel": pixel} for tag_id, pixel in tag_pixels.items()]
        else:
            items = tag_pixels

        for item in items:
            tag_id = int(item["id"])
            pixel = np.asarray(item["pixel"], dtype=np.float64)
            if pixel.shape != (2,):
                raise ValueError(f"Tag {tag_id} pixel must be [u, v], got shape={pixel.shape}")
            targets[tag_id] = pixel

        missing_ids = [tag_id for tag_id in TAG_IDS if tag_id not in targets]
        if missing_ids:
            raise ValueError(f"Missing target pixels for tag ids {missing_ids} in {path}")

        target_text = ", ".join(
            f"{tag_id}=({targets[tag_id][0]:.1f},{targets[tag_id][1]:.1f})"
            for tag_id in TAG_IDS
        )
        print(f"[IBVS] loaded target pixels from {path}: {target_text}")
        return targets

    def _set_ibvs_target_pixels(self):
        uv = np.asarray([self.target_tag_pixels[tag_id] for tag_id in TAG_IDS], dtype=np.float64)
        self.ibvs.target_u = uv[:, 0].copy()
        self.ibvs.target_v = uv[:, 1].copy()
        self.ibvs.target_z = None

    def _make_pose_targets(self):
        mujoco.mj_forward(self.model, self.data)
        grasp = self.data.site_xpos[self.peg_grasp_site_id].copy()
        hole_base = self.data.xpos[self.hole_base_body_id].copy()
        hole_approach = hole_base + np.array([0.0, 0.0, HOLE_APPROACH_LOCAL_Z], dtype=np.float64)
        if not ENABLE_IBVS:
            self.observation_xy_offset = INSERT_TEST_OFFSET_XY.copy()
            self.observation_rpy_offset[:] = 0.0
            print(
                "[Plan] IBVS skipped, direct insert xy offset: "
                f"x={self.observation_xy_offset[0]:.4f}, y={self.observation_xy_offset[1]:.4f}"
            )
        elif self.observation_xy_offset is None:
            random_offset = np.array(
                [
                    self.rng.uniform(*HOLE_OBSERVATION_RANDOM_XY_RANGE[0]),
                    self.rng.uniform(*HOLE_OBSERVATION_RANDOM_XY_RANGE[1]),
                ],
                dtype=np.float64,
            )
            self.observation_xy_offset = HOLE_OBSERVATION_OFFSET_XY + random_offset
            self.observation_rpy_offset = np.array(
                [
                    self.rng.uniform(*HOLE_OBSERVATION_RANDOM_RPY_RANGE[0]),
                    self.rng.uniform(*HOLE_OBSERVATION_RANDOM_RPY_RANGE[1]),
                    self.rng.uniform(*HOLE_OBSERVATION_RANDOM_RPY_RANGE[2]),
                ],
                dtype=np.float64,
            )
            print(
                "[Plan] random observation xy offset: "
                f"x={self.observation_xy_offset[0]:.4f}, y={self.observation_xy_offset[1]:.4f}"
            )
            print(
                "[Plan] random observation rpy offset(deg): "
                f"roll={np.rad2deg(self.observation_rpy_offset[0]):.2f}, "
                f"pitch={np.rad2deg(self.observation_rpy_offset[1]):.2f}, "
                f"yaw={np.rad2deg(self.observation_rpy_offset[2]):.2f}"
            )
        above = grasp + np.array([0.0, 0.0, APPROACH_CLEARANCE], dtype=np.float64)
        high_above_hole = hole_approach + np.array(
            [self.observation_xy_offset[0], self.observation_xy_offset[1], HOLE_OBSERVATION_EXTRA_HEIGHT],
            dtype=np.float64,
        )
        return {
            GraspState.MOVE_ABOVE_PIN: above,
            GraspState.DESCEND: grasp,
            GraspState.LIFT: high_above_hole,
        }

    def _plan_targets(self):
        pose_targets = self._make_pose_targets()
        self.pose_targets = pose_targets

        safe_above = pose_targets[GraspState.DESCEND] + np.array(
            [0.0, 0.0, SAFE_APPROACH_CLEARANCE],
            dtype=np.float64,
        )
        success, q_safe = self.kinematics.ik(self.data.qpos[:7].copy(), self.grasp_rot, safe_above)
        if not success:
            raise RuntimeError(f"IK failed for safe approach, target_pos={safe_above}")
        self.approach_q_path = [np.asarray(q_safe, dtype=np.float64).copy()]
        self.approach_q_path.extend(
            self._plan_cartesian_path(
                "MOVE_ABOVE_PIN",
                [safe_above, pose_targets[GraspState.MOVE_ABOVE_PIN]],
                self.approach_q_path[-1],
                [APPROACH_STEPS_DOWN],
            )
        )
        self.descend_q_path = self._plan_cartesian_path(
            "DESCEND",
            [pose_targets[GraspState.MOVE_ABOVE_PIN], pose_targets[GraspState.DESCEND]],
            self.approach_q_path[-1],
            [DESCEND_STEPS],
        )
        self.lift_q_path = self._plan_cartesian_path(
            "LIFT",
            [pose_targets[GraspState.DESCEND], pose_targets[GraspState.LIFT]],
            self.descend_q_path[-1],
            [DESCEND_STEPS],
            rpy_offset_end=self.observation_rpy_offset,
        )
        self._set_state(GraspState.MOVE_ABOVE_PIN)

    def _plan_cartesian_path(self, label, points, q_start, steps_per_segment, rpy_offset_end=None):
        path = []
        q_seed = np.asarray(q_start, dtype=np.float64)
        total_steps = sum(steps_per_segment)
        planned_steps = 0
        rpy_offset_end = (
            np.zeros(3, dtype=np.float64)
            if rpy_offset_end is None
            else np.asarray(rpy_offset_end, dtype=np.float64)
        )
        for segment_idx, steps in enumerate(steps_per_segment):
            start_pos = np.asarray(points[segment_idx], dtype=np.float64)
            target_pos = np.asarray(points[segment_idx + 1], dtype=np.float64)
            for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
                pos = (1.0 - t) * start_pos + t * target_pos
                planned_steps += 1
                rpy_offset = rpy_offset_end * (planned_steps / total_steps)
                target_rot = euler2rotmat(
                    np.pi + rpy_offset[0],
                    rpy_offset[1],
                    CAMERA_CLOCKWISE_YAW_OFFSET + rpy_offset[2],
                )
                success, q_target = self.kinematics.ik(q_seed, target_rot, pos)
                if not success:
                    raise RuntimeError(f"IK failed for {label} waypoint, target_pos={pos}")
                q_seed = np.asarray(q_target, dtype=np.float64)
                path.append(q_seed.copy())
        print(
            f"[Plan] {label}: "
            f"{len(path)} waypoints from={format_vec(points[0])} to={format_vec(points[-1])}"
        )
        return path

    def _set_state(self, state):
        self.state = state
        self.state_step = 0
        if state == GraspState.MOVE_ABOVE_PIN:
            self.approach_follower.start(self.approach_q_path)
            print(
                "[Move] approach: "
                f"to={format_vec(self.pose_targets[GraspState.MOVE_ABOVE_PIN])}"
            )
        if state == GraspState.DESCEND:
            self.descend_follower.start(self.descend_q_path)
            print(
                "[Move] descend to pin: "
                f"from={format_vec(self.pose_targets[GraspState.MOVE_ABOVE_PIN])} "
                f"to={format_vec(self.pose_targets[GraspState.DESCEND])}"
            )
        if state == GraspState.CLOSE_GRIPPER:
            self.grasp_failed_reported = False
        if state == GraspState.LIFT:
            self.lift_follower.start(self.lift_q_path)
            print(
                "[Move] observation pose: "
                f"from={format_vec(self.pose_targets[GraspState.DESCEND])} "
                f"to={format_vec(self.pose_targets[GraspState.LIFT])}"
            )
        if state == GraspState.CHECK_IBVS_ERROR:
            self.joint_position_target = self.data.qpos[:7].copy()
            print("[IBVS] aligning to loaded target pixels")
        if state == GraspState.INSERT_DESCEND:
            self.insert_start_q = self.data.qpos[:7].copy()
            self.insert_start_pos, _ = self._get_ee_pose()
            self.insert_target_pos = self.insert_start_pos.copy()
            self.insert_force_bias = self._ee_force()
            self.insert_admittance_offset_xy[:] = 0.0
            self.insert_admittance_velocity_xy[:] = 0.0
            self.insert_admittance_active = False
            self.insert_succeeded = False
            self.insert_stop_reason = "下探中"
            print(f"[Insert] start: ee={format_vec(self.insert_start_pos)}")
        if state == GraspState.INSERT_HOLD:
            self.joint_position_target = self.data.qpos[:7].copy()
            ee_pos, _ = self._get_ee_pose()
            print(f"[停止] 原因：{self.insert_stop_reason}，保持当前位置 ee={format_vec(ee_pos)}")
        if state == GraspState.RELEASE_GRIPPER:
            if self.grasp_weld_id >= 0:
                self.data.eq_active[self.grasp_weld_id] = 0
            print(f"[松手] 原因：{self.insert_stop_reason}，松开夹爪")
        if state == GraspState.RETREAT_ABORT:
            print(f"[停止] 原因：{self.insert_stop_reason}，回到下探起点")

    def _servo_to_joint_target(self, q_target, reached_tol):
        result = self.arm_controller.move_to_joint(q_target, reached_tol)
        if self.print_counter % 50 == 0:
            print(
                f"[Move] {self.state.name}: "
                f"q_err_norm={result.q_err_norm:.4f} "
                f"dq_norm={result.dq_norm:.4f}"
            )
        return result.reached

    def _follow_joint_path(self, label, follower, include_pin=False):
        done, result = follower.step()
        ee_pos, _ = self._get_ee_pose()
        if self.print_counter % 50 == 0:
            line = (
                f"[Move] {label}: "
                f"waypoint={min(follower.index + 1, follower.count)}/{follower.count} "
                f"ee={format_vec(ee_pos)}"
            )
            if result is not None:
                line += f" q_err_norm={result.q_err_norm:.4f}"
            if include_pin:
                line += f" pin={format_vec(self.data.xpos[self.peg_body_id])}"
            print(line)
        return done

    def _gripper_command_for_state(self):
        if self.state in (
            GraspState.CLOSE_GRIPPER,
            GraspState.LIFT,
            GraspState.CHECK_IBVS_ERROR,
            GraspState.INSERT_DESCEND,
            GraspState.INSERT_HOLD,
            GraspState.RETREAT_ABORT,
        ):
            return GRIPPER_CLOSE
        if self.state == GraspState.DONE and not self.insert_succeeded:
            return GRIPPER_CLOSE
        return GRIPPER_OPEN

    def _close_gripper_until_contact(self):
        self._servo_to_joint_target(self.descend_q_path[-1], JOINT_REACHED_TOL)
        self.state_step += 1
        left_contact, right_contact = self._finger_contacts_with_pin()
        if self.print_counter % 50 == 0 or self.state_step == 1:
            print(
                "[Grasp] closing: "
                f"step={self.state_step} "
                f"left_contact={left_contact} right_contact={right_contact} "
                f"pin_pos={format_vec(self.data.xpos[self.peg_body_id])}"
            )

        contact_ready = left_contact and right_contact and self.state_step >= GRIPPER_CLOSE_MIN_STEPS
        timed_out = self.state_step >= GRIPPER_CLOSE_TIMEOUT_STEPS
        if contact_ready:
            self._activate_grasp_weld()
            self._set_state(GraspState.LIFT)
        elif timed_out and not self.grasp_failed_reported:
            print("[Grasp] failed: pin was not contacted by both fingers")
            self.grasp_failed_reported = True

    def _activate_grasp_weld(self):
        if self.grasp_weld_id < 0:
            raise RuntimeError("peg_grasp_weld equality is missing from the model")

        ee_pos, ee_quat = self._get_ee_pose()
        peg_pos = self.data.xpos[self.peg_body_id].copy()
        peg_quat = self.data.xquat[self.peg_body_id].copy()

        ee_quat_inv = np.zeros(4, dtype=np.float64)
        mujoco.mju_negQuat(ee_quat_inv, ee_quat)
        rel_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(rel_quat, ee_quat_inv, peg_quat)

        ee_rot = self.data.body(self.ee_id).xmat.reshape(3, 3).copy()
        rel_pos = ee_rot.T @ (peg_pos - ee_pos)
        # MuJoCo weld equality data stores relpose at [3:10]:
        # [0:3] is unused/anchor-style data, [3:6] rel_pos, [6:10] rel_quat,
        # [10] torquescale.
        self.model.eq_data[self.grasp_weld_id, 3:6] = rel_pos
        self.model.eq_data[self.grasp_weld_id, 6:10] = rel_quat
        self.data.eq_active[self.grasp_weld_id] = 1
        mujoco.mj_forward(self.model, self.data)
        print(
            "[Grasp] weld activated: "
            f"rel_pos={format_vec(rel_pos)} rel_quat={format_vec(rel_quat)}"
        )

    def _finger_contacts_with_pin(self):
        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.peg_geom_id:
                other_geom = contact.geom2
            elif contact.geom2 == self.peg_geom_id:
                other_geom = contact.geom1
            else:
                continue

            other_body = self.model.geom_bodyid[other_geom]
            if other_body == self.left_finger_body_id:
                left_contact = True
            elif other_body == self.right_finger_body_id:
                right_contact = True
        return left_contact, right_contact

    def _get_ee_pose(self):
        return self.data.body(self.ee_id).xpos.copy(), self.data.body(self.ee_id).xquat.copy()

    def _tag_depths_from_projection(self):
        cam_R = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        cam_t = self.data.cam_xpos[self.camera_id].copy()
        world_R_cam = cam_R.T

        depths = []
        for tag_id in TAG_IDS:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hole_tag_{tag_id}")
            point_world = self.data.xpos[body_id].copy()
            point_cam = world_R_cam @ (point_world - cam_t)
            depth = float(-point_cam[2])
            if depth <= 1e-6:
                return None
            depths.append(depth)
        return np.asarray(depths, dtype=np.float64)

    def _features_from_detections(self, detections_by_id):
        depths = self._tag_depths_from_projection()
        if depths is None:
            return None

        centers = np.asarray(
            [detections_by_id[tag_id].center for tag_id in TAG_IDS],
            dtype=np.float64,
        )
        return ImageFeatures(uv=centers, z=depths)

    def _get_body_transform(self, body_id):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body_id]
        return T

    def _get_camera_transform(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        T[:3, 3] = self.data.cam_xpos[self.camera_id]
        return T

    def _camera_optical_velocity_to_ee(self, v_cam_cv):
        T_world_ee = self._get_body_transform(self.ee_id)
        T_world_cam_mj = self._get_camera_transform()
        T_ee_cam_cv = (
            np.linalg.inv(T_world_ee)
            @ T_world_cam_mj
            @ self.T_MJ_CAMERA_FROM_CV_CAMERA
        )
        return (adjoint(T_ee_cam_cv) @ np.asarray(v_cam_cv, dtype=np.float64).reshape(6, 1)).flatten()

    def _end_effector_jacobian(self):
        Jp = np.zeros((3, self.model.nv), dtype=np.float64)
        Jr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, Jp, Jr, self.ee_id)

        J_world = np.vstack([Jp, Jr])[:, :7]
        ee_R_world = self._get_body_transform(self.ee_id)[:3, :3].T
        X_ee_world = np.zeros((6, 6), dtype=np.float64)
        X_ee_world[:3, :3] = ee_R_world
        X_ee_world[3:, 3:] = ee_R_world
        return X_ee_world @ J_world

    def _compute_joint_velocity(self, v_ee):
        J_ee = self._end_effector_jacobian()
        damping_matrix = DLS_DAMPING * DLS_DAMPING * np.eye(6)
        q_dot = J_ee.T @ np.linalg.solve(J_ee @ J_ee.T + damping_matrix, v_ee)
        return np.clip(q_dot, -JOINT_SPEED_LIMIT, JOINT_SPEED_LIMIT), J_ee

    def _write_joint_position_target(self, q_target):
        q_target = np.asarray(q_target, dtype=np.float64)
        q_target = np.clip(
            q_target,
            self.arm_controller.joint_ctrl_range[:, 0],
            self.arm_controller.joint_ctrl_range[:, 1],
        )
        q_ctrl = self.arm_controller._gravity_compensated_position_target(q_target)
        self.data.ctrl[self.arm_controller.joint_actuator_ids] = q_ctrl
        self.arm_controller._apply_gripper_command()
        self.joint_position_target = q_target.copy()
        return q_target

    def _integrate_joint_velocity_command(self, q_dot):
        q = self.data.qpos[:7].copy()
        if self.joint_position_target is None:
            self.joint_position_target = q.copy()

        tracking_error = self.joint_position_target - q
        if np.linalg.norm(tracking_error) > POSITION_TARGET_TRACKING_LIMIT:
            self.joint_position_target = q.copy()

        dt = float(self.model.opt.timestep)
        delta_q = np.clip(q_dot, -JOINT_SPEED_LIMIT, JOINT_SPEED_LIMIT) * dt
        return self._write_joint_position_target(self.joint_position_target + delta_q)

    def _draw_ibvs_visualization(self, frame, detections_by_id):
        for tag_id in TAG_IDS:
            target = self.target_tag_pixels[tag_id]
            tu, tv = np.round(target).astype(int)
            cv2.circle(frame, (tu, tv), 7, (0, 0, 255), 2)
            cv2.putText(
                frame,
                str(tag_id),
                (tu + 8, tv - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            det = detections_by_id.get(tag_id)
            if det is None:
                continue
            cu, cv = np.round(det.center).astype(int)
            cv2.circle(frame, (cu, cv), 6, (0, 255, 0), -1)
            cv2.putText(
                frame,
                str(tag_id),
                (cu + 8, cv + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW_NAME, *CAMERA_WINDOW_SIZE)
        cv2.moveWindow(CAMERA_WINDOW_NAME, *CAMERA_WINDOW_POS)
        cv2.imshow(CAMERA_WINDOW_NAME, frame)
        cv2.waitKey(1)

    def _check_ibvs_pixel_error(self, frame):
        if frame is None:
            return

        success, _, detections_by_id, missing_ids = self.tag_detector.detect(frame)
        self._draw_ibvs_visualization(frame.copy(), detections_by_id)
        if not success:
            if self.print_counter % 20 == 0:
                print(
                    "[IBVS] waiting for tags: "
                    f"detected={sorted(detections_by_id)} missing={missing_ids}"
                )
            return

        current_features = self._features_from_detections(detections_by_id)
        if current_features is None:
            if self.print_counter % 20 == 0:
                print("[IBVS] tag depth projection is invalid")
            return

        v_cam_cv, error = self.ibvs.compute_camera_velocity(current_features)
        v_ee = self._camera_optical_velocity_to_ee(v_cam_cv)
        q_dot, J_ee = self._compute_joint_velocity(v_ee)
        self._integrate_joint_velocity_command(q_dot)

        error_norm = float(np.linalg.norm(error))
        if self.print_counter % 10 == 0:
            print(
                f"[IBVS] error_norm={error_norm:.3f}px "
                f"q_dot_norm={np.linalg.norm(q_dot):.4f} "
                f"J_rank={np.linalg.matrix_rank(J_ee)}"
            )
        if error_norm < IBVS_ALIGNED_THRESHOLD_PX:
            print(
                "[IBVS] aligned: "
                f"error_norm={error_norm:.3f}px < {IBVS_ALIGNED_THRESHOLD_PX:.1f}px"
            )
            self._set_state(GraspState.INSERT_DESCEND)

    def _servo_to_cartesian_position(self, target_pos):
        q_seed = self.data.qpos[:7].copy()
        success, q_target = self.kinematics.ik(q_seed, self.grasp_rot, target_pos)
        if not success:
            print(f"[Insert] IK failed: target_pos={format_vec(target_pos)}")
            return False
        self.arm_controller.move_to_joint(q_target, PATH_WAYPOINT_TOL)
        return True

    def _peg_contacts_hole(self):
        max_contact_force = 0.0
        contact_geom_name = None
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.peg_geom_id:
                other_geom = contact.geom2
            elif contact.geom2 == self.peg_geom_id:
                other_geom = contact.geom1
            else:
                continue

            other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_name.startswith("hole_"):
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                force_norm = float(np.linalg.norm(contact_force[:3]))
                if force_norm > max_contact_force:
                    max_contact_force = force_norm
                    contact_geom_name = other_name
        return max_contact_force > INSERT_CONTACT_FORCE_THRESHOLD, contact_geom_name, max_contact_force

    def _peg_insert_depth_error(self):
        peg_bottom = self.data.site_xpos[self.peg_bottom_site_id].copy()
        hole_depth = self.data.site_xpos[self.hole_bottom_site_id].copy()
        delta = peg_bottom - hole_depth
        xy_error = float(np.linalg.norm(delta[:2]))
        z_error = float(delta[2])
        inserted = xy_error < INSERT_DEPTH_XY_TOL and z_error <= INSERT_DEPTH_Z_TOL
        return inserted, xy_error, z_error, delta, peg_bottom, hole_depth

    def _read_sensor_vec3(self, sensor_id):
        if sensor_id < 0:
            return np.zeros(3, dtype=np.float64)
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        if dim != 3:
            raise ValueError(f"Expected 3D sensor, got sensor_id={sensor_id}, dim={dim}")
        return self.data.sensordata[adr:adr + dim].copy()

    def _ee_force(self):
        return self._read_sensor_vec3(self.ee_force_sensor_id)

    def _insert_force_delta(self):
        return self._ee_force() - self.insert_force_bias

    def _force_to_world(self, force):
        if self.ee_force_site_id >= 0:
            sensor_rot = self.data.site_xmat[self.ee_force_site_id].reshape(3, 3)
        else:
            sensor_rot = self.data.body(self.ee_id).xmat.reshape(3, 3)
        return sensor_rot @ np.asarray(force, dtype=np.float64)

    def _update_insert_xy_admittance(self, force_delta_world):
        dt = float(self.model.opt.timestep)
        force_xy = INSERT_ADMITTANCE_FORCE_SIGN * np.asarray(force_delta_world[:2], dtype=np.float64)
        acceleration_xy = (
            force_xy
            - INSERT_ADMITTANCE_D * self.insert_admittance_velocity_xy
            - INSERT_ADMITTANCE_K * self.insert_admittance_offset_xy
        ) / INSERT_ADMITTANCE_M

        self.insert_admittance_velocity_xy += acceleration_xy * dt
        self.insert_admittance_velocity_xy = np.clip(
            self.insert_admittance_velocity_xy,
            -INSERT_ADMITTANCE_MAX_VELOCITY,
            INSERT_ADMITTANCE_MAX_VELOCITY,
        )
        self.insert_admittance_offset_xy += self.insert_admittance_velocity_xy * dt
        self.insert_admittance_offset_xy = np.clip(
            self.insert_admittance_offset_xy,
            -INSERT_ADMITTANCE_MAX_OFFSET,
            INSERT_ADMITTANCE_MAX_OFFSET,
        )
        return self.insert_admittance_offset_xy.copy()

    def _insert_descend_step(self):
        if self.insert_start_pos is None or self.insert_target_pos is None:
            self._set_state(GraspState.INSERT_DESCEND)
            return

        contact, contact_geom, contact_force_norm = self._peg_contacts_hole()
        force_delta = self._insert_force_delta()
        force_delta_world = self._force_to_world(force_delta)
        if ENABLE_INSERT_ADMITTANCE and contact:
            self.insert_admittance_active = True
        if self.insert_admittance_active:
            admittance_xy = self._update_insert_xy_admittance(force_delta_world)
        else:
            self.insert_admittance_velocity_xy[:] = 0.0
            admittance_xy = self.insert_admittance_offset_xy.copy()
        force_norm = float(np.linalg.norm(force_delta))
        z_force = abs(float(force_delta_world[2]))
        force = self._ee_force()
        inserted, xy_error, z_error, bottom_delta, peg_bottom, hole_depth = self._peg_insert_depth_error()
        descended = float(self.insert_start_pos[2] - self.data.body(self.ee_id).xpos[2])
        contact_label = contact_geom if contact else "none"
        z_step_scale = np.clip(1.0 - z_force / INSERT_FORCE_THRESHOLD, 0.0, 1.0)
        if ENABLE_INSERT_ADMITTANCE and contact:
            z_step_scale *= INSERT_CONTACT_DESCEND_SCALE
        z_step = INSERT_DESCEND_STEP * z_step_scale
        print(
            f"[Insert] ee_force={format_vec(force)} "
            f"force_delta={format_vec(force_delta)} "
            f"|F-F0|={force_norm:.3f}N "
            f"adm_xy={format_vec(admittance_xy)} "
            f"z_step_speed={z_step:.5f} "
            f"z_err={z_error:.4f} "
            f"xy_err={xy_error:.4f} "
            f"contact={contact_label}"
        )
        if not ENABLE_INSERT_ADMITTANCE and contact:
            self.insert_stop_reason = "检测到碰撞且导纳关闭"
            print(
                "[InsertTest] contact detected, holding current pose: "
                f"force_norm={force_norm:.3f}N z_force={z_force:.3f}N "
                f"contact={contact_label} "
                f"contact_force={contact_force_norm:.3f}N "
                f"bottom_delta={format_vec(bottom_delta)}"
            )
            self._set_state(GraspState.INSERT_HOLD)
            return
        if inserted:
            self.insert_succeeded = True
            self.insert_stop_reason = "轴底部已到达孔底部目标深度"
            print(
                "[Insert] depth reached: "
                f"peg_bottom={format_vec(peg_bottom)} hole_depth={format_vec(hole_depth)}"
            )
            self._set_state(GraspState.RELEASE_GRIPPER)
            return
        if descended >= INSERT_MAX_DESCEND:
            self.insert_stop_reason = "达到最大下探深度但未确认插入到位"
            print(
                "[Insert] max descend reached without confirmed insertion: "
                f"descended={descended:.4f}, xy_err={xy_error:.4f}, z_err={z_error:.4f}"
            )
            self._set_state(GraspState.RETREAT_ABORT)
            return

        self.insert_target_pos[:2] = self.insert_start_pos[:2] + admittance_xy
        self.insert_target_pos[2] -= z_step
        self._servo_to_cartesian_position(self.insert_target_pos)

    def _hold_insert_pose_step(self):
        if self.joint_position_target is None:
            self.joint_position_target = self.data.qpos[:7].copy()
        self.arm_controller.move_to_joint(self.joint_position_target, JOINT_REACHED_TOL)

    def _release_gripper_step(self):
        self.arm_controller.set_gripper(GRIPPER_OPEN)
        self.state_step += 1
        if self.state_step >= RELEASE_OPEN_STEPS:
            self._set_state(GraspState.RETREAT_AFTER_RELEASE)

    def _retreat_after_release_step(self):
        if self.insert_start_q is None:
            self._set_state(GraspState.DONE)
            return
        result = self.arm_controller.move_to_joint(self.insert_start_q, JOINT_REACHED_TOL)
        if result.reached:
            self._set_state(GraspState.DONE)

    def runFunc(self):
        frame = self.get_camera_image(show=self.state != GraspState.CHECK_IBVS_ERROR)

        if self.state == GraspState.PLAN:
            self._plan_targets()
        elif self.state == GraspState.MOVE_ABOVE_PIN:
            if self._follow_joint_path("APPROACH_LINEAR", self.approach_follower):
                self._set_state(GraspState.DESCEND)
        elif self.state == GraspState.DESCEND:
            if self._follow_joint_path("DESCEND_LINEAR", self.descend_follower):
                self._set_state(GraspState.CLOSE_GRIPPER)
        elif self.state == GraspState.CLOSE_GRIPPER:
            self._close_gripper_until_contact()
        elif self.state == GraspState.LIFT:
            if self._follow_joint_path("LIFT_LINEAR", self.lift_follower, include_pin=True):
                if not ENABLE_IBVS:
                    print("[IBVS] skipped for insert test, starting insert descend")
                    self._set_state(GraspState.INSERT_DESCEND)
                else:
                    self._set_state(GraspState.CHECK_IBVS_ERROR)
        elif self.state == GraspState.CHECK_IBVS_ERROR:
            self._check_ibvs_pixel_error(frame)
        elif self.state == GraspState.INSERT_DESCEND:
            self._insert_descend_step()
        elif self.state == GraspState.INSERT_HOLD:
            self._hold_insert_pose_step()
        elif self.state == GraspState.RELEASE_GRIPPER:
            self._release_gripper_step()
        elif self.state == GraspState.RETREAT_AFTER_RELEASE:
            self._retreat_after_release_step()
        elif self.state == GraspState.RETREAT_ABORT:
            self._retreat_after_release_step()
        elif self.state == GraspState.DONE:
            if self.insert_start_q is not None:
                self.arm_controller.move_to_joint(self.insert_start_q, JOINT_REACHED_TOL)

        self.print_counter += 1


def main():
    scene_xml = make_loadable_scene_xml()
    try:
        robot = PegInHoleGraspDemo(str(scene_xml))
        robot.run_loop()
    finally:
        scene_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
