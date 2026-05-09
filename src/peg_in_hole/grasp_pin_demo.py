import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import mujoco
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from mujoco_viewer import ArmBaseViewer
from utils import euler2rotmat

from view_scene import make_loadable_scene_xml

JOINT_REACHED_TOL = 0.008
GRIPPER_OPEN = 255
GRIPPER_CLOSE = 0
GRIPPER_CLOSE_MIN_STEPS = 180
GRIPPER_CLOSE_TIMEOUT_STEPS = 500
APPROACH_STEPS_DOWN = 35
DESCEND_STEPS = 45
INSERT_STEPS = 35
RELEASE_STEPS = 120
PATH_WAYPOINT_TOL = 0.006

APPROACH_CLEARANCE = 0.14
SAFE_APPROACH_CLEARANCE = 0.28


class GraspState(Enum):
    PLAN = auto()
    MOVE_ABOVE_PIN = auto()
    DESCEND = auto()
    CLOSE_GRIPPER = auto()
    LIFT = auto()
    INSERT = auto()
    RELEASE = auto()
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


def format_vec(vec, precision=4):
    return np.array2string(
        np.asarray(vec),
        precision=precision,
        suppress_small=True,
        separator=", ",
    )


class PegInHoleGraspDemo(ArmBaseViewer):
    def __init__(self, render_path):
        super().__init__(render_path, render_path)
        self.state = GraspState.PLAN
        self.state_step = 0
        self.pose_targets = {}
        self.approach_q_path = []
        self.descend_q_path = []
        self.lift_q_path = []
        self.insert_q_path = []
        self.grasp_rot = euler2rotmat(np.pi, 0.0, 0.0)
        self.arm_controller = PositionArmController(
            self.model,
            self.data,
            gripper_command=self._gripper_command_for_state,
        )
        self.approach_follower = PositionJointPathFollower(self.arm_controller, PATH_WAYPOINT_TOL)
        self.descend_follower = PositionJointPathFollower(self.arm_controller, PATH_WAYPOINT_TOL)
        self.lift_follower = PositionJointPathFollower(self.arm_controller, PATH_WAYPOINT_TOL)
        self.insert_follower = PositionJointPathFollower(self.arm_controller, PATH_WAYPOINT_TOL)
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg_body")
        self.peg_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_collision")
        self.peg_grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_grasp_site")
        self.peg_bottom_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_bottom_site")
        self.hole_approach_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "hole_approach_site",
        )
        self.hole_insert_depth_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "hole_insert_depth_site",
        )
        self.grasp_weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "peg_grasp_weld")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.grasp_failed_reported = False
        if self.kinematics.EE_FRAME_ID != self.kinematics.FRAME_ID:
            self.kinematics.FRAME_ID = self.kinematics.EE_FRAME_ID
            print("[GraspDemo] IK target frame set to ee_center_body")

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
        print("[GraspDemo] start with position control: plan -> grasp -> move above hole -> insert -> release")

    def _make_pose_targets(self):
        mujoco.mj_forward(self.model, self.data)
        grasp = self.data.site_xpos[self.peg_grasp_site_id].copy()
        peg_bottom = self.data.site_xpos[self.peg_bottom_site_id].copy()
        hole_approach = self.data.site_xpos[self.hole_approach_site_id].copy()
        hole_insert_depth = self.data.site_xpos[self.hole_insert_depth_site_id].copy()
        above = grasp + np.array([0.0, 0.0, APPROACH_CLEARANCE], dtype=np.float64)
        insert = hole_insert_depth - (peg_bottom - grasp)
        return {
            GraspState.MOVE_ABOVE_PIN: above,
            GraspState.DESCEND: grasp,
            GraspState.LIFT: hole_approach,
            GraspState.INSERT: insert,
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
        )
        self.insert_q_path = self._plan_cartesian_path(
            "INSERT",
            [pose_targets[GraspState.LIFT], pose_targets[GraspState.INSERT]],
            self.lift_q_path[-1],
            [INSERT_STEPS],
        )
        self._set_state(GraspState.MOVE_ABOVE_PIN)

    def _plan_cartesian_path(self, label, points, q_start, steps_per_segment):
        path = []
        q_seed = np.asarray(q_start, dtype=np.float64)
        for segment_idx, steps in enumerate(steps_per_segment):
            start_pos = np.asarray(points[segment_idx], dtype=np.float64)
            target_pos = np.asarray(points[segment_idx + 1], dtype=np.float64)
            for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
                pos = (1.0 - t) * start_pos + t * target_pos
                success, q_target = self.kinematics.ik(q_seed, self.grasp_rot, pos)
                if not success:
                    raise RuntimeError(f"IK failed for {label} waypoint, target_pos={pos}")
                q_seed = np.asarray(q_target, dtype=np.float64)
                path.append(q_seed.copy())
        print(
            f"[GraspDemo] planned {label} cartesian path: "
            f"{len(path)} waypoints from={format_vec(points[0])} to={format_vec(points[-1])}"
        )
        return path

    def _set_state(self, state):
        self.state = state
        self.state_step = 0
        if state == GraspState.MOVE_ABOVE_PIN:
            self.approach_follower.start(self.approach_q_path)
            print(
                "[GraspDemo] cartesian approach start: "
                f"to={format_vec(self.pose_targets[GraspState.MOVE_ABOVE_PIN])}"
            )
        if state == GraspState.DESCEND:
            self.descend_follower.start(self.descend_q_path)
            print(
                "[GraspDemo] linear descend start: "
                f"from={format_vec(self.pose_targets[GraspState.MOVE_ABOVE_PIN])} "
                f"to={format_vec(self.pose_targets[GraspState.DESCEND])}"
            )
        if state == GraspState.CLOSE_GRIPPER:
            self.grasp_failed_reported = False
        if state == GraspState.LIFT:
            self.lift_follower.start(self.lift_q_path)
            print(
                "[GraspDemo] move above hole start: "
                f"from={format_vec(self.pose_targets[GraspState.DESCEND])} "
                f"to={format_vec(self.pose_targets[GraspState.LIFT])}"
            )
        if state == GraspState.INSERT:
            self.insert_follower.start(self.insert_q_path)
            print(
                "[GraspDemo] insert start: "
                f"from={format_vec(self.pose_targets[GraspState.LIFT])} "
                f"to={format_vec(self.pose_targets[GraspState.INSERT])}"
            )
        if state == GraspState.RELEASE:
            print("[GraspDemo] release gripper and disable grasp weld")
        if state == GraspState.DONE:
            print("[GraspDemo] insertion complete: holding final pose with gripper open")
        print(f"[GraspDemo] state -> {state.name}")

    def _servo_to_joint_target(self, q_target, reached_tol):
        result = self.arm_controller.move_to_joint(q_target, reached_tol)
        if self.print_counter % 50 == 0:
            print(
                f"[GraspDemo] {self.state.name} "
                f"q_err_norm={result.q_err_norm:.4f} "
                f"dq_norm={result.dq_norm:.4f} "
                f"q_target={format_vec(result.q_target)} "
                f"q_ctrl={format_vec(result.q_ctrl)}"
            )
        return result.reached

    def _follow_joint_path(self, label, follower, include_pin=False):
        done, result = follower.step()
        ee_pos, _ = self._get_ee_pose()
        if self.print_counter % 50 == 0:
            line = (
                f"[GraspDemo] {label} "
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
        if self.state in (GraspState.CLOSE_GRIPPER, GraspState.LIFT, GraspState.INSERT):
            return GRIPPER_CLOSE
        return GRIPPER_OPEN

    def _close_gripper_until_contact(self):
        self._servo_to_joint_target(self.descend_q_path[-1], JOINT_REACHED_TOL)
        self.state_step += 1
        left_contact, right_contact = self._finger_contacts_with_pin()
        if self.print_counter % 50 == 0 or self.state_step == 1:
            print(
                "[GraspDemo] closing gripper with physical contact "
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
            print("[GraspDemo] grasp failed: pin was not contacted by both fingers, holding closed")
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
            "[GraspDemo] grasp weld activated after two-finger contact "
            f"rel_pos={format_vec(rel_pos)} rel_quat={format_vec(rel_quat)}"
        )

    def _release_grasp(self):
        self._servo_to_joint_target(self.insert_q_path[-1], JOINT_REACHED_TOL)
        self.arm_controller.set_gripper(GRIPPER_OPEN)
        if self.grasp_weld_id >= 0:
            self.data.eq_active[self.grasp_weld_id] = 0

        self.state_step += 1
        if self.print_counter % 50 == 0 or self.state_step == 1:
            print(
                "[GraspDemo] releasing peg "
                f"step={self.state_step} "
                f"peg_pos={format_vec(self.data.xpos[self.peg_body_id])}"
            )

        if self.state_step >= RELEASE_STEPS:
            self._set_state(GraspState.DONE)

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

    def runFunc(self):
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
                self._set_state(GraspState.INSERT)
        elif self.state == GraspState.INSERT:
            if self._follow_joint_path("INSERT_LINEAR", self.insert_follower, include_pin=True):
                self._set_state(GraspState.RELEASE)
        elif self.state == GraspState.RELEASE:
            self._release_grasp()
        elif self.state == GraspState.DONE:
            self._servo_to_joint_target(self.insert_q_path[-1], JOINT_REACHED_TOL)

        self.print_counter += 1
        
        frame = self.get_camera_image(show=True)


def main():
    scene_xml = make_loadable_scene_xml()
    try:
        robot = PegInHoleGraspDemo(str(scene_xml))
        robot.run_loop()
    finally:
        scene_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
