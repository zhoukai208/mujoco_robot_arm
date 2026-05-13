import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

import mujoco
import numpy as np

from mujoco_viewer import ArmBaseViewer
from utils import euler2rotmat, format_vec
from xml_paths import PANDA_POS_SCENE_XML


GRIPPER_OPEN = 255
GRIPPER_CLOSE = 0


class GraspState(Enum):
    RESET = auto()
    PLAN = auto()
    OPEN_GRIPPER = auto()
    MOVE_TO_PRE_GRASP = auto()
    DESCEND_TO_GRASP = auto()
    CLOSE_GRIPPER = auto()
    VERIFY_GRASP = auto()
    LIFT = auto()
    SUCCESS = auto()
    FAIL = auto()


@dataclass
class GraspConfig:
    workspace_x: tuple = (0.38, 0.55)
    workspace_y: tuple = (-0.16, 0.16)
    cube_center_z: float = 0.025

    # PandaKinematics.ik() targets link7. With the fixed downward grasp
    # orientation, ee_center_body sits about 0.212 m below link7.
    link7_to_ee_center_z: float = 0.212
    pre_grasp_ee_z_offset: float = 0.15
    grasp_ee_z_offset: float = 0.0
    lift_ee_z_offset: float = 0.23

    open_steps: int = 120
    close_min_steps: int = 180
    close_timeout_steps: int = 500
    verify_steps: int = 80
    terminal_hold_steps: int = 120
    move_timeout_steps: int = 900
    lift_timeout_steps: int = 1000

    joint_reached_tol: float = 0.04
    path_waypoint_tol: float = 0.025
    descend_path_steps: int = 35
    lift_path_steps: int = 55
    lifted_height_threshold: float = 0.08
    max_ee_box_distance: float = 0.12
    print_interval_steps: int = 80

    grasp_roll: float = np.pi
    grasp_pitch: float = 0.0
    grasp_yaw: float = 0.0


@dataclass
class JointCommandResult:
    q_target: np.ndarray
    q_error: np.ndarray
    dq: np.ndarray
    reached: bool

    @property
    def q_error_norm(self) -> float:
        return float(np.linalg.norm(self.q_error))

    @property
    def dq_norm(self) -> float:
        return float(np.linalg.norm(self.dq))


@dataclass
class GraspTargets:
    pre_grasp_q: np.ndarray
    grasp_q: np.ndarray
    lift_q: np.ndarray
    descend_path: list
    lift_path: list
    pre_grasp_link7_pos: np.ndarray
    grasp_link7_pos: np.ndarray
    lift_link7_pos: np.ndarray


@dataclass
class GraspStats:
    success_count: int = 0
    fail_count: int = 0
    failure_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.success_count + self.fail_count

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success_count / self.total

    def record_success(self):
        self.success_count += 1

    def record_failure(self, reason: str):
        self.fail_count += 1
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1


class PositionArmController:
    """Small wrapper for Panda joint position actuators and the tendon gripper."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.joint_actuator_ids = np.array(
            [model.actuator(f"joint{i}_pos").id for i in range(1, 8)],
            dtype=np.int32,
        )
        self.joint_ctrl_range = model.actuator_ctrlrange[self.joint_actuator_ids].copy()
        self.joint_kp = model.actuator_gainprm[self.joint_actuator_ids, 0].copy()

    def move_to_joint(self, q_target, reached_tol: float) -> JointCommandResult:
        q_target = np.asarray(q_target, dtype=np.float64)
        q_target = np.clip(
            q_target,
            self.joint_ctrl_range[:, 0],
            self.joint_ctrl_range[:, 1],
        )
        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()
        q_error = q_target - q
        self.data.ctrl[self.joint_actuator_ids] = self._gravity_compensated_target(q_target)
        return JointCommandResult(
            q_target=q_target.copy(),
            q_error=q_error,
            dq=dq,
            reached=float(np.linalg.norm(q_error)) < reached_tol,
        )

    def hold_current_joint_position(self):
        self.move_to_joint(self.data.qpos[:7].copy(), reached_tol=0.0)

    def set_gripper(self, command: float):
        if self.model.nu > 7:
            self.data.ctrl[7:] = command

    def _gravity_compensated_target(self, q_target):
        qfrc_bias = self.data.qfrc_bias[:7].copy()
        q_ctrl = q_target + qfrc_bias / self.joint_kp
        return np.clip(
            q_ctrl,
            self.joint_ctrl_range[:, 0],
            self.joint_ctrl_range[:, 1],
        )


class BoxGraspScene:
    """Object lookup, randomization, contact queries, and success checks."""

    def __init__(self, model, data, ee_body_id, config: GraspConfig):
        self.model = model
        self.data = data
        self.ee_body_id = ee_body_id
        self.config = config
        self.box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_body")
        self.box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_body")
        self.box_qpos_adr = model.jnt_qposadr[self.box_joint_id]
        self.left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.initial_box_pos = np.zeros(3, dtype=np.float64)

    def randomize_box(self, rng: np.random.Generator) -> np.ndarray:
        pos = np.array(
            [
                rng.uniform(*self.config.workspace_x),
                rng.uniform(*self.config.workspace_y),
                self.config.cube_center_z,
            ],
            dtype=np.float64,
        )
        self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3] = pos
        self.data.qpos[self.box_qpos_adr + 3 : self.box_qpos_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.initial_box_pos = self.box_pos()
        return self.initial_box_pos.copy()

    def box_pos(self) -> np.ndarray:
        return self.data.xpos[self.box_body_id].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.xpos[self.ee_body_id].copy()

    def ee_box_distance(self) -> float:
        return float(np.linalg.norm(self.ee_pos() - self.box_pos()))

    def box_lift_height(self) -> float:
        return float(self.box_pos()[2] - self.initial_box_pos[2])

    def finger_contacts_with_box(self):
        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 == self.box_body_id:
                other_body = body2
            elif body2 == self.box_body_id:
                other_body = body1
            else:
                continue

            if other_body == self.left_finger_body_id:
                left_contact = True
            elif other_body == self.right_finger_body_id:
                right_contact = True
        return left_contact, right_contact

    def has_gripper_box_contact(self) -> bool:
        left_contact, right_contact = self.finger_contacts_with_box()
        return left_contact and right_contact

    def is_lift_success(self) -> bool:
        return (
            self.box_lift_height() >= self.config.lifted_height_threshold
            and self.ee_box_distance() <= self.config.max_ee_box_distance
            and self.box_pos()[2] > self.initial_box_pos[2] + 0.04
        )


class TopDownGraspPlanner:
    """Builds the three IK targets used by the first closed-loop grasp demo."""

    def __init__(self, kinematics, config: GraspConfig):
        self.kinematics = kinematics
        self.config = config
        self.grasp_rot = euler2rotmat(
            config.grasp_roll,
            config.grasp_pitch,
            config.grasp_yaw,
        )

    def plan(self, q_seed, cube_pos) -> Optional[GraspTargets]:
        q_seed = np.asarray(q_seed, dtype=np.float64).copy()
        pre_pos = self._link7_target(cube_pos, self.config.pre_grasp_ee_z_offset)
        grasp_pos = self._link7_target(cube_pos, self.config.grasp_ee_z_offset)
        lift_pos = self._link7_target(cube_pos, self.config.lift_ee_z_offset)

        success, pre_q = self.kinematics.ik(q_seed, self.grasp_rot, pre_pos)
        if not success:
            return None

        descend_path = self._plan_vertical_path(
            np.asarray(pre_q),
            cube_pos,
            self.config.pre_grasp_ee_z_offset,
            self.config.grasp_ee_z_offset,
            self.config.descend_path_steps,
        )
        if descend_path is None:
            return None
        grasp_q = descend_path[-1]

        lift_path = self._plan_vertical_path(
            grasp_q,
            cube_pos,
            self.config.grasp_ee_z_offset,
            self.config.lift_ee_z_offset,
            self.config.lift_path_steps,
        )
        if lift_path is None:
            return None
        lift_q = lift_path[-1]

        return GraspTargets(
            pre_grasp_q=np.asarray(pre_q, dtype=np.float64),
            grasp_q=np.asarray(grasp_q, dtype=np.float64),
            lift_q=np.asarray(lift_q, dtype=np.float64),
            descend_path=descend_path,
            lift_path=lift_path,
            pre_grasp_link7_pos=pre_pos,
            grasp_link7_pos=grasp_pos,
            lift_link7_pos=lift_pos,
        )

    def _link7_target(self, cube_pos, ee_z_offset):
        return np.asarray(cube_pos, dtype=np.float64) + np.array(
            [0.0, 0.0, ee_z_offset + self.config.link7_to_ee_center_z],
            dtype=np.float64,
        )

    def _plan_vertical_path(self, q_seed, cube_pos, start_ee_z_offset, end_ee_z_offset, steps):
        path = []
        q_current = np.asarray(q_seed, dtype=np.float64).copy()
        for ee_z_offset in np.linspace(start_ee_z_offset, end_ee_z_offset, steps):
            target_pos = self._link7_target(cube_pos, ee_z_offset)
            success, q_next = self.kinematics.ik(q_current, self.grasp_rot, target_pos)
            if not success:
                return None
            q_current = np.asarray(q_next, dtype=np.float64)
            path.append(q_current.copy())
        return path


class ClosedLoopGraspDemo(ArmBaseViewer):
    def __init__(self, render_path, arm_path, max_trials: int = 0, seed: Optional[int] = None):
        super().__init__(render_path, arm_path)
        self.config = GraspConfig()
        self.rng = np.random.default_rng(seed)
        self.max_trials = max_trials

        self.controller = PositionArmController(self.model, self.data)
        self.scene = BoxGraspScene(self.model, self.data, self.ee_id, self.config)
        self.planner = TopDownGraspPlanner(self.kinematics, self.config)
        self.stats = GraspStats()

        self.state = GraspState.RESET
        self.state_steps = 0
        self.episode_steps = 0
        self.last_failure_reason = ""
        self.cube_pos = np.zeros(3, dtype=np.float64)
        self.targets: Optional[GraspTargets] = None
        self.path_index = 0

    def runBefore(self):
        super().runBefore()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._transition(GraspState.RESET)
        print("\n闭环抓取 Demo")
        print("流程: reset -> plan -> open -> approach -> descend -> close -> verify -> lift")
        print("关闭 MuJoCo viewer 或按 Ctrl+C 结束。")

    def runFunc(self):
        if self.max_trials > 0 and self.stats.total >= self.max_trials:
            print("[Done] max trials reached")
            self.handle.close()
            return

        self.episode_steps += 1
        self.state_steps += 1

        if self.state == GraspState.RESET:
            self._step_reset()
        elif self.state == GraspState.PLAN:
            self._step_plan()
        elif self.state == GraspState.OPEN_GRIPPER:
            self._step_open_gripper()
        elif self.state == GraspState.MOVE_TO_PRE_GRASP:
            self._step_move_to_pre_grasp()
        elif self.state == GraspState.DESCEND_TO_GRASP:
            self._step_descend_to_grasp()
        elif self.state == GraspState.CLOSE_GRIPPER:
            self._step_close_gripper()
        elif self.state == GraspState.VERIFY_GRASP:
            self._step_verify_grasp()
        elif self.state == GraspState.LIFT:
            self._step_lift()
        elif self.state == GraspState.SUCCESS:
            self._step_terminal()
        elif self.state == GraspState.FAIL:
            self._step_terminal()

        self._print_periodic_status()

    def _step_reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.controller.set_gripper(GRIPPER_OPEN)
        self.cube_pos = self.scene.randomize_box(self.rng)
        self.targets = None
        self.episode_steps = 0
        print(f"\n[Reset] cube_pos={format_vec(self.cube_pos)}")
        self._transition(GraspState.PLAN)

    def _step_plan(self):
        self.targets = self.planner.plan(self.data.qpos[:7].copy(), self.cube_pos)
        if self.targets is None:
            self._fail("ik_fail")
            return

        print("[Plan] targets ready")
        print(f"  pre_link7={format_vec(self.targets.pre_grasp_link7_pos)}")
        print(f"  grasp_link7={format_vec(self.targets.grasp_link7_pos)}")
        print(f"  lift_link7={format_vec(self.targets.lift_link7_pos)}")
        self._transition(GraspState.OPEN_GRIPPER)

    def _step_open_gripper(self):
        self.controller.hold_current_joint_position()
        self.controller.set_gripper(GRIPPER_OPEN)
        if self.state_steps >= self.config.open_steps:
            self._transition(GraspState.MOVE_TO_PRE_GRASP)

    def _step_move_to_pre_grasp(self):
        result = self.controller.move_to_joint(
            self.targets.pre_grasp_q,
            self.config.joint_reached_tol,
        )
        self.controller.set_gripper(GRIPPER_OPEN)
        if result.reached:
            self._transition(GraspState.DESCEND_TO_GRASP)
        elif self.state_steps >= self.config.move_timeout_steps:
            self._fail("pre_grasp_timeout")

    def _step_descend_to_grasp(self):
        self.controller.set_gripper(GRIPPER_OPEN)
        if self._follow_path(self.targets.descend_path):
            self._transition(GraspState.CLOSE_GRIPPER)
        elif self.state_steps >= self.config.move_timeout_steps:
            self._fail("descend_timeout")

    def _step_close_gripper(self):
        self.controller.move_to_joint(self.targets.grasp_q, self.config.joint_reached_tol)
        self.controller.set_gripper(GRIPPER_CLOSE)
        has_contact = self.scene.has_gripper_box_contact()
        if self.state_steps >= self.config.close_min_steps and has_contact:
            self._transition(GraspState.VERIFY_GRASP)
        elif self.state_steps >= self.config.close_timeout_steps:
            self._fail("no_contact")

    def _step_verify_grasp(self):
        self.controller.move_to_joint(self.targets.grasp_q, self.config.joint_reached_tol)
        self.controller.set_gripper(GRIPPER_CLOSE)
        if self.state_steps < self.config.verify_steps:
            return

        if not self.scene.has_gripper_box_contact():
            self._fail("unstable_grasp")
            return
        if self.scene.ee_box_distance() > self.config.max_ee_box_distance:
            self._fail("box_too_far_after_close")
            return
        self._transition(GraspState.LIFT)

    def _step_lift(self):
        self.controller.set_gripper(GRIPPER_CLOSE)
        if self.scene.is_lift_success():
            self._success()
        elif self._follow_path(self.targets.lift_path):
            self._fail("lift_reached_without_box")
        elif self.state_steps >= self.config.lift_timeout_steps:
            self._fail("lift_timeout")

    def _step_terminal(self):
        self.controller.hold_current_joint_position()
        if self.state == GraspState.SUCCESS:
            self.controller.set_gripper(GRIPPER_CLOSE)
        else:
            self.controller.set_gripper(GRIPPER_OPEN)

        if self.state_steps >= self.config.terminal_hold_steps:
            self._transition(GraspState.RESET)

    def _success(self):
        self.stats.record_success()
        print(
            "[Success] "
            f"lift={self.scene.box_lift_height():.3f}m "
            f"ee_box_dist={self.scene.ee_box_distance():.3f}m "
            f"stats={self._stats_text()}"
        )
        self._transition(GraspState.SUCCESS)

    def _fail(self, reason: str):
        self.last_failure_reason = reason
        self.stats.record_failure(reason)
        print(
            "[Fail] "
            f"reason={reason} "
            f"lift={self.scene.box_lift_height():.3f}m "
            f"ee_box_dist={self.scene.ee_box_distance():.3f}m "
            f"stats={self._stats_text()}"
        )
        self._transition(GraspState.FAIL)

    def _transition(self, next_state: GraspState):
        if self.state != next_state:
            print(f"[State] {self.state.name} -> {next_state.name}")
        self.state = next_state
        self.state_steps = 0
        if next_state in (GraspState.DESCEND_TO_GRASP, GraspState.LIFT):
            self.path_index = 0

    def _follow_path(self, path) -> bool:
        if not path:
            return True
        self.path_index = min(self.path_index, len(path) - 1)
        result = self.controller.move_to_joint(path[self.path_index], self.config.path_waypoint_tol)
        if result.reached and self.path_index < len(path) - 1:
            self.path_index += 1
            return False
        return result.reached and self.path_index >= len(path) - 1

    def _print_periodic_status(self):
        if self.state_steps % self.config.print_interval_steps != 0:
            return
        left_contact, right_contact = self.scene.finger_contacts_with_box()
        print(
            f"[Status] state={self.state.name} "
            f"step={self.state_steps} "
            f"box={format_vec(self.scene.box_pos())} "
            f"lift={self.scene.box_lift_height():.3f}m "
            f"ee_dist={self.scene.ee_box_distance():.3f}m "
            f"contact(L/R)={left_contact}/{right_contact}"
        )

    def _stats_text(self):
        return (
            f"success={self.stats.success_count} "
            f"fail={self.stats.fail_count} "
            f"rate={self.stats.success_rate * 100:.1f}%"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop top-down grasp demo.")
    parser.add_argument(
        "--trials",
        type=int,
        default=0,
        help="Number of grasp attempts before closing the viewer. 0 means run forever.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for cube placement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    robot = ClosedLoopGraspDemo(
        PANDA_POS_SCENE_XML,
        PANDA_POS_SCENE_XML,
        max_trials=args.trials,
        seed=args.seed,
    )
    robot.run_loop()
