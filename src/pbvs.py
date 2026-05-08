import cv2
import mujoco
import numpy as np

from mujoco_viewer import ArmBaseViewer
from utils import euler2rotmat, rot_to_quat
from xml_paths import PANDA_PBVS_SCENE_XML


POSITION_TOLERANCE = 0.01
ROTATION_TOLERANCE = 0.05
JOINT_SPEED_LIMIT = 0.6
POSITION_GAIN = 1.5
ROTATION_GAIN = 0.1
DLS_DAMPING = 0.08
INITIAL_POSE_KP = 2.0
INITIAL_POSE_REACHED_THRESHOLD = 0.2


def format_vec(vec, precision=4):
    return np.array2string(
        np.asarray(vec),
        precision=precision,
        suppress_small=True,
        separator=", ",
    )


def format_mat(mat, precision=4):
    return np.array2string(
        np.asarray(mat),
        precision=precision,
        suppress_small=True,
        separator=", ",
    )


def orientation_error(desired_quat, current_quat):
    q_err = np.zeros(4, dtype=np.float64)
    q_curr_inv = np.zeros(4, dtype=np.float64)
    rot_err = np.zeros(3, dtype=np.float64)

    mujoco.mju_negQuat(q_curr_inv, current_quat)
    mujoco.mju_mulQuat(q_err, desired_quat, q_curr_inv)
    mujoco.mju_quat2Vel(rot_err, q_err, 1.0)
    return rot_err


class ArmPBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path):
        super().__init__(render_path, arm_path)

        self.window_name = "PBVS Servo"
        self.box_body_name = "box_body"
        self.box_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.box_body_name,
        )
        self.box_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.box_body_name,
        )
        self.box_qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        self.box_qvel_adr = self.model.jnt_dofadr[self.box_joint_id]

        self.grasp_height = 0.38
        self.reachable_min = np.array([0.30, -0.35, 0.03], dtype=np.float64)
        self.reachable_max = np.array([0.65, 0.35, 0.12], dtype=np.float64)
        self.grasp_rot = euler2rotmat(np.pi, 0.0, 0.0)
        self.grasp_quat = np.asarray(rot_to_quat(self.grasp_rot), dtype=np.float64)
        self.rng = np.random.default_rng()
        self.initial_q = np.array([0.0, 0.314, 0.0, -0.754, 0.0, 1.19, 0.0], dtype=np.float64)
        self.reach_initial_pose = False

    def runBefore(self):
        super().runBefore()
        self.model.opt.gravity[:] = 0.0
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.reach_initial_pose = False
        box_pos = self.sample_box_pos()
        self.set_box_pos(box_pos)
        self.data.ctrl[:7] = 0.0
        if self.model.nu > 7:
            self.data.ctrl[7:] = 255
        # cv2.namedWindow(self.window_name)
        self.print_help()
        print(f"[PBVS] random box pos: {box_pos.round(3)}")
        print(
            "[PBVS Init] 正在移动到初始位置，PBVS 尚未开始；"
            f"target_q={format_vec(self.initial_q)}"
        )

    def print_help(self):
        print("\nPBVS 方块抓取位姿控制")
        print("每次启动随机设置方块位置")
        print("先移动到初始关节位置，到位后再开始 PBVS")
        print("目标位姿: 方块正上方 %.2fm，末端固定向下抓取姿态" % self.grasp_height)
        print("")

    def get_box_pos(self):
        return self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 3].copy()

    def sample_box_pos(self):
        return self.rng.uniform(self.reachable_min, self.reachable_max).astype(np.float64)

    def set_box_pos(self, pos):
        pos = np.clip(np.asarray(pos, dtype=np.float64), self.reachable_min, self.reachable_max)
        self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 3] = pos
        self.data.qpos[self.box_qpos_adr + 3:self.box_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.box_qvel_adr:self.box_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_target_pose(self):
        box_pos = self.get_box_pos()
        target_pos = box_pos + np.array([0.0, 0.0, self.grasp_height], dtype=np.float64)
        return target_pos, self.grasp_quat

    def get_ee_pose(self):
        ee_pos = self.data.body(self.ee_id).xpos.copy()
        ee_quat = self.data.body(self.ee_id).xquat.copy()
        return ee_pos, ee_quat

    def compute_pose_error(self):
        target_pos, target_quat = self.get_target_pose()
        ee_pos, ee_quat = self.get_ee_pose()
        pos_err = target_pos - ee_pos
        rot_err = orientation_error(target_quat, ee_quat)
        return pos_err, rot_err

    def jacobian_condition_number(self, J):
        singular_values = np.linalg.svd(J, compute_uv=False)
        if singular_values[-1] <= np.finfo(np.float64).eps:
            return np.inf
        return singular_values[0] / singular_values[-1]

    def end_effector_jacobian(self):
        Jp = np.zeros((3, self.model.nv), dtype=np.float64)
        Jr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, Jp, Jr, self.ee_id)
        return np.vstack([Jp, Jr])[:, :7]

    def compute_joint_velocity(self, pos_err, rot_err):
        if (
            np.linalg.norm(pos_err) < POSITION_TOLERANCE
            and np.linalg.norm(rot_err) < ROTATION_TOLERANCE
        ):
            return np.zeros(7, dtype=np.float64)

        twist = np.concatenate([POSITION_GAIN * pos_err, ROTATION_GAIN * rot_err])
        J = self.end_effector_jacobian()
        damping_matrix = DLS_DAMPING * DLS_DAMPING * np.eye(6)
        q_dot = J.T @ np.linalg.solve(J @ J.T + damping_matrix, twist)
        return np.clip(q_dot, -JOINT_SPEED_LIMIT, JOINT_SPEED_LIMIT)

    def move_to_initial_pose(self):
        q = self.data.qpos[:7].copy()
        q_err = self.initial_q - q
        q_diff = np.linalg.norm(q_err)

        self.data.ctrl[:7] = np.clip(
            INITIAL_POSE_KP * q_err,
            -JOINT_SPEED_LIMIT,
            JOINT_SPEED_LIMIT,
        )
        if self.model.nu > 7:
            self.data.ctrl[7:] = 255

        self.print_counter += 1
        if self.print_counter % 50 == 0:
            print(
                "\n[PBVS Init]\n"
                "  正在移动到初始位置，PBVS 尚未开始\n"
                f"  target_q: {format_vec(self.initial_q)}\n"
                f"  current_q: {format_vec(q)}\n"
                f"  q_err_norm: {q_diff:.4f}"
            )

        if q_diff >= INITIAL_POSE_REACHED_THRESHOLD:
            return False

        self.data.ctrl[:7] = 0.0
        self.reach_initial_pose = True
        print("\n[PBVS Init] 已到达初始位置，开始 PBVS 接近目标")
        return True

    def draw_status(self, frame, pos_err, rot_err):
        box_pos = self.get_box_pos()
        lines = [
            f"box xyz: {box_pos[0]:.2f}, {box_pos[1]:.2f}, {box_pos[2]:.2f}",
            f"pos err: {np.linalg.norm(pos_err):.3f} m",
            f"rot err: {np.linalg.norm(rot_err):.3f} rad",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (16, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    def show_frame(self, pos_err, rot_err):
        frame = self.get_camera_image(show=False)
        self.draw_status(frame, pos_err, rot_err)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def runFunc(self):
        if not self.reach_initial_pose:
            self.move_to_initial_pose()
            return

        pos_err, rot_err = self.compute_pose_error()
        self.data.ctrl[:7] = self.compute_joint_velocity(pos_err, rot_err)
        if self.model.nu > 7:
            self.data.ctrl[7:] = 255

        self.print_counter += 1
        if self.print_counter % 100 == 0:
            target_pos, target_quat = self.get_target_pose()
            ee_pos, ee_quat = self.get_ee_pose()
            J = self.end_effector_jacobian()
            jac_cond = self.jacobian_condition_number(J)
            print(
                "\n[PBVS]\n"
                f"  box_pos: {format_vec(self.get_box_pos())}\n"
                f"  target_pos: {format_vec(target_pos)}\n"
                f"  target_quat: {format_vec(target_quat)}\n"
                f"  ee_pos: {format_vec(ee_pos)}\n"
                f"  ee_quat: {format_vec(ee_quat)}\n"
                f"  pos_err: {format_vec(pos_err)} | norm={np.linalg.norm(pos_err):.4f} m\n"
                f"  rot_err: {format_vec(rot_err)} | norm={np.linalg.norm(rot_err):.4f} rad\n"
                f"  jacobian_cond: {jac_cond:.4f}\n"
                f"  jacobian:\n{format_mat(J)}"
            )

        # self.show_frame(pos_err, rot_err)


if __name__ == "__main__":
    SCENE_XML_PATH = PANDA_PBVS_SCENE_XML

    robot = ArmPBVS(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
