import cv2
import mujoco
import numpy as np

from mujoco_viewer import ArmBaseViewer, ROOT_DIR
from utils import euler2rotmat, rot_to_quat


KEY_NONE = -1
KEY_LEFT = {81, 65361, 2424832}
KEY_UP = {82, 65362, 2490368}
KEY_RIGHT = {83, 65363, 2555904}
KEY_DOWN = {84, 65364, 2621440}
KEY_PAGE_UP = {65365, 2162688}
KEY_PAGE_DOWN = {65366, 2228224}

POSITION_TOLERANCE = 0.01
ROTATION_TOLERANCE = 0.05
JOINT_SPEED_LIMIT = 0.6
POSITION_GAIN = 1.5
ROTATION_GAIN = 0.1
DLS_DAMPING = 0.08


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

        self.grasp_height = 0.18
        self.box_move_step = 0.03
        self.workspace_min = np.array([0.2, -0.5, 0.02], dtype=np.float64)
        self.workspace_max = np.array([0.8, 0.5, 0.4], dtype=np.float64)
        self.grasp_rot = euler2rotmat(np.pi, 0.0, 0.0)
        self.grasp_quat = np.asarray(rot_to_quat(self.grasp_rot), dtype=np.float64)
        self.box_target_pos = None

    def runBefore(self):
        super().runBefore()
        self.model.opt.gravity[:] = 0.0
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.box_target_pos = self.get_box_pos()
        self.set_box_pos(self.box_target_pos)
        self.data.ctrl[:7] = 0.0
        if self.model.nu > 7:
            self.data.ctrl[7:] = 255
        cv2.namedWindow(self.window_name)
        self.print_help()

    def print_help(self):
        print("\nPBVS 方块抓取位姿控制")
        print("请先点击/聚焦 OpenCV 图像窗口，再按以下按键")
        print("方向键: 控制方块 X/Y 横移")
        print("PageUp/PageDown 或 u/o: 控制方块 Z")
        print("目标位姿: 方块正上方 %.2fm，末端固定向下抓取姿态" % self.grasp_height)
        print("")

    def get_box_pos(self):
        return self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 3].copy()

    def set_box_pos(self, pos):
        pos = np.clip(np.asarray(pos, dtype=np.float64), self.workspace_min, self.workspace_max)
        self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 3] = pos
        self.data.qpos[self.box_qpos_adr + 3:self.box_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.box_qvel_adr:self.box_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def process_keyboard(self, key):
        if key == KEY_NONE:
            return

        pos = self.box_target_pos.copy()
        ascii_key = key & 0xFF

        if key in KEY_LEFT:
            pos[0] -= self.box_move_step
        elif key in KEY_RIGHT:
            pos[0] += self.box_move_step
        elif key in KEY_UP:
            pos[1] += self.box_move_step
        elif key in KEY_DOWN:
            pos[1] -= self.box_move_step
        elif key in KEY_PAGE_UP or ascii_key == ord("u"):
            pos[2] += self.box_move_step
        elif key in KEY_PAGE_DOWN or ascii_key == ord("o"):
            pos[2] -= self.box_move_step
        else:
            return

        self.box_target_pos = np.clip(pos, self.workspace_min, self.workspace_max)

    def get_target_pose(self):
        box_pos = self.box_target_pos
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

        twist = np.concatenate([-POSITION_GAIN * pos_err, ROTATION_GAIN * rot_err])
        J = self.end_effector_jacobian()
        damping_matrix = DLS_DAMPING * DLS_DAMPING * np.eye(6)
        q_dot = J.T @ np.linalg.solve(J @ J.T + damping_matrix, twist)
        return np.clip(q_dot, -JOINT_SPEED_LIMIT, JOINT_SPEED_LIMIT)

    def draw_status(self, frame, pos_err, rot_err):
        box_pos = self.get_box_pos()
        lines = [
            f"box xyz: {box_pos[0]:.2f}, {box_pos[1]:.2f}, {box_pos[2]:.2f}",
            f"pos err: {np.linalg.norm(pos_err):.3f} m",
            f"rot err: {np.linalg.norm(rot_err):.3f} rad",
            "focus this OpenCV window: arrows xy, PgUp/PgDn or u/o z",
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

    def show_frame_and_process_key(self, pos_err, rot_err):
        frame = self.get_camera_image(show=False)
        self.draw_status(frame, pos_err, rot_err)
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKeyEx(1)
        self.process_keyboard(key)

    def runFunc(self):
        self.set_box_pos(self.box_target_pos)
        pos_err, rot_err = self.compute_pose_error()
        self.data.ctrl[:7] = self.compute_joint_velocity(pos_err, rot_err)
        if self.model.nu > 7:
            self.data.ctrl[7:] = 255

        self.print_counter += 1
        if self.print_counter % 100 == 0:
            print(
                f"[PBVS] pos_err={np.linalg.norm(pos_err):.4f}m "
                f"rot_err={np.linalg.norm(rot_err):.4f}rad "
                f"box={self.get_box_pos()}"
            )

        self.show_frame_and_process_key(pos_err, rot_err)
        self.set_box_pos(self.box_target_pos)


if __name__ == "__main__":
    SCENE_XML_PATH = str(ROOT_DIR / "model/franka_emika_panda/panda_pbvs.xml")

    robot = ArmPBVS(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
