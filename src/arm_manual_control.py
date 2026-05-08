import contextlib
import io

import cv2
import mujoco
import numpy as np
from mujoco_viewer import ArmBaseViewer
from utils import euler2rotmat
from xml_paths import PANDA_POS_SCENE_XML


KEY_NONE = -1
KEY_LEFT = {81, 65361, 2424832}
KEY_UP = {82, 65362, 2490368}
KEY_RIGHT = {83, 65363, 2555904}
KEY_DOWN = {84, 65364, 2621440}
KEY_PAGE_UP = {65365, 2162688}
KEY_PAGE_DOWN = {65366, 2228224}


class ArmManualControlViewer(ArmBaseViewer):
    def __init__(self, render_path, arm_path):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(render_path, arm_path)

        # 笛卡尔目标位姿
        self.ee_target_pos = None
        self.ee_target_rot = None
        self.window_name = "Manual Control Camera"

    def print_control_help(self):
        print("\n手动控制按键")
        print("请先点击/聚焦 OpenCV 图像窗口，再按以下按键")
        print(f"当前模式: {self.control_mode}")
        print("Tab: 切换 JOINT/CART 模式")
        if self.control_mode == "JOINT":
            print("JOINT: 1-7 增加对应关节角度")
            print("g: 闭合夹爪")
            print("h: 张开夹爪")
        elif self.control_mode == "CART":
            print("CART: 方向键移动末端 X/Y")
            print("PageUp/PageDown: 移动末端 Z")
        print("")

    def switch_mode(self):
        modes = ["JOINT", "CART"]
        idx = modes.index(self.control_mode)
        self.control_mode = modes[(idx + 1) % 2]
        if self.control_mode == "CART":
            self.ee_target_pos = self.data.body(self.ee_id).xpos.copy()
            self.ee_target_rot = euler2rotmat(np.pi, 0, 0)
        self.print_control_help()

    def process_key(self, key):
        if key == KEY_NONE:
            return

        ascii_key = key & 0xFF
        if ascii_key == 9:
            self.switch_mode()
            return

        if self.control_mode == "JOINT":
            delta_joint = 0.03
            if ord("1") <= ascii_key <= ord("7"):
                j_idx = ascii_key - ord("1")
                self.joint_cmd[j_idx] = np.clip(
                    self.joint_cmd[j_idx] + delta_joint,
                    -2.897,
                    2.897,
                )
            elif ascii_key == ord("g"):
                self.gripper_cmd = 0
            elif ascii_key == ord("h"):
                self.gripper_cmd = 255

        elif self.control_mode == "CART":
            delta_cart = 0.01
            if key in KEY_UP:
                self.ee_target_pos[1] -= delta_cart
            elif key in KEY_DOWN:
                self.ee_target_pos[1] += delta_cart
            elif key in KEY_LEFT:
                self.ee_target_pos[0] += delta_cart
            elif key in KEY_RIGHT:
                self.ee_target_pos[0] -= delta_cart
            elif key in KEY_PAGE_UP:
                self.ee_target_pos[2] += delta_cart
            elif key in KEY_PAGE_DOWN:
                self.ee_target_pos[2] -= delta_cart

    def show_camera_and_read_key(self):
        frame = self.get_camera_image(show=False)
        cv2.imshow(self.window_name, frame)
        return cv2.waitKeyEx(1)

    def runBefore(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        cv2.namedWindow(self.window_name)
        self.control_mode = "JOINT"
        self.joint_cmd = self.data.qpos.copy()
        self.ee_target_pos = self.data.body(self.ee_id).xpos.copy()
        self.ee_target_rot = self.data.body(self.ee_id).xmat.reshape(3, 3).copy()
        self.print_control_help()
        

    def runFunc(self):
        key = self.show_camera_and_read_key()
        self.process_key(key)
                
        # ---------------------- 关节空间控制 ----------------------
        if self.control_mode == "JOINT":
            self.data.ctrl[:7] = self.joint_cmd[:7]

        # ---------------------- 笛卡尔空间控制 ----------------------
        elif self.control_mode == "CART":
            q_curr = self.data.qpos[:7].copy()
            with contextlib.redirect_stdout(io.StringIO()):
                success, q_target = self.kinematics.ik(q_curr, self.ee_target_rot, self.ee_target_pos)
            if success and np.all(np.isfinite(q_target)):
                self.data.ctrl[:7] = q_target

        # 夹爪
        if self.model.nv >= 9:
            self.data.ctrl[7:] = self.gripper_cmd

if __name__ == '__main__':
    SCENE_XML_PATH = PANDA_POS_SCENE_XML
    robot = ArmManualControlViewer(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
