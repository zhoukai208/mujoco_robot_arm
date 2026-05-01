import mujoco
import numpy as np
import pynput.keyboard as pkb
from arm_base import ArmBaseViewer
from utils import euler2rotmat

class ArmManualControlViewer(ArmBaseViewer):
    def __init__(self, render_path, arm_path):
        super().__init__(render_path, arm_path)

        # 键盘监听
        self.pressed_keys = set()
        self.key_listener = pkb.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )

        # 笛卡尔目标位姿
        self.ee_target_pos = None
        self.ee_target_rot = None

    def _on_key_press(self, key):
        try:
            # 普通字符键
            if hasattr(key, 'char'):
                self.pressed_keys.add(key.char.lower())
            else:
                # 特殊键
                if key == pkb.Key.tab:
                    self.pressed_keys.add("tab")
                elif key == pkb.Key.up:
                    self.pressed_keys.add("up")
                elif key == pkb.Key.down:
                    self.pressed_keys.add("down")
                elif key == pkb.Key.left:
                    self.pressed_keys.add("left")
                elif key == pkb.Key.right:
                    self.pressed_keys.add("right")
                elif key == pkb.Key.page_up:
                    self.pressed_keys.add("page_up")
                elif key == pkb.Key.page_down:
                    self.pressed_keys.add("page_down")
        except:
            pass

    def _on_key_release(self, key):
        try:
            if hasattr(key, 'char'):
                self.pressed_keys.discard(key.char.lower())
            else:
                if key == pkb.Key.tab:
                    self.pressed_keys.discard("tab")
                elif key == pkb.Key.up:
                    self.pressed_keys.discard("up")
                elif key == pkb.Key.down:
                    self.pressed_keys.discard("down")
                elif key == pkb.Key.left:
                    self.pressed_keys.discard("left")
                elif key == pkb.Key.right:
                    self.pressed_keys.discard("right")
                elif key == pkb.Key.page_up:
                    self.pressed_keys.discard("page_up")
                elif key == pkb.Key.page_down:
                    self.pressed_keys.discard("page_down")
        except:
            pass

    def runBefore(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.key_listener.start()
        self.control_mode = "JOINT"
        self.joint_cmd = self.data.qpos.copy()
        self.ee_target_pos = self.data.body(self.ee_id).xpos.copy()
        self.ee_target_rot = self.data.body(self.ee_id).xmat.reshape(3, 3).copy()
        

    def runFunc(self):
        # ---------------------- 模式切换：Tab ----------------------
        if "tab" in self.pressed_keys:
            modes = ["JOINT", "CART"]
            idx = modes.index(self.control_mode)
            self.control_mode = modes[(idx + 1) % 2]
            self.pressed_keys.remove("tab")  # 防止连续触发
            print(f"\n🎮 模式切换: {self.control_mode}")
            if self.control_mode == "CART":
                self.ee_target_pos = self.data.body(self.ee_id).xpos.copy()
                self.ee_target_rot = euler2rotmat(np.pi, 0, 0)
                
        # ---------------------- 关节空间控制 ----------------------
        if self.control_mode == "JOINT":
            delta_joint = 0.03  # 增大步长
            for c in self.pressed_keys:
                if c in '1234567':
                    j_idx = int(c) - 1
                    self.joint_cmd[j_idx] = np.clip(self.joint_cmd[j_idx] + delta_joint, -2.897, 2.897)
                elif c == 'g':
                    self.gripper_cmd = 0
                elif c == 'h':
                    self.gripper_cmd = 255

            self.data.ctrl[:7] = self.joint_cmd[:7]

        # ---------------------- 笛卡尔空间控制 ----------------------
        elif self.control_mode == "CART":
            delta_cart = 0.01  # 增大步长

            if "up" in self.pressed_keys:
                self.ee_target_pos[1] -= delta_cart
            if "down" in self.pressed_keys:
                self.ee_target_pos[1] += delta_cart
            if "left" in self.pressed_keys:
                self.ee_target_pos[0] += delta_cart
            if "right" in self.pressed_keys:
                self.ee_target_pos[0] -= delta_cart

            if "page_up" in self.pressed_keys:
                self.ee_target_pos[2] += delta_cart
            if "page_down" in self.pressed_keys:
                self.ee_target_pos[2] -= delta_cart

            q_curr = self.data.qpos[:7].copy()
            success, q_target = self.kinematics.ik(q_curr, self.ee_target_rot, self.ee_target_pos)
            if success and np.all(np.isfinite(q_target)):
                self.data.ctrl[:7] = q_target

        # 夹爪
        if self.model.nv >= 9:
            self.data.ctrl[7:] = self.gripper_cmd

        # 相机图像
        self.get_camera_image(show=True)

if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_pos.xml'
    robot = ArmManualControlViewer(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()