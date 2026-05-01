import yaml
import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation as R
from arm_base import ArmBaseViewer
from utils import *

from pupil_apriltags import Detector
detector = Detector(families="tag36h11", nthreads=4)
fx = 415.7
fy = 415.7
cx = 320.0
cy = 240.0
camera_params = (fx, fy, cx, cy)
tag_size = 0.1

def detect_apriltag(frame):
    """
    输入：RGB图像
    返回：检测结果 + 画好的图
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size
    )

    # 画图
    for det in detections:
        # 画角点
        corners = det.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

        # 画ID
        cx = int(det.center[0])
        cy = int(det.center[1])
        cv2.putText(frame, f"ID:{det.tag_id}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return frame, detections


R_MJ_FROM_CV = np.eye(4)
R_MJ_FROM_CV[:3, :3] = np.diag([1.0, -1.0, -1.0])


class ArmIBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 多点轨迹（原有）
        self.target_poses_list = []
        self.trajectory_list = []
        self.current_target_idx = 0
        self.traj_idx = 0

        # ===================== AprilTag 控制参数 =====================
        self.STEP_TRANS = 0.005
        self.STEP_ROT = 0.015
        self.tag_body_id = None
        self.reached = False

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_center_body")
        self.tag_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apriltag_0")


        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand")

        self.window_name = "IBVS"
        cv2.namedWindow(self.window_name)
    
    def runBefore(self):
        super().runBefore()
        self.data.qpos[:7] = self.initial_pos[:7]

    def runFunc(self):
        if not self.reached and not np.allclose(self.data.qpos[:7], self.initial_pos[:7]):
            self.data.ctrl[:7] = self.initial_pos[:7]
            self.reached = True


        frame = self.get_camera_image(show=False)
        frame, detections = detect_apriltag(frame)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

        if detections:
            # -------------------------
            # camera pose in world
            # -------------------------
            R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
            t = self.data.cam_xpos[self.cam_id]

            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R
            T_world_cam[:3, 3]  = t

            
            # -------------------------
            # AprilTag: tag -> cam
            # -------------------------
            T_cam_tag = np.eye(4)
            T_cam_tag[:3, :3] = detections[0].pose_R
            T_cam_tag[:3, 3]  = detections[0].pose_t.flatten()

            # -------------------------
            # world estimate
            # -------------------------
            T_world_tag = T_world_cam @ R_MJ_FROM_CV @ T_cam_tag

            est_tag = T_world_tag[:3, 3]

            # -------------------------
            # TRUE GT (IMPORTANT FIX)
            # mocap body must be used, NOT xpos
            # -------------------------
            mocap_id, _ = self.mocaps._mocap_map["apriltag_0"]
            gt_tag = self.data.mocap_pos[mocap_id]

            print(R_MJ_FROM_CV)
            print(self.data.cam_xpos[self.cam_id])
            print("GT :", gt_tag)
            print("EST:", est_tag)
            print("ERR:", est_tag - gt_tag)


        self.print_counter += 1
        if self.print_counter % 50 == 0:
            pass


if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_with_apriltag.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
    
    # 修复构造函数参数
    robot = ArmIBVS(SCENE_XML_PATH, SCENE_XML_PATH, YAML_PATH)
    robot.run_loop()