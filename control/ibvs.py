import yaml
import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation as R
from arm_base import ArmBaseViewer
from utils import *
from pupil_apriltags import Detector

# ====================== AprilTag 位姿估计器（封装类，不变） ======================
class AprilTagPoseEstimator:
    def __init__(self, fx, fy, cx, cy, tag_size):
        self.detector = Detector(families="tag36h11", nthreads=4)
        self.camera_params = (fx, fy, cx, cy)
        self.tag_size = tag_size
        # OpenCV -> MuJoCo 坐标校正
        self.R_MJ_FROM_CV = np.eye(4)
        self.R_MJ_FROM_CV[:3, :3] = np.diag([1.0, -1.0, -1.0])

    def detect_and_estimate(self, frame, T_world_cam):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=self.camera_params, tag_size=self.tag_size
        )
        annotated_frame = frame.copy()
        if not detections:
            return False, None, annotated_frame, None

        det = detections[0]
        self._draw_detection(annotated_frame, det)

        # Tag在相机系位姿
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = det.pose_R
        T_cam_tag[:3, 3] = det.pose_t.flatten()
        # Tag在世界系位姿
        T_world_tag = T_world_cam @ self.R_MJ_FROM_CV @ T_cam_tag

        return True, T_world_tag, annotated_frame, det

    def _draw_detection(self, frame, det):
        corners = det.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.putText(frame, f"ID:{det.tag_id}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# ====================== IBVS 视觉伺服核心类 ======================
class IBVSController:
    def __init__(self, fx, fy, cx, cy, gain=0.5):
        self.fx = fx    # 相机内参
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.lambda_ibvs = gain  # IBVS控制增益

    def get_depth_z(self, cam_pos_world, tag_pos_world):
        """
        【核心】用Base系位姿计算 相机坐标系下的深度Z
        :param cam_pos_world: 相机世界坐标 (3,)
        :param tag_pos_world: Tag世界坐标 (3,)
        :return: Z (特征点深度)
        """
        return np.linalg.norm(cam_pos_world - tag_pos_world)

    def build_image_jacobian(self, u, v, Z):
        """
        构建**单个2D图像点**的图像雅可比矩阵 (2x6)
        :param u: 像素x
        :param v: 像素y
        :param Z: 深度
        :return: J: 图像雅可比 2x6
        """
        if Z < 0.01: Z = 0.01  # 防止除零
        u0 = u - self.cx
        v0 = v - self.cy

        J = np.array([
            [-self.fx/Z, 0, u0/Z, (u0*v0)/self.fx, -(self.fx**2 + u0**2)/self.fx, v0],
            [0, -self.fy/Z, v0/Z, (self.fy**2 + v0**2)/self.fy, -(u0*v0)/self.fy, -u0]
        ])
        return J

    def compute_ibvs_control(self, corners, Z, cam_jacobian):
        """
        IBVS主控制律
        :param corners: Tag 4角点像素坐标 (4,2)
        :param Z: 深度
        :param cam_jacobian: 相机6D雅可比矩阵 (6,7) 熊猫臂7自由度
        :return: q_dot: 关节控制速度 (7,)
        """
        # 目标特征点：图像中心（理想位置）
        u_des = self.cx
        v_des = self.cy

        # 拼接所有特征点的图像雅可比 (8x6)
        J_image = []
        error = []
        for (u, v) in corners:
            # 图像误差
            e_u = u - u_des
            e_v = v - v_des
            error.append([e_u, e_v])
            # 图像雅可比
            J_i = self.build_image_jacobian(u, v, Z)
            J_image.append(J_i)

        error = np.array(error).flatten()  # (8,)
        J_image = np.vstack(J_image)       # (8,6)

        # 总雅可比：图像雅可比 × 相机6D雅可比
        J_total = J_image @ cam_jacobian   # (8,7)

        # 伪逆 + 控制律
        J_pinv = np.linalg.pinv(J_total)
        q_dot = -self.lambda_ibvs * J_pinv @ error

        return q_dot, error

# ====================== 主程序：机械臂IBVS伺服控制 ======================
class ArmIBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 机械臂参数
        self.reached = False
        self.arm_dof = 7  # Panda机械臂7自由度

        # 相机ID
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand")
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_center_body")

        # 相机内参
        self.fx, self.fy = 415.7, 415.7
        self.cx, self.cy = 320.0, 240.0
        self.tag_size = 0.1

        # 初始化工具类
        self.tag_estimator = AprilTagPoseEstimator(self.fx, self.fy, self.cx, self.cy, self.tag_size)
        self.ibvs = IBVSController(self.fx, self.fy, self.cx, self.cy, gain=0.1)

        # 窗口
        self.window_name = "IBVS Servo"
        cv2.namedWindow(self.window_name)

    def runBefore(self):
        super().runBefore()
        self.data.qpos[:7] = self.initial_pos[:7]

    def get_camera_jacobian(self):
        q = self.data.qpos[:7].copy()
        return self.kinematics.J(q)

    def runFunc(self):
        # 初始复位
        if not self.reached and not np.allclose(self.data.qpos[:7], self.initial_pos[:7]):
            self.data.ctrl[:7] = self.initial_pos[:7]
            self.reached = True
            return

        # 1. 获取相机图像 + 位姿
        frame = self.get_camera_image(show=False)
        cam_R = self.data.cam_xmat[self.cam_id].reshape(3,3)
        cam_t = self.data.cam_xpos[self.cam_id]
        T_world_cam = np.eye(4)
        T_world_cam[:3,:3] = cam_R
        T_world_cam[:3,3] = cam_t

        # 2. 检测Tag + 获取世界系位姿
        success, T_world_tag, annotated_frame, det = self.tag_estimator.detect_and_estimate(frame, T_world_cam)
        cv2.imshow(self.window_name, annotated_frame)
        cv2.waitKey(1)

        if success:
            # 3. 【核心】计算深度Z（Base系位姿计算）
            tag_t_world = T_world_tag[:3,3]
            Z = self.ibvs.get_depth_z(cam_t, tag_t_world)

            # 4. 获取相机雅可比
            cam_jac = self.get_camera_jacobian()

            # 5. IBVS计算关节速度
            corners = det.corners
            q_dot, img_error = self.ibvs.compute_ibvs_control(corners, Z, cam_jac)

            # 6. 速度限幅（防止超速）
            q_dot = np.clip(q_dot, -0.5, 0.5)

            # 7. 发送控制指令
            self.data.ctrl[:7] = self.data.qpos[:7] + q_dot * 0.01  # 积分步长

            # 打印信息
            if self.print_counter % 50 == 0:
                print("-"*60)
                print(f"深度Z: {Z:.3f} m")
                print(f"图像误差范数: {np.linalg.norm(img_error):.2f}")
                print(f"关节速度: {np.round(q_dot, 3)}")

        self.print_counter += 1

if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_with_apriltag.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
    
    robot = ArmIBVS(SCENE_XML_PATH, SCENE_XML_PATH, YAML_PATH)
    robot.run_loop()