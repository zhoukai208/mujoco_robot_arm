import yaml
import numpy as np
import mujoco
import cv2
import pinocchio
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


class IBVSController:
    def __init__(self, fx, fy, cx, cy, gain=1.5):
        self.fx = fx    # 相机内参
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.lambda_ibvs = gain  # IBVS控制增益
        
        # 目标图像特征：4个点 (u, v)
        self.target_u = np.array([219.7, 259.3, 225.9, 185.8])
        self.target_v = np.array([81.9,  117.4, 155.9, 120.8])

    def set_uv(self, u, v):
        self.target_u = u
        self.target_v = v

    def build_image_jacobian(self, u, v, Z):
        """
        构建 单个点 的图像雅可比矩阵 (2x6)
        :param u: 当前像素x
        :param v: 当前像素y
        :param Z: 深度
        :return: J_i: 2x6 图像雅可比
        """
        if Z < 0.01: Z = 0.01  # 防止深度除零
        u0 = u - self.cx
        v0 = v - self.cy

        J_i = np.array([
            [-self.fx/Z,    0,         u0/Z,        (u0*v0)/self.fx,   -(self.fx**2 + u0**2)/self.fx,  v0],
            [0,             -self.fy/Z, v0/Z,        (self.fy**2 + v0**2)/self.fy, -(u0*v0)/self.fy,  -u0]
        ])
        return J_i

    def compute_ibvs_control(self, U, V, Z, cam_jacobian=None):
        """
        IBVS 主控制律（输入 4 个点）
        :param U: 当前 4 个角点的 u 坐标     (4,)
        :param V: 当前 4 个角点的 v 坐标     (4,)
        :param Z: 当前 4 个角点的深度        (4,)
        :param cam_jacobian: 相机6D速度雅可比 (6, DOF) 例如熊猫臂7自由度：(6,7)
        :return: q_dot: 关节速度指令          (DOF,)
                 error: 图像误差               (8,)
        """
        J_image = []
        error = []

        # ======================
        # 遍历 4 个点，逐个构建图像雅可比 + 计算误差
        # ======================
        for i in range(4):
            u = U[i]
            v = V[i]
            z = Z[i]

            # 1. 计算单个点的图像误差 (e_u, e_v)
            e_u = u - self.target_u[i]
            e_v = v - self.target_v[i]

            # 2. 构建单个点的图像雅可比 2x6
            J_i = self.build_image_jacobian(u, v, z)

            # 3. 存入列表
            error.append([e_u, e_v])
            J_image.append(J_i)

        # ======================
        # 拼接成全局 8x6 图像雅可比 & 8维误差
        # ======================
        error = np.array(error).flatten()  # (8,)
        J_image = np.vstack(J_image)       # (8,6)

        # print(f"J_image: {list(J_image)}")

        J_pinv = np.linalg.pinv(J_image)
        v_cam = -self.lambda_ibvs * (J_pinv @ error)
        # print(f"v_cam: {list(v_cam)}")
        return v_cam, error

    def get_target_uv(self):
        return self.target_u, self.target_v

# ====================== 主程序：机械臂IBVS伺服控制 ======================
class ArmIBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 相机ID
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand")
        self.tag_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apriltag_0")
        # init 里
        self.cam_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "cam_site"
        )
        # 相机内参
        self.fx, self.fy = 415.7, 415.7
        self.cx, self.cy = 320.0, 240.0
        self.tag_size = 0.1

        # 初始化工具类
        self.tag_estimator = AprilTagPoseEstimator(self.fx, self.fy, self.cx, self.cy, self.tag_size)
        self.ibvs = IBVSController(self.fx, self.fy, self.cx, self.cy, gain=0.1)

        self.R_MJ_FROM_CV = np.eye(4)
        self.R_MJ_FROM_CV[:3, :3] = np.diag([1.0, -1.0, -1.0])
        
        self.ob_q = [0, 0.314, 0, - 0.754, 0, 1.19, 0]
        self.reach_ob = False
        # 窗口
        self.window_name = "IBVS Servo"
        cv2.namedWindow(self.window_name)

    def runBefore(self):
        super().runBefore()


    def get_tag_corners(self, T_world_cam):
        T_world_tag = np.eye(4)
        T_world_tag[:3,3] = self.data.xpos[self.tag_id].copy()
        T_world_tag[:3,:3] =  self.data.xmat[self.tag_id].reshape(3,3).copy()
        
        half = self.tag_size / 2
        corners_tag_local = [
            [ half,  half, 0, 1],   # 右上角
            [ half, -half, 0, 1],   # 右下角
            [-half, -half, 0, 1],   # 左下角
            [-half,  half, 0, 1],   # 左上角
        ]

        corners_world = []
        for p_local in corners_tag_local:
            p_world = T_world_tag @ p_local   # 齐次乘法，一步到位
            corners_world.append(p_world[:3]) # 取前3位：[x,y,z]

        T_cam_world = np.linalg.inv(T_world_cam)

        corners_cam = []
        for p_world in corners_world:
            p_world_homo = [p_world[0], p_world[1], p_world[2], 1]
            p_cam = T_cam_world @ p_world_homo
            p_cam = self.R_MJ_FROM_CV @ p_cam
            corners_cam.append(p_cam[:3])

        return np.array(corners_cam)

    def sort_corners(self, corners):
        c = corners.copy()
        center = np.mean(c, axis=0)
        angles = np.arctan2(c[:,1]-center[1], c[:,0]-center[0])
        idx = np.argsort(angles)
        return c[idx]

    def runFunc(self):

        frame = self.get_camera_image(show=False)
        cam_R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        cam_t = self.data.cam_xpos[self.cam_id]
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = cam_R
        T_world_cam[:3, 3] = cam_t


        if not self.reach_ob:
            q = self.data.qpos[:7].copy()
            Kp = 2
            self.data.ctrl[:7] = Kp * (self.ob_q - q)
            q_diff = np.linalg.norm(q - self.ob_q)
            if q_diff < 0.2:
                self.reach_ob = True
                success, T_world_tag, annotated_frame, det = self.tag_estimator.detect_and_estimate(frame, T_world_cam)
                corners = self.sort_corners(det.corners)  
                U = corners[:, 0]
                V = corners[:, 1]
                print(f"set U: {U}")
                print(f"set V: {V}")
                self.ibvs.set_uv(U, V)  
                print("Reach obstacle")
            else:
                print(f"diff: {q_diff}")
            return


        key = cv2.waitKey(1) & 0xFF

        # 每次按键盘动 0.02 米（可以自己改大小）
        step = 0.06  

        # 获取当前 tag 的 3 个轴位置
        tx = self.data.joint("tag_x").qpos[0]
        ty = self.data.joint("tag_y").qpos[0]
        tz = self.data.joint("tag_z").qpos[0]

        # 方向控制
        if key == ord('j'): tx -= step  # 左
        if key == ord('l'): tx += step  # 右
        if key == ord('i'): ty += step  # 前
        if key == ord('k'): ty -= step  # 后
        if key == ord('u'): tz += step  # 上
        if key == ord('o'): tz -= step  # 下

        # 把位置发给执行器（位置控制）
        self.data.ctrl[self.model.actuator("tag_x_act").id] = tx
        self.data.ctrl[self.model.actuator("tag_y_act").id] = ty
        self.data.ctrl[self.model.actuator("tag_z_act").id] = tz

        if self.print_counter == 0:
            print("\n" + "#"*60)
            print("# IBVS Controller Initialized")
            print("#"*60)
            print(f"[Init] Camera ID: {self.cam_id}")
            print(f"[Init] Camera Site ID: {self.cam_site_id}")
            print(f"[Init] Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
            print(f"[Init] IBVS gain: {self.ibvs.lambda_ibvs}")
            print(f"[Init] Target UV: U={self.ibvs.target_u}, V={self.ibvs.target_v}")
            print("#"*60 + "\n")
        

        if self.print_counter % 100 == 0:
            print(f"\n[Camera] Position: {cam_t}")
            print(f"[Camera] Rotation matrix:\n{cam_R}")

        success, T_world_tag, annotated_frame, det = self.tag_estimator.detect_and_estimate(frame, T_world_cam)
        
        if not success:
            if self.print_counter % 10 == 0:
                print(f"[IBVS] Frame {self.print_counter}: AprilTag NOT detected")

        if det is not None:
            target_u, target_v = self.ibvs.get_target_uv()
            for u, v in zip(target_u, target_v):
                cv2.circle(annotated_frame, (int(u), int(v)), 5, (0, 255, 0), -1)

            # Sort corners by angle
            corners = self.sort_corners(det.corners)
            
            U = corners[:, 0]
            V = corners[:, 1]

            Z = self.get_tag_corners(T_world_cam)[:, 2]
 
            v_cam_mj, error = self.ibvs.compute_ibvs_control(U, V, Z, None)
            

            def Ad(T):
                """
                SE(3) adjoint transform
                T: 4x4
                """
                R = T[:3, :3]
                p = T[:3, 3]

                p_hat = np.array([
                    [0, -p[2], p[1]],
                    [p[2], 0, -p[0]],
                    [-p[1], p[0], 0]
                ])

                Ad_T = np.zeros((6, 6))
                Ad_T[:3, :3] = R
                Ad_T[:3, 3:] = p_hat @ R
                Ad_T[3:, 3:] = R
                return Ad_T
            v_world = Ad(T_world_cam) @ v_cam_mj
            
            
            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.cam_site_id)
            J = np.vstack([Jp, Jr])[:, :7]
            
           
            lam = 0.05
            q_dot = J.T @ np.linalg.inv(J @ J.T + lam * lam * np.eye(6)) @ v_world

            self.data.ctrl[:7] = q_dot
        

        if self.print_counter % 50 == 0:
            print("\n" + "="*60)
            print(f"[IBVS] Frame {self.print_counter}")
            print(f"[IBVS] AprilTag detected: ID={det.tag_id}")
            print(f"[IBVS] Current corners (U,V):")
            for i in range(4):
                print(f"  Point {i}: U={U[i]:.1f}, V={V[i]:.1f}")
            print(f"[IBVS] Target corners (U,V):")
            for i in range(4):
                print(f"  Point {i}: U={target_u[i]:.1f}, V={target_v[i]:.1f}")

            print(f"[IBVS] Depth Z: {Z}")
            print(f"[IBVS] Tag pose in camera: t={det.pose_t.flatten()}")
            print(f"[IBVS] Image error (8D): {error}")
            print(f"[IBVS] Error norm: {np.linalg.norm(error):.4f}")
            print(f"[IBVS] Camera velocity (6D): {v_cam_mj}")
            print(f"[IBVS] World velocity (6D): {v_world}")
            print(f"[IBVS] Jacobian shape: {J.shape}")
            print(f"[IBVS] Jacobian condition number: {np.linalg.cond(J):.2f}")
            print(f"[IBVS] Jacobian rank: {np.linalg.matrix_rank(J)}")
            print(f"[IBVS] Joint velocity (7D): {q_dot}")
            print(f"[IBVS] Joint velocity norm: {np.linalg.norm(q_dot):.4f}")
            print(f"[IBVS] Control command applied")
            print("="*60 + "\n")
        self.print_counter += 1


        cv2.imshow(self.window_name, annotated_frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_with_apriltag.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
    
    robot = ArmIBVS(SCENE_XML_PATH, SCENE_XML_PATH, YAML_PATH)
    robot.run_loop()