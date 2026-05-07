import yaml
import numpy as np
import mujoco
import cv2
import pinocchio
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from arm_base import ArmBaseViewer
from utils import *
from pupil_apriltags import Detector


def skew(p):
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0],
    ], dtype=np.float64)


def adjoint(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


@dataclass(frozen=True)
class ImageFeatures:
    uv: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        uv = np.asarray(self.uv, dtype=np.float64)
        z = np.asarray(self.z, dtype=np.float64)
        if uv.shape != (4, 2) or z.shape != (4,):
            raise ValueError(f"ImageFeatures expects uv=(4, 2), z=(4,), got {uv.shape}, {z.shape}")
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "z", z)

    @property
    def u(self):
        return self.uv[:, 0]

    @property
    def v(self):
        return self.uv[:, 1]

# ====================== AprilTag 位姿估计器 ======================
class AprilTagPoseEstimator:
    def __init__(self, tag_ids=(0, 1, 2, 3)):
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            refine_edges=1,
        )
        self.tag_ids = tuple(tag_ids)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {
            int(det.tag_id): det
            for det in detections
            if int(det.tag_id) in self.tag_ids
        }

        annotated_frame = frame.copy()
        for det in detection_by_id.values():
            self._draw_detection(annotated_frame, det)

        missing_ids = [tag_id for tag_id in self.tag_ids if tag_id not in detection_by_id]
        return len(missing_ids) == 0, annotated_frame, detection_by_id, missing_ids

    def _draw_detection(self, frame, det):
        corners = det.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.putText(frame, f"ID:{det.tag_id}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


class IBVSController:
    def __init__(self, fx, fy, cx, cy, gain=1.5, stop_threshold_px=8.0):
        self.fx = fx    # 相机内参
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.lambda_ibvs = gain  # IBVS控制增益
        self.stop_threshold_px = stop_threshold_px
        
        # 目标图像特征：4个 tag 中心点 (u, v)
        self.target_u = np.array([219.7, 259.3, 225.9, 185.8])
        self.target_v = np.array([81.9,  117.4, 155.9, 120.8])
        self.target_z = None

    def set_target(self, features):
        self.target_u = features.u.copy()
        self.target_v = features.v.copy()
        self.target_z = features.z.copy()


    def build_image_jacobian(self, u, v, Z):
        """
        构建 单个点 的图像雅可比矩阵 (2x6)
        :param u: 当前像素x
        :param v: 当前像素y
        :param Z: 深度
        :return: J_i: 2x6 图像雅可比
        """
        Z = max(float(Z), 1e-6)
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        J_i = np.array([
            [-1/Z, 0, x/Z, x*y, -(1 + x*x), y],
            [0, -1/Z, y/Z, 1 + y*y, -x*y, -x],
        ], dtype=np.float64)
        return J_i

    def compute_ibvs_control(self, features, cam_jacobian=None):
        """
        IBVS 主控制律（输入 4 个 tag 中心点）
        :param features: 当前 4 个 tag 中心点图像坐标和深度
        :param cam_jacobian: 相机6D速度雅可比 (6, DOF) 例如熊猫臂7自由度：(6,7)
        :return: q_dot: 关节速度指令          (DOF,)
                 error: 图像误差               (8,)
        """
        U = features.u
        V = features.v
        Z = features.z
        Z_control = self.target_z if self.target_z is not None else Z
        J_image = []
        error_pixel = []
        error_norm = []

        for i in range(4):
            u = U[i]
            v = V[i]
            z = Z_control[i]
            e_u = self.target_u[i] - u
            e_v = self.target_v[i] - v
            J_i = self.build_image_jacobian(u, v, z)

            error_pixel.append([e_u, e_v])
            error_norm.append([e_u / self.fx, e_v / self.fy])
            J_image.append(J_i)

        error_pixel = np.array(error_pixel, dtype=np.float64).flatten()
        error_norm = np.array(error_norm, dtype=np.float64).reshape(8, 1)
        J_image = np.vstack(J_image)

        if np.linalg.norm(error_pixel) < self.stop_threshold_px:
            return np.zeros(6, dtype=np.float64), error_pixel

        J_pinv = np.linalg.pinv(J_image)
        v_cam = self.lambda_ibvs * (J_pinv @ error_norm).flatten()
        return v_cam, error_pixel

    def get_target_uv(self):
        return self.target_u, self.target_v

# ====================== 主程序：机械臂IBVS伺服控制 ======================
class ArmIBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 相机ID
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tag_target")
        # init 里
        self.cam_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "cam_site"
        )
        # 相机内参
        self.fx, self.fy, self.cx, self.cy = self.calc_intrinsics()
        self.tag_size = 0.1
        self.tag_ids = (0, 1, 2, 3)
        self.tag_feature_offsets_local = self.get_tag_feature_offsets_local()
        self.current_tag_corners = None

        # 初始化工具类
        self.tag_estimator = AprilTagPoseEstimator(self.tag_ids)
        self.ibvs = IBVSController(self.fx, self.fy, self.cx, self.cy, gain=0.8, stop_threshold_px=8.0)

        self.R_MJ_FROM_CV = np.eye(4)
        self.R_MJ_FROM_CV[:3, :3] = np.diag([1.0, -1.0, -1.0])
        self.R_CV_FROM_MJ = self.R_MJ_FROM_CV[:3, :3]
        
        self.ob_q = [0, 0.314, 0, - 0.754, 0, 1.19, 0]
        self.reach_ob = False
        self.tag_actuator_ids = {
            "x": self.model.actuator("tag_x_pos").id,
            "y": self.model.actuator("tag_y_pos").id,
            "z": self.model.actuator("tag_z_pos").id,
            "roll": self.model.actuator("tag_roll_pos").id,
            "pitch": self.model.actuator("tag_pitch_pos").id,
            "yaw": self.model.actuator("tag_yaw_pos").id,
        }
        self.tag_translation_step = 0.1
        self.tag_rotation_step = 0.1
        # 窗口
        self.window_name = "IBVS Servo"
        cv2.namedWindow(self.window_name)

    def runBefore(self):
        super().runBefore()
        self.data.qvel[:] = 0.0
        if getattr(self.data, "act", None) is not None and self.data.act.size >= 7:
            for i in range(7):
                jid = self.model.actuator_trnid[i][0]
                qadr = self.model.jnt_qposadr[jid]
                self.data.act[i] = self.data.qpos[qadr]
        mujoco.mj_forward(self.model, self.data)


    def get_tag_feature_offsets_local(self):
        offsets = []
        for tag_id in self.tag_ids:
            geom_id = self.model.geom(f"apriltag_{tag_id}").id
            offsets.append(self.model.geom_pos[geom_id].copy())
        return np.asarray(offsets, dtype=np.float64)

    def get_target_feature_points_world(self):
        target_pos = self.data.xpos[self.target_body_id].copy()
        target_rot = self.data.xmat[self.target_body_id].reshape(3, 3)
        return np.array([
            target_pos + target_rot @ offset
            for offset in self.tag_feature_offsets_local
        ])

    def project_target_feature_points(self):
        points_world = self.get_target_feature_points_world()
        cam_R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        cam_t = self.data.cam_xpos[self.cam_id].copy()
        cam_R_world = cam_R.T

        uv = []
        z = []
        for point_world in points_world:
            point_cam = cam_R_world @ (point_world - cam_t)
            depth = float(-point_cam[2])
            if depth <= 1e-6:
                return None
            u = self.cx + self.fx * point_cam[0] / depth
            v = self.cy - self.fy * point_cam[1] / depth
            uv.append([u, v])
            z.append(depth)

        return ImageFeatures(uv=np.asarray(uv, dtype=np.float64), z=np.asarray(z, dtype=np.float64))

    def detection_image_features(self, detection_by_id):
        projected_features = self.project_target_feature_points()
        if projected_features is None:
            return None

        self.current_tag_corners = {
            tag_id: detection_by_id[tag_id].corners.astype(np.float32)
            for tag_id in self.tag_ids
        }
        centers = np.array(
            [detection_by_id[tag_id].center for tag_id in self.tag_ids],
            dtype=np.float64,
        )
        return ImageFeatures(uv=centers, z=projected_features.z)

    def process_tag_keyboard(self, key):
        if key == 255:
            return

        trans_ids = [self.tag_actuator_ids[k] for k in ("x", "y", "z")]
        rot_ids = [self.tag_actuator_ids[k] for k in ("roll", "pitch", "yaw")]
        pos = self.data.ctrl[trans_ids].copy()
        rot = self.data.ctrl[rot_ids].copy()

        if key == 81: pos[0] -= self.tag_translation_step
        elif key == 83: pos[0] += self.tag_translation_step
        elif key == 82: pos[1] += self.tag_translation_step
        elif key == 84: pos[1] -= self.tag_translation_step
        elif key == ord('u'): pos[2] += self.tag_translation_step
        elif key == ord('o'): pos[2] -= self.tag_translation_step
        elif key == ord('q'): rot[0] += self.tag_rotation_step
        elif key == ord('e'): rot[0] -= self.tag_rotation_step
        elif key == ord('w'): rot[1] += self.tag_rotation_step
        elif key == ord('s'): rot[1] -= self.tag_rotation_step
        elif key == ord('a'): rot[2] += self.tag_rotation_step
        elif key == ord('d'): rot[2] -= self.tag_rotation_step
        else:
            return

        pos_range = self.model.actuator_ctrlrange[trans_ids]
        rot_range = self.model.actuator_ctrlrange[rot_ids]
        self.data.ctrl[trans_ids] = np.clip(pos, pos_range[:, 0], pos_range[:, 1])
        self.data.ctrl[rot_ids] = np.clip(rot, rot_range[:, 0], rot_range[:, 1])

    def draw_features(self, frame, features):
        if self.current_tag_corners is not None:
            for tag_id, corners in self.current_tag_corners.items():
                cv2.polylines(frame, [np.int32(np.round(corners))], True, (0, 255, 0), 2)

        if features is not None:
            for i, tag_id in enumerate(self.tag_ids):
                u, v = np.round(features.uv[i]).astype(int)
                cv2.circle(frame, (u, v), 6, (255, 0, 0), -1)
                cv2.putText(frame, str(tag_id), (u - 18, v - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        target_u, target_v = self.ibvs.get_target_uv()
        for i, tag_id in enumerate(self.tag_ids):
            u, v = int(round(target_u[i])), int(round(target_v[i]))
            cv2.circle(frame, (u, v), 7, (0, 0, 255), 2)
            cv2.putText(frame, str(tag_id), (u - 18, v - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def get_body_transform(self, body_id):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body_id]
        return T

    def get_camera_transform(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        T[:3, 3] = self.data.cam_xpos[self.cam_id]
        return T

    def camera_optical_velocity_to_ee(self, v_cam_cv):
        T_world_ee = self.get_body_transform(self.ee_id)
        T_world_cam_mj = self.get_camera_transform()
        T_cam_mj_cam_cv = np.eye(4, dtype=np.float64)
        T_cam_mj_cam_cv[:3, :3] = self.R_MJ_FROM_CV[:3, :3]

        T_ee_cam_cv = np.linalg.inv(T_world_ee) @ T_world_cam_mj @ T_cam_mj_cam_cv
        return (adjoint(T_ee_cam_cv) @ np.asarray(v_cam_cv, dtype=np.float64).reshape(6, 1)).flatten()

    def end_effector_jacobian(self, q):
        Jp = np.zeros((3, self.model.nv), dtype=np.float64)
        Jr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, Jp, Jr, self.ee_id)
        J_world = np.vstack([Jp, Jr])[:, :7]
        R_ee_world = self.get_body_transform(self.ee_id)[:3, :3].T
        X_ee_world = np.zeros((6, 6), dtype=np.float64)
        X_ee_world[:3, :3] = R_ee_world
        X_ee_world[3:, 3:] = R_ee_world
        return X_ee_world @ J_world

    def runFunc(self):

        frame = self.get_camera_image(show=False)
        cam_R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        cam_t = self.data.cam_xpos[self.cam_id]


        if not self.reach_ob:
            q = self.data.qpos[:7].copy()
            Kp = 2
            self.data.ctrl[:7] = np.clip(Kp * (self.ob_q - q), -1.0, 1.0)
            q_diff = np.linalg.norm(q - self.ob_q)
            if q_diff < 0.2:
                success, annotated_frame, detection_by_id, missing_ids = self.tag_estimator.detect(frame)
                if not success:
                    print(f"Reach obstacle pose, but AprilTag detection is incomplete, missing={missing_ids}")
                    return

                target_features = self.detection_image_features(detection_by_id)
                if target_features is None:
                    print("Reach obstacle pose, but target feature depth is invalid")
                    return
                U = target_features.u
                V = target_features.v
                print(f"set U: {U}")
                print(f"set V: {V}")
                self.ibvs.set_target(target_features)
                self.reach_ob = True
                print("Reach obstacle")
            else:
                print(f"diff: {q_diff}")
            return


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

        success, annotated_frame, detection_by_id, missing_ids = self.tag_estimator.detect(frame)
        current_features = None

        if not success:
            if self.print_counter % 10 == 0:
                detected_ids = sorted(detection_by_id.keys())
                print(f"[IBVS] Frame {self.print_counter}: AprilTag detection incomplete, detected={detected_ids}, missing={missing_ids}")

        if success:
            current_features = self.detection_image_features(detection_by_id)
            if current_features is None:
                self.print_counter += 1
                return
            target_u, target_v = self.ibvs.get_target_uv()
            U = current_features.u
            V = current_features.v
            Z = current_features.z
            self.draw_features(annotated_frame, current_features)
 
            v_cam_cv, error = self.ibvs.compute_ibvs_control(current_features, None)
            v_ee = self.camera_optical_velocity_to_ee(v_cam_cv)
            q = self.data.qpos[:7].copy()
            J_ee = self.end_effector_jacobian(q)
            
           
            lam = 0.02
            q_dot = J_ee.T @ np.linalg.inv(J_ee @ J_ee.T + lam * lam * np.eye(6)) @ v_ee
            q_dot = np.clip(q_dot, -1.0, 1.0)

            self.data.ctrl[:7] = q_dot
        

        if success and self.print_counter % 50 == 0:
            print("\n" + "="*60)
            print(f"[IBVS] Frame {self.print_counter}")
            print(f"[IBVS] AprilTag detected: ids={self.tag_ids}")
            print(f"[IBVS] Current centers (U,V):")
            for i in range(4):
                print(f"  Point {i}: U={U[i]:.1f}, V={V[i]:.1f}")
            print(f"[IBVS] Target centers (U,V):")
            for i in range(4):
                print(f"  Point {i}: U={target_u[i]:.1f}, V={target_v[i]:.1f}")

            print(f"[IBVS] Depth Z: {Z}")
            print(f"[IBVS] Image error (8D): {error}")
            print(f"[IBVS] Error norm: {np.linalg.norm(error):.4f}")
            print(f"[IBVS] Camera optical velocity (6D): {v_cam_cv}")
            print(f"[IBVS] EE velocity (6D): {v_ee}")
            print(f"[IBVS] EE Jacobian shape: {J_ee.shape}")
            print(f"[IBVS] EE Jacobian condition number: {np.linalg.cond(J_ee):.2f}")
            print(f"[IBVS] EE Jacobian rank: {np.linalg.matrix_rank(J_ee)}")
            print(f"[IBVS] Joint velocity (7D): {q_dot}")
            print(f"[IBVS] Joint velocity norm: {np.linalg.norm(q_dot):.4f}")
            print(f"[IBVS] Control command applied")
            print("="*60 + "\n")
        self.print_counter += 1


        cv2.imshow(self.window_name, annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        self.process_tag_keyboard(key)
        if current_features is not None and key == ord('t'):
            self.ibvs.set_target(current_features)
            print("Updated target features from current tag centers")


if __name__ == '__main__':
    SCENE_XML_PATH = '/home/kplnb050/study/mujoco_robot_arm/model/franka_emika_panda/scene_with_apriltag.xml'
    YAML_PATH = '/home/kplnb050/study/mujoco_robot_arm/control/target_pos.yaml'
    
    robot = ArmIBVS(SCENE_XML_PATH, SCENE_XML_PATH, YAML_PATH)
    robot.run_loop()
