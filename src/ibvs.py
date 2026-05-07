from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
from pupil_apriltags import Detector

from arm_base import ArmBaseViewer, ROOT_DIR


TAG_IDS = (0, 1, 2, 3)

KEY_NONE = 255
KEY_LEFT = 81
KEY_RIGHT = 83
KEY_UP = 82
KEY_DOWN = 84

JOINT_SPEED_LIMIT = 1.0
OBSERVATION_KP = 2.0
OBSERVATION_REACHED_THRESHOLD = 0.2
DLS_DAMPING = 0.02


def skew(p):
    return np.array(
        [
            [0.0, -p[2], p[1]],
            [p[2], 0.0, -p[0]],
            [-p[1], p[0], 0.0],
        ],
        dtype=np.float64,
    )


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


@dataclass(frozen=True)
class ServoState:
    features: ImageFeatures
    error: np.ndarray
    v_cam_cv: np.ndarray
    v_ee: np.ndarray
    J_ee: np.ndarray
    q_dot: np.ndarray


class AprilTagPoseEstimator:
    def __init__(self, tag_ids=TAG_IDS):
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

    @staticmethod
    def _draw_detection(frame, det):
        corners = det.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.putText(
            frame,
            f"ID:{det.tag_id}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )


class IBVSController:
    def __init__(self, fx, fy, cx, cy, gain=1.5, stop_threshold_px=8.0):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.lambda_ibvs = float(gain)
        self.stop_threshold_px = float(stop_threshold_px)

        self.target_u = np.zeros(4, dtype=np.float64)
        self.target_v = np.zeros(4, dtype=np.float64)
        self.target_z = None

    def set_target(self, features):
        self.target_u = features.u.copy()
        self.target_v = features.v.copy()
        self.target_z = features.z.copy()

    def target_uv(self):
        return self.target_u, self.target_v

    def build_point_interaction_matrix(self, u, v, z):
        z = max(float(z), 1e-6)
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        return np.array(
            [
                [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y],
                [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x],
            ],
            dtype=np.float64,
        )

    def compute_camera_velocity(self, features):
        z_control = self.target_z if self.target_z is not None else features.z
        image_jacobian = []
        error_pixel = []
        error_normalized = []

        for i in range(4):
            e_u = self.target_u[i] - features.u[i]
            e_v = self.target_v[i] - features.v[i]

            error_pixel.append([e_u, e_v])
            error_normalized.append([e_u / self.fx, e_v / self.fy])
            image_jacobian.append(
                self.build_point_interaction_matrix(features.u[i], features.v[i], z_control[i])
            )

        error_pixel = np.asarray(error_pixel, dtype=np.float64).flatten()
        error_normalized = np.asarray(error_normalized, dtype=np.float64).reshape(8, 1)
        image_jacobian = np.vstack(image_jacobian)

        if np.linalg.norm(error_pixel) < self.stop_threshold_px:
            return np.zeros(6, dtype=np.float64), error_pixel

        v_cam = self.lambda_ibvs * (np.linalg.pinv(image_jacobian) @ error_normalized).flatten()
        return v_cam, error_pixel


class ArmIBVS(ArmBaseViewer):
    def __init__(self, render_path, arm_path):
        super().__init__(render_path, arm_path)

        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tag_target")

        self.fx, self.fy, self.cx, self.cy = self.calc_intrinsics()
        self.tag_ids = TAG_IDS
        self.tag_feature_offsets_local = self.get_tag_feature_offsets_local()

        self.tag_estimator = AprilTagPoseEstimator(self.tag_ids)
        self.ibvs = IBVSController(self.fx, self.fy, self.cx, self.cy, gain=0.8, stop_threshold_px=8.0)

        self.T_MJ_CAMERA_FROM_CV_CAMERA = np.eye(4, dtype=np.float64)
        self.T_MJ_CAMERA_FROM_CV_CAMERA[:3, :3] = np.diag([1.0, -1.0, -1.0])

        self.ob_q = np.array([0.0, 0.314, 0.0, -0.754, 0.0, 1.19, 0.0], dtype=np.float64)
        self.reach_ob = False
        self.tag_translation_step = 0.1
        self.tag_rotation_step = 0.1
        self.tag_actuator_ids = self.get_tag_actuator_ids()

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

    def get_tag_actuator_ids(self):
        return {
            "x": self.model.actuator("tag_x_pos").id,
            "y": self.model.actuator("tag_y_pos").id,
            "z": self.model.actuator("tag_z_pos").id,
            "roll": self.model.actuator("tag_roll_pos").id,
            "pitch": self.model.actuator("tag_pitch_pos").id,
            "yaw": self.model.actuator("tag_yaw_pos").id,
        }

    def get_tag_feature_offsets_local(self):
        offsets = []
        for tag_id in self.tag_ids:
            geom_id = self.model.geom(f"apriltag_{tag_id}").id
            offsets.append(self.model.geom_pos[geom_id].copy())
        return np.asarray(offsets, dtype=np.float64)

    def get_target_feature_points_world(self):
        target_pos = self.data.xpos[self.target_body_id].copy()
        target_rot = self.data.xmat[self.target_body_id].reshape(3, 3)
        return np.array([target_pos + target_rot @ offset for offset in self.tag_feature_offsets_local])

    def project_target_feature_points(self):
        points_world = self.get_target_feature_points_world()
        cam_R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        cam_t = self.data.cam_xpos[self.cam_id].copy()
        world_R_cam = cam_R.T

        uv = []
        z = []
        for point_world in points_world:
            point_cam = world_R_cam @ (point_world - cam_t)
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

        centers = np.array(
            [detection_by_id[tag_id].center for tag_id in self.tag_ids],
            dtype=np.float64,
        )
        return ImageFeatures(uv=centers, z=projected_features.z)

    def process_tag_keyboard(self, key):
        if key == KEY_NONE:
            return

        trans_ids = [self.tag_actuator_ids[k] for k in ("x", "y", "z")]
        rot_ids = [self.tag_actuator_ids[k] for k in ("roll", "pitch", "yaw")]
        pos = self.data.ctrl[trans_ids].copy()
        rot = self.data.ctrl[rot_ids].copy()

        if key == KEY_LEFT:
            pos[0] -= self.tag_translation_step
        elif key == KEY_RIGHT:
            pos[0] += self.tag_translation_step
        elif key == KEY_UP:
            pos[1] += self.tag_translation_step
        elif key == KEY_DOWN:
            pos[1] -= self.tag_translation_step
        elif key == ord("u"):
            pos[2] += self.tag_translation_step
        elif key == ord("o"):
            pos[2] -= self.tag_translation_step
        elif key == ord("q"):
            rot[0] += self.tag_rotation_step
        elif key == ord("e"):
            rot[0] -= self.tag_rotation_step
        elif key == ord("w"):
            rot[1] += self.tag_rotation_step
        elif key == ord("s"):
            rot[1] -= self.tag_rotation_step
        elif key == ord("a"):
            rot[2] += self.tag_rotation_step
        elif key == ord("d"):
            rot[2] -= self.tag_rotation_step
        else:
            return

        pos_range = self.model.actuator_ctrlrange[trans_ids]
        rot_range = self.model.actuator_ctrlrange[rot_ids]
        self.data.ctrl[trans_ids] = np.clip(pos, pos_range[:, 0], pos_range[:, 1])
        self.data.ctrl[rot_ids] = np.clip(rot, rot_range[:, 0], rot_range[:, 1])

    def draw_features(self, frame, features):
        if features is not None:
            for i, tag_id in enumerate(self.tag_ids):
                u, v = np.round(features.uv[i]).astype(int)
                cv2.circle(frame, (u, v), 6, (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    str(tag_id),
                    (u - 18, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        target_u, target_v = self.ibvs.target_uv()
        for i, tag_id in enumerate(self.tag_ids):
            u, v = int(round(target_u[i])), int(round(target_v[i]))
            cv2.circle(frame, (u, v), 7, (0, 0, 255), 2)
            cv2.putText(
                frame,
                str(tag_id),
                (u - 18, v - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

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
        T_ee_cam_cv = (
            np.linalg.inv(T_world_ee)
            @ T_world_cam_mj
            @ self.T_MJ_CAMERA_FROM_CV_CAMERA
        )
        return (adjoint(T_ee_cam_cv) @ np.asarray(v_cam_cv, dtype=np.float64).reshape(6, 1)).flatten()

    def end_effector_jacobian(self):
        Jp = np.zeros((3, self.model.nv), dtype=np.float64)
        Jr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, Jp, Jr, self.ee_id)

        J_world = np.vstack([Jp, Jr])[:, :7]
        ee_R_world = self.get_body_transform(self.ee_id)[:3, :3].T
        X_ee_world = np.zeros((6, 6), dtype=np.float64)
        X_ee_world[:3, :3] = ee_R_world
        X_ee_world[3:, 3:] = ee_R_world
        return X_ee_world @ J_world

    def compute_joint_velocity(self, v_ee):
        J_ee = self.end_effector_jacobian()
        damping_matrix = DLS_DAMPING * DLS_DAMPING * np.eye(6)
        q_dot = J_ee.T @ np.linalg.solve(J_ee @ J_ee.T + damping_matrix, v_ee)
        q_dot = np.clip(q_dot, -JOINT_SPEED_LIMIT, JOINT_SPEED_LIMIT)
        return q_dot, J_ee

    def move_to_observation_pose(self, frame):
        q = self.data.qpos[:7].copy()
        self.data.ctrl[:7] = np.clip(
            OBSERVATION_KP * (self.ob_q - q),
            -JOINT_SPEED_LIMIT,
            JOINT_SPEED_LIMIT,
        )

        q_diff = np.linalg.norm(q - self.ob_q)
        if q_diff >= OBSERVATION_REACHED_THRESHOLD:
            print(f"diff: {q_diff}")
            return

        success, _, detection_by_id, missing_ids = self.tag_estimator.detect(frame)
        if not success:
            print(f"Reach obstacle pose, but AprilTag detection is incomplete, missing={missing_ids}")
            return

        target_features = self.detection_image_features(detection_by_id)
        if target_features is None:
            print("Reach obstacle pose, but target feature depth is invalid")
            return

        self.ibvs.set_target(target_features)
        self.reach_ob = True
        print(f"set U: {target_features.u}")
        print(f"set V: {target_features.v}")
        print("Reach obstacle")

    def run_visual_servo_step(self, detection_by_id):
        current_features = self.detection_image_features(detection_by_id)
        if current_features is None:
            return None

        v_cam_cv, error = self.ibvs.compute_camera_velocity(current_features)
        v_ee = self.camera_optical_velocity_to_ee(v_cam_cv)
        q_dot, J_ee = self.compute_joint_velocity(v_ee)
        self.data.ctrl[:7] = q_dot

        return ServoState(
            features=current_features,
            error=error,
            v_cam_cv=v_cam_cv,
            v_ee=v_ee,
            J_ee=J_ee,
            q_dot=q_dot,
        )

    def print_init_once(self):
        if self.print_counter != 0:
            return

        print("\n" + "#" * 60)
        print("# IBVS Controller Initialized")
        print("#" * 60)
        print(f"[Init] Camera ID: {self.cam_id}")
        print(f"[Init] Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        print(f"[Init] IBVS gain: {self.ibvs.lambda_ibvs}")
        print(f"[Init] Target UV: U={self.ibvs.target_u}, V={self.ibvs.target_v}")
        print("#" * 60 + "\n")

    def print_camera_debug(self):
        if self.print_counter % 100 != 0:
            return

        cam_R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        cam_t = self.data.cam_xpos[self.cam_id]
        print(f"\n[Camera] Position: {cam_t}")
        print(f"[Camera] Rotation matrix:\n{cam_R}")

    def print_detection_miss(self, detection_by_id, missing_ids):
        if self.print_counter % 10 != 0:
            return

        detected_ids = sorted(detection_by_id.keys())
        print(
            f"[IBVS] Frame {self.print_counter}: AprilTag detection incomplete, "
            f"detected={detected_ids}, missing={missing_ids}"
        )

    def print_servo_state(self, servo_state):
        if self.print_counter % 50 != 0:
            return

        target_u, target_v = self.ibvs.target_uv()
        features = servo_state.features

        print("\n" + "=" * 60)
        print(f"[IBVS] Frame {self.print_counter}")
        print(f"[IBVS] AprilTag detected: ids={self.tag_ids}")
        print("[IBVS] Current centers (U,V):")
        for i in range(4):
            print(f"  Point {i}: U={features.u[i]:.1f}, V={features.v[i]:.1f}")
        print("[IBVS] Target centers (U,V):")
        for i in range(4):
            print(f"  Point {i}: U={target_u[i]:.1f}, V={target_v[i]:.1f}")

        print(f"[IBVS] Depth Z: {features.z}")
        print(f"[IBVS] Image error (8D): {servo_state.error}")
        print(f"[IBVS] Error norm: {np.linalg.norm(servo_state.error):.4f}")
        print(f"[IBVS] Camera optical velocity (6D): {servo_state.v_cam_cv}")
        print(f"[IBVS] EE velocity (6D): {servo_state.v_ee}")
        print(f"[IBVS] EE Jacobian shape: {servo_state.J_ee.shape}")
        print(f"[IBVS] EE Jacobian condition number: {np.linalg.cond(servo_state.J_ee):.2f}")
        print(f"[IBVS] EE Jacobian rank: {np.linalg.matrix_rank(servo_state.J_ee)}")
        print(f"[IBVS] Joint velocity (7D): {servo_state.q_dot}")
        print(f"[IBVS] Joint velocity norm: {np.linalg.norm(servo_state.q_dot):.4f}")
        print("[IBVS] Control command applied")
        if np.linalg.norm(servo_state.error) < self.ibvs.stop_threshold_px:
            print(f"[IBVS] Error is within threshold ({self.ibvs.stop_threshold_px:.1f}px); target satisfied")
        print("=" * 60 + "\n")

    def show_frame_and_process_key(self, frame, current_features):
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        self.process_tag_keyboard(key)

        if current_features is not None and key == ord("t"):
            self.ibvs.set_target(current_features)
            print("Updated target features from current tag centers")

    def runFunc(self):
        frame = self.get_camera_image(show=False)

        if not self.reach_ob:
            self.move_to_observation_pose(frame)
            return

        self.print_init_once()
        self.print_camera_debug()

        success, annotated_frame, detection_by_id, missing_ids = self.tag_estimator.detect(frame)
        servo_state = None

        if success:
            servo_state = self.run_visual_servo_step(detection_by_id)
            current_features = servo_state.features if servo_state is not None else None
            self.draw_features(annotated_frame, current_features)
            if servo_state is not None:
                self.print_servo_state(servo_state)
        else:
            current_features = None
            self.draw_features(annotated_frame, current_features)
            self.print_detection_miss(detection_by_id, missing_ids)

        self.print_counter += 1
        self.show_frame_and_process_key(annotated_frame, current_features)


if __name__ == "__main__":
    SCENE_XML_PATH = str(ROOT_DIR / "model/franka_emika_panda/scene_with_apriltag.xml")

    robot = ArmIBVS(SCENE_XML_PATH, SCENE_XML_PATH)
    robot.run_loop()
