#!/usr/bin/env python3
"""Compare MuJoCo end-effector frames with the RTB kinematic model.

This script is intentionally GUI-free.  It gives a repeatable snapshot of the
fixed transform errors that matter before tuning IBVS gains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rbt_module import RobotController  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "universal_robots_ur5e" / "scene.xml"
SITE_NAME = "attachment_site"
CAMERA_NAME = "end_effector_camera"


def transform_from_pos_mat(pos: np.ndarray, mat: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = mat.reshape(3, 3)
    transform[:3, 3] = pos
    return transform


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    return float(np.rad2deg(Rotation.from_matrix(delta).magnitude()))


def format_vector(vector: np.ndarray) -> str:
    return np.array2string(np.asarray(vector), precision=6, suppress_small=True)


def format_transform(transform: np.ndarray) -> str:
    return np.array2string(transform, precision=6, suppress_small=True)


def parse_qpos(raw_values: list[str] | None) -> np.ndarray | None:
    if raw_values is None:
        return None
    values = [float(value) for value in raw_values]
    if len(values) != 6:
        raise ValueError("--qpos expects exactly 6 joint values in radians")
    return np.asarray(values, dtype=np.float64)


def load_model(qpos: np.ndarray | None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if qpos is None:
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            data.qpos[:6] = 0.0
    else:
        data.qpos[:6] = qpos

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def print_pose_block(name: str, transform: np.ndarray) -> None:
    euler_xyz = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    print(f"\n{name}")
    print(f"  position: {format_vector(transform[:3, 3])}")
    print(f"  euler xyz deg: {format_vector(euler_xyz)}")
    print(f"  transform:\n{format_transform(transform)}")


def print_pair_error(label: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    position_error = np.linalg.norm(reference[:3, 3] - candidate[:3, 3])
    angle_error = rotation_error_deg(reference, candidate)
    print(f"\n{label}")
    print(f"  world origin distance: {position_error:.6f} m")
    print(f"  world rotation delta:  {angle_error:.6f} deg")


def calc_mujoco_camera_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    camera_body_id = int(model.cam_bodyid[camera_id])
    camera_pos = data.cam_xpos[camera_id].copy()
    mujoco.mj_jac(model, data, jacp, jacr, camera_pos, camera_body_id)
    return np.vstack((jacp[:, :6], jacr[:, :6]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MuJoCo attachment/camera frames against RTB fkine()."
    )
    parser.add_argument(
        "--qpos",
        nargs=6,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Optional joint values in radians. Defaults to XML keyframe 0.",
    )
    args = parser.parse_args()

    qpos = parse_qpos(args.qpos)
    model, data = load_model(qpos)
    robot_controller = RobotController()

    site_id = model.site(SITE_NAME).id
    camera_id = model.camera(CAMERA_NAME).id

    q = data.qpos[:6].copy()
    rtb_tool = np.asarray(robot_controller.get_forward_kinematics(q).A, dtype=np.float64)
    mj_site = transform_from_pos_mat(data.site_xpos[site_id], data.site_xmat[site_id])
    mj_camera = transform_from_pos_mat(data.cam_xpos[camera_id], data.cam_xmat[camera_id])

    print(f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"qpos rad: {format_vector(q)}")

    print_pose_block("MuJoCo attachment_site frame", mj_site)
    print_pose_block("MuJoCo end_effector_camera frame", mj_camera)
    print_pose_block("RTB fkine(q) tool frame", rtb_tool)

    print_pair_error("RTB tool vs MuJoCo attachment_site", rtb_tool, mj_site)
    print_pair_error("RTB tool vs MuJoCo end_effector_camera", rtb_tool, mj_camera)
    print_pair_error("MuJoCo attachment_site vs end_effector_camera", mj_site, mj_camera)

    t_rtb_tool_mj_site = np.linalg.inv(rtb_tool) @ mj_site
    t_rtb_tool_mj_camera = np.linalg.inv(rtb_tool) @ mj_camera
    t_mj_site_mj_camera = np.linalg.inv(mj_site) @ mj_camera

    print("\nRelative transform: inv(T_world_rtb_tool) @ T_world_mujoco_site")
    print(format_transform(t_rtb_tool_mj_site))
    print("\nRelative transform: inv(T_world_rtb_tool) @ T_world_mujoco_camera")
    print(format_transform(t_rtb_tool_mj_camera))
    print("\nRelative transform: inv(T_world_mujoco_site) @ T_world_mujoco_camera")
    print(format_transform(t_mj_site_mj_camera))

    rtb_jacobe = np.asarray(robot_controller.get_jacobian(q), dtype=np.float64)
    mj_camera_jac_world = calc_mujoco_camera_jacobian(model, data, camera_id)
    print("\nJacobian diagnostics")
    print(f"  RTB jacobe shape: {rtb_jacobe.shape}, cond: {np.linalg.cond(rtb_jacobe):.6e}")
    print(
        "  MuJoCo camera-point world Jacobian shape: "
        f"{mj_camera_jac_world.shape}, cond: {np.linalg.cond(mj_camera_jac_world):.6e}"
    )
    print("  RTB jacobe:\n" + format_transform(rtb_jacobe))
    print("  MuJoCo camera-point world Jacobian:\n" + format_transform(mj_camera_jac_world))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
