# Franka Emika Panda XML Overview

This document summarizes the XML files under `model/franka_emika_panda`, how they differ, and which project scripts reference them directly.

## Direct Script Usage

| XML | Referenced by | Purpose |
|---|---|---|
| `scene_pos.xml` | `arm_manual_control.py`, `arm_move.py`, `arm_grasp.py`, `bc_grasp.py`, `load_grasp_data.py`, `gen_grasp_data.py` | Position-control Panda with `box_body` freejoint target object. |
| `scene_tau.xml` | `joint_impedance_control.py`, `joint_admittance_control.py`, `ee_impedance_control.py` | Torque-control Panda scene. |
| `scene_with_apriltag.xml` | `ibvs.py` | IBVS scene with Panda and controllable AprilTag target. |
| `panda_pbvs.xml` | `pbvs.py` | PBVS scene with velocity-control Panda and controllable box target. |

## Core Robot Models

| XML | Control type | nq/nv/nu | Notes | Current usage |
|---|---:|---:|---|---|
| `panda.xml` | general / affine servo | 9/9/8 | Standard Panda with gripper. | Included by several generic scenes. |
| `panda_pos.xml` | position | 9/9/8 | Seven joint position actuators plus gripper actuator. | Included by `scene_pos.xml` and position-control obstacle scenes. |
| `panda_tau.xml` | motor / torque | 9/9/8 | Seven joint torque motors plus gripper actuator. | Included by `scene_tau.xml`. |
| `panda_vel.xml` | velocity | 9/9/8 | Seven joint velocity actuators plus gripper actuator. | Included by `scene_vel.xml` and `panda_pbvs.xml`. |
| `panda_tag.xml` | intvelocity | 9/9/8 | Panda variant for IBVS velocity-style control. | Included by `scene_with_apriltag.xml`. |
| `mjx_panda.xml` | position | 9/9/8 | MJX Panda variant. | Included by `mjx_scene.xml`. |
| `mjx_panda_nohand.xml` | position | 7/7/7 | MJX Panda without gripper. | Not referenced by `src`. |
| `panda_nohand.xml` | general / affine servo | 7/7/7 | Panda without gripper. | Not referenced by `src`. |
| `panda_remove_finger.xml` | position | 7/7/7 | Panda with fingers removed. | Not referenced by `src`. |
| `hand.xml` | gripper only | 2/2/1 | Standalone Franka hand model. | Not referenced by `src`. |
| `tag.xml` | position | 6/6/6 | Controllable 6-DOF AprilTag target. | Included by `scene_with_apriltag.xml`. |

## Scene Files

| XML | Includes | nq/nv/nu | Key difference | Script usage |
|---|---|---:|---|---|
| `scene_pos.xml` | `panda_pos.xml` | 16/15/8 | Position-control Panda plus `box_body` freejoint and 16D `home` keyframe. | Directly used. |
| `scene_tau.xml` | `panda_tau.xml` | 9/9/8 | Torque-control Panda with world frame. | Directly used. |
| `scene_with_apriltag.xml` | `panda_tag.xml`, `tag.xml` | 15/15/14 | Panda plus 6-DOF controllable AprilTag target. | Directly used. |
| `panda_pbvs.xml` | `panda_vel.xml` | 16/15/8 | Velocity-control Panda plus `box_body` freejoint and 16D `home` keyframe. | Directly used. |
| `scene_vel.xml` | `panda_vel.xml` | 9/9/8 | Plain velocity-control Panda scene. | Not referenced by `src`. |
| `scene.xml` | `panda.xml` | 9/9/8 | Basic Panda scene. | Not referenced by `src`. |
| `scene_with_cube.xml` | `panda.xml` | 16/15/8 | Panda plus `cube` freejoint. | Not referenced by `src`. |
| `scene_withcamera.xml` | `panda.xml` | 16/15/8 | Panda plus cube/camera-related scene content. | Not referenced by `src`. |
| `scene_withmocap.xml` | `panda.xml` | 9/9/8 | Panda plus mocap/world-frame helpers. | Not referenced by `src`. |
| `scene_withtarget.xml` | `panda.xml` | 9/9/8 | Panda plus target/world-frame helpers. | Not referenced by `src`. |
| `scene_with_obstacles.xml` | `panda.xml` | 9/9/8 | Generic obstacle scene. | Not referenced by `src`. |
| `scene_pos_with_obstacles.xml` | `panda_pos.xml` | 9/9/8 | Position-control Panda plus generated obstacle geometry. | Not referenced by `src`. |
| `scene_pos_withfixedobstacles.xml` | `panda_pos.xml` | 9/9/8 | Position-control Panda plus fixed obstacle geometry. | Not referenced by `src`. |
| `scene_camera_calibration.xml` | `panda.xml` | 9/9/8 | Camera calibration scene with AprilTag board. | Not referenced by `src`. |
| `scene_with_checkerboard.xml` | `panda.xml` | 9/9/8 | Checkerboard calibration scene. | Not referenced by `src`. |
| `scene_calibration_simple.xml` | `panda.xml` | 9/9/8 | Simplified calibration scene. | Not referenced by `src`. |
| `mjx_scene.xml` | `mjx_panda.xml` | 9/9/8 | Basic MJX scene. | Not referenced by `src`. |
| `mjx_single_cube.xml` | `mjx_scene.xml` | 16/15/8 | MJX scene with box/mocap target and three keyframes. | Not referenced by `src`. |

## Summary

- `panda_pos.xml`, `panda_tau.xml`, and `panda_vel.xml` share the same robot structure but expose different actuator types: position, torque, and velocity.
- Scenes with a freejoint object have `nq=16` and `nv=15`; plain Panda-with-gripper scenes have `nq=9` and `nv=9`.
- `scene_with_apriltag.xml` has `nu=14` because it combines Panda actuators with six target actuators from `tag.xml`.
- Data collection, behavior cloning, manual control, and trajectory playback use `scene_pos.xml`.
- Impedance and admittance controllers use `scene_tau.xml`.
- Visual servoing is split between `scene_with_apriltag.xml` for IBVS and `panda_pbvs.xml` for PBVS.
