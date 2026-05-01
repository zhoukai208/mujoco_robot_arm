import yaml
import numpy as np
from arm_base import ArmBaseViewer
from utils import *


class ArmMoveController(ArmBaseViewer):
    def __init__(self, render_path, arm_path, yaml_path):
        super().__init__(render_path, arm_path)
        self.yaml_path = yaml_path

        # 多点轨迹
        self.target_poses_list = []
        self.trajectory_list = []
        self.current_target_idx = 0
        self.traj_idx = 0

    def load_target_poses_from_yaml(self):
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        targets = data.get("target_poses", [])
        print(f"✅ 加载 {len(targets)} 个目标点")
        self.target_poses_list = targets

    def plan_all_trajectories(self):
        q_start = self.initial_pos[:7]
        self.trajectory_list = []

        for pose in self.target_poses_list:
            t_pos = np.array(pose["pos"])
            r = np.deg2rad(pose["roll"])
            p = np.deg2rad(pose["pitch"])
            y = np.deg2rad(pose["yaw"])
            t_rot = euler2rotmat(r, p, y)

            success, q_target = self.kinematics.ik(q_start, t_rot, t_pos)
            traj = self.movel_planner.plan(q_start, t_pos, t_rot)
            self.trajectory_list.append(traj)
            q_start = q_target

    def runBefore(self):
        super().runBefore()
        self.load_target_poses_from_yaml()
        self.plan_all_trajectories()
        self.control_mode = "TRAJ"
        self.current_target_idx = 0
        self.traj_idx = 0

    def runFunc(self):
        # 只执行自动多点轨迹
        if self.control_mode == "TRAJ":
            if self.current_target_idx >= len(self.trajectory_list):
                print("\n🎉 全部目标点执行完毕")
                self.control_mode = "IDLE"
                return

            traj = self.trajectory_list[self.current_target_idx]
            if self.traj_idx < len(traj):
                self.data.ctrl[:7] = traj[self.traj_idx]
                self.traj_idx += 1
            else:
                print(f"✅ 到达点 {self.current_target_idx+1}")
                self.current_target_idx += 1
                self.traj_idx = 0

        # 相机图像
        self.get_camera_image(show=True)

        # 打印
        self.print_counter += 1
        if self.print_counter % 50 == 0:
            print(f"执行点: {self.current_target_idx+1}/{len(self.target_poses_list)}")
            
            
if __name__ == '__main__':
    SCENE_XML_PATH = '/home/ethan/work/mujoco-learning-main/model/franka_emika_panda/scene_pos.xml'
    YAML_PATH = '/home/ethan/work/mujoco-learning-main/control/target_pos.yaml'
        
    robot = ArmMoveController(SCENE_XML_PATH, YAML_PATH, YAML_PATH)
    robot.run_loop()