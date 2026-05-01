import src.mujoco_viewer as mujoco_viewer
import time
import mujoco
import cv2

class UrEnv(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        self.initial_pos = self.model.qpos0.copy()

    def runFunc(self):
        pass

if __name__ == "__main__":
    env = UrEnv("/home/ethan/work/mujoco-learning-main/mujoco_menagerie-main/universal_robots_ur5e/scene.xml")
    env.run_loop()
