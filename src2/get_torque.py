import os
# 关键！屏蔽 OpenCV 的 Qt 插件，解决 xcb 报错
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["OPENCV_OPENCL_RUNTIME"] = ""
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import src.mujoco_viewer as mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import time, math

# 必须加这句！matplotlib 避免 Qt 冲突
plt.rcParams['backend'] = 'TkAgg'

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=180, elevation=-30)
        self.path = path
        self.plot_done = False  # 防止重复绘图
    
    def runBefore(self):
        # 存储关节力矩的列表
        self.torque_history = []
        self.time_history = []
       
    def runFunc(self):
        # 记录力矩
        self.time_history.append(self.data.time)
        self.torque_history.append(self.data.qfrc_actuator.copy())

        # 达到长度后绘图（只画一次）
        if len(self.torque_history) > 2000 and not self.plot_done:
            self.plot_done = True
            torque_history = np.array(self.torque_history)
            
            plt.figure(figsize=(10, 6))
            for i in range(torque_history.shape[1]):
                plt.subplot(torque_history.shape[1], 1, i + 1)
                plt.plot(self.time_history, torque_history[:, i], label=f'Joint {i+1} Torque')
                plt.xlabel('Time (s)')
                plt.ylabel('Torque (N·m)')
                plt.title(f'Joint {i+1} Torque')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    test = Test("model/trs_so_arm100/scene.xml")
    test.run_loop()