"""
Eye-in-Hand 相机实时可视化

功能:
1. 加载带有腕部相机的 Panda 机械臂模型
2. 在 MuJoCo viewer 中显示仿真场景
3. 在独立窗口中实时显示腕部相机图像

控制:
- 拖拽机械臂: 在 MuJoCo viewer 中鼠标拖拽
- 退出: 在图像窗口按 ESC 或 Q, 或关闭 MuJoCo viewer
"""

import mujoco
import mujoco.viewer
import numpy as np
import glfw
import cv2
import time


class EyeInHandCameraViewer:
    def __init__(self, scene_xml_path):
        """
        初始化 eye-in-hand 相机可视化器
        
        Args:
            scene_xml_path: MuJoCo 场景 XML 文件路径
        """
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 设置初始位姿
        self.initial_pos = self.model.key_qpos[0] if self.model.nkey > 0 else self.model.qpos0
        self.data.qpos[:len(self.initial_pos)] = self.initial_pos
        
        # 相机参数
        self.camera_name = "eye_in_hand"
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        
        if self.camera_id == -1:
            raise ValueError(f"未找到相机 '{self.camera_name}', 请检查 XML 配置")
        
        print(f"✓ 成功加载相机: {self.camera_name} (ID: {self.camera_id})")
        
        # 图像分辨率
        self.width = 640
        self.height = 480
        
        # 初始化离屏渲染
        self._init_offscreen_rendering()
        
        # 末端执行器 ID
        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body'
        )
    
    def _init_offscreen_rendering(self):
        """初始化 OpenGL 离屏渲染上下文"""
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.glfw_window = glfw.create_window(
            self.width, self.height, "Eye-in-Hand Camera", None, None
        )
        glfw.make_context_current(self.glfw_window)
        
        # 创建 MuJoCo 渲染场景和上下文
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        
        # 设置离屏缓冲
        self.framebuffer = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        
        # 创建相机视图对象
        self.camera_view = mujoco.MjvCamera()
        self.camera_view.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera_view.fixedcamid = self.camera_id
    
    def render_camera_image(self):
        """
        渲染当前帧的腕部相机图像
        
        Returns:
            BGR 格式的 numpy 数组 (OpenCV 格式)
        """
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        
        # 更新场景
        mujoco.mjv_updateScene(
            self.model, 
            self.data, 
            mujoco.MjvOption(),
            mujoco.MjvPerturb(), 
            self.camera_view,
            mujoco.mjtCatBit.mjCAT_ALL, 
            self.scene
        )
        
        # 渲染
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        # 读取像素
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        
        # 转换为 BGR 并翻转 (MuJoCo 的 y 轴是反的)
        bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        
        return bgr
    
    def get_ee_pose(self):
        """获取末端执行器位姿"""
        if self.ee_id == -1:
            return None, None
        pos = self.data.body(self.ee_id).xpos.copy()
        quat = self.data.body(self.ee_id).xquat.copy()
        return pos, quat
    
    def show_info(self, image):
        """在图像上显示信息文字"""
        pos, quat = self.get_ee_pose()
        
        # 添加信息文字
        if pos is not None:
            info_text = f"EE Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            cv2.putText(
                image, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        cv2.putText(
            image, "Press ESC or Q to quit", (10, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        return image
    
    def run(self):
        """主运行循环"""
        print("\n" + "="*60)
        print("🎥 Eye-in-Hand Camera Visualization")
        print("="*60)
        print(f"📷 相机名称: {self.camera_name}")
        print(f"📐 分辨率: {self.width}x{self.height}")
        print(f"🤖 场景文件: {self.model_path}")
        print("\n💡 提示:")
        print("  - 在 MuJoCo Viewer 中用鼠标拖拽机械臂")
        print("  - 按 ESC 或 Q 退出")
        print("="*60 + "\n")
        
        # 启动 MuJoCo Viewer (阻塞式,在另一个窗口)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 3
            viewer.cam.azimuth = -45
            viewer.cam.elevation = -30
            
            print("✓ MuJoCo Viewer 已启动")
            
            while viewer.is_running():
                # 物理步进
                mujoco.mj_step(self.model, self.data)
                
                # 渲染相机图像
                image = self.render_camera_image()
                image = self.show_info(image)
                
                # 显示图像
                cv2.imshow('Eye-in-Hand Camera', image)
                
                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q')]:  # ESC 或 Q
                    print("\n⚠️  收到退出信号,正在关闭...")
                    break
                
                # 同步 viewer
                viewer.sync()
                
                # 控制帧率
                time.sleep(self.model.opt.timestep)
        
        # 清理资源
        cv2.destroyAllWindows()
        glfw.terminate()
        print("✓ 资源已清理")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Eye-in-Hand Camera 实时可视化')
    parser.add_argument(
        '--scene', 
        type=str, 
        default='./model/franka_emika_panda/scene.xml',
        help='MuJoCo 场景 XML 文件路径 (默认: ./model/franka_emika_panda/scene.xml)'
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=640,
        help='相机图像宽度 (默认: 640)'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=480,
        help='相机图像高度 (默认: 480)'
    )
    
    args = parser.parse_args()
    
    # 创建并运行可视化器
    viewer = EyeInHandCameraViewer(args.scene)
    viewer.width = args.width
    viewer.height = args.height
    viewer.model_path = args.scene
    viewer.run()


if __name__ == "__main__":
    main()
