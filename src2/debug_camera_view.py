"""
调试相机视野 - 查看 eye-to-hand 相机实际看到的图像
"""

import mujoco
import glfw
import numpy as np
import cv2

# 加载模型
model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene_calibration_simple.xml')
data = mujoco.MjData(model)

# 初始化 GLFW
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(640, 480, "Debug", None, None)
glfw.make_context_current(window)

scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

# 获取相机
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_to_hand")
print(f"相机 ID: {camera_id}")
print(f"相机数量: {model.ncam}")

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
camera.fixedcamid = camera_id

# 渲染并保存前 10 帧
for i in range(10):
    mujoco.mj_step(model, data)
    
    viewport = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(),
                         mujoco.MjvPerturb(), camera,
                         mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, context)
    bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    
    # 显示
    cv2.imshow(f'Frame {i}', bgr)
    cv2.waitKey(500)
    
    # 保存
    cv2.imwrite(f'debug_camera_frame_{i}.png', bgr)
    print(f"保存帧 {i}")

cv2.destroyAllWindows()
glfw.terminate()

print("\n查看保存的图像: debug_camera_frame_*.png")
print("如果图像中没有棋盘格，说明:")
print("  1. 相机位置/角度不对")
print("  2. 标定板位置不对")
print("  3. 纹理加载失败")
