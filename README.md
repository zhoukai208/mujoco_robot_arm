# MuJoCo 机械臂 Demo

本项目基于 MuJoCo 搭建 Franka Panda 机械臂仿真，包含轨迹运动、手动控制、阻抗/导纳控制、视觉伺服、行为克隆和插孔任务示例。请在仓库根目录运行脚本，避免相对路径失效。

## 运行准备

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

大多数可视化 demo 会打开 MuJoCo viewer。通用鼠标操作：左键拖拽旋转视角，右键拖拽平移视角，滚轮缩放；关闭 viewer 或按 `Ctrl+C` 结束程序。

## Demo 列表

### 基础运动

- `python src/arm_move.py`
  - 功能：读取 `config/target_pos.yaml` 中的多个目标位姿，自动规划并依次执行末端轨迹。
  - 操作：无需键盘控制；MuJoCo viewer 可用鼠标调整视角，OpenCV 相机窗口只用于观察。

- `python src/arm_manual_control.py`
  - 功能：手动控制机械臂关节或末端位置。
  - 操作：先点击/聚焦 `Manual Control Camera` 窗口。
  - 按键：`Tab` 切换 `JOINT/CART` 模式；`JOINT` 模式下 `1`-`7` 分别增加对应关节角，`g` 闭合夹爪，`h` 张开夹爪；`CART` 模式下方向键移动末端 `X/Y`，`PageUp/PageDown` 移动末端 `Z`。

### 力控示例

- `python src/joint_impedance_control.py`
  - 功能：关节空间阻抗控制，保持初始关节目标并做重力补偿。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。

- `python src/joint_admittance_control.py`
  - 功能：关节空间导纳控制示例，内部生成期望关节轨迹并用 PD 力矩跟踪。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。

- `python src/ee_impedance_control.py`
  - 功能：任务空间末端阻抗控制，保持启动时的末端位姿，并带 null-space 稳定项。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。

### 视觉伺服

- `python src/pbvs.py`
  - 功能：PBVS 位姿伺服。启动后随机放置方块，机械臂先到初始姿态，再靠近方块上方目标位姿。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。

- `python src/ibvs.py`
  - 功能：IBVS 图像视觉伺服。机械臂先移动到观察位姿，再根据 AprilTag 图像误差控制末端运动。
  - 操作：先点击/聚焦 `IBVS Servo` 窗口。
  - 按键：方向键移动 AprilTag 目标的 `X/Y`；`u/o` 调整 `Z`；`q/e` 调整 roll；`w/s` 调整 pitch；`a/d` 调整 yaw；`t` 将当前检测到的 tag 像素设为目标。

- `python src/ibvs_pos.py`
  - 功能：与 `ibvs.py` 类似，但使用关节位置执行器积分速度指令，更适合位置控制模型。
  - 操作：同 `ibvs.py`。

### 行为克隆 Reach 流程

- `python src/gen_grasp_data.py`
  - 功能：无渲染后台生成 BC reach 数据集，保存到 `artifacts/bc_reach/bc_reach_dataset.pkl`。
  - 操作：无需键盘/鼠标控制。

- `python src/arm_grasp.py`
  - 功能：带可视化的数据采集版本，随机放置方块、求解 IK、到位后保存样本。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。

- `python src/load_grasp_data.py`
  - 功能：加载并随机回放 `bc_reach_dataset.pkl`，用于可视化检查数据是否正确。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。

- `python src/bc_train.py`
  - 功能：训练 BC 模型，读取 `bc_reach_dataset.pkl`，保存最优模型、最终模型和 loss 曲线到 `artifacts/bc_reach/`。
  - 操作：无需键盘/鼠标控制。

- `python src/bc_grasp.py`
  - 功能：加载 `bc_reach_best_model.pth`，随机放置方块并用模型预测目标关节角，统计成功/失败次数。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。

推荐顺序：

```bash
python src/gen_grasp_data.py
python src/bc_train.py
python src/load_grasp_data.py
python src/bc_grasp.py
```

### Peg-in-Hole 插孔任务

- `python src/peg_in_hole/view_scene.py`
  - 功能：只加载并查看插孔场景。
  - 操作：无需键盘控制；MuJoCo viewer 鼠标调整视角。
  - 可选参数：`--duration 5` 表示运行 5 秒后自动退出；默认 `0` 表示一直运行到关闭窗口。

- `python src/peg_in_hole/grasp_pin_demo.py`
  - 功能：完整插孔流程：规划抓取 peg、闭合夹爪、抬起、通过 IBVS 对准孔位、下探插入、松爪撤回。
  - 操作：流程自动执行，无需键盘控制；MuJoCo viewer 鼠标调整视角，OpenCV 相机窗口显示 tag 对准过程。

## 目录说明

- `src/`：主要 demo、控制器、视觉伺服和训练脚本。
- `src/peg_in_hole/`：插孔任务场景和流程脚本。
- `model/`：MuJoCo XML、网格和纹理资源。
- `config/`：目标位姿、IBVS 目标像素等配置。
- `artifacts/bc_reach/`：生成的数据集、模型权重和训练曲线。
- `assets/`：提交到仓库的额外资源。
