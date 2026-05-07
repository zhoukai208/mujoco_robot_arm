import mujoco
import mujoco.viewer
import numpy as np
from typing import Optional, Dict, Tuple


class MocapManager:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        统一管理场景中所有的 mocap 物体
        :param model: mujoco MjModel
        :param data: mujoco MjData（必须是 viewer 绑定的那个）
        """
        self.model = model
        self.data = data

        # 缓存：name -> (mocap_id, geom_id)
        self._mocap_map: Dict[str, Tuple[int, int]] = {}
        self._scan_all_mocaps()

    def _scan_all_mocaps(self):
        """扫描 XML 中所有 mocap body，自动匹配对应的 geom（同名规则）"""
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not body_name:
                continue

            # 判断是不是 mocap 体
            if self.model.body_mocapid[body_id] >= 0:
                mocap_id = self.model.body_mocapid[body_id]

                # 寻找同名 geom（你的 XML 必须保持 body 和 geom 同名！）
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, body_name)
                if geom_id < 0:
                    print(f"[警告] Mocap {body_name} 未找到同名 geom")
                    continue

                self._mocap_map[body_name] = (mocap_id, geom_id)
                print(f"[MocapManager] 已加载: {body_name}")

    # -------------------------------------------------------------------------
    # 核心接口：根据名称设置位置
    # -------------------------------------------------------------------------
    def set_position(self, name: str, x: float, y: float, z: float):
        if name not in self._mocap_map:
            raise KeyError(f"不存在 mocap: {name}")
        mocap_id, _ = self._mocap_map[name]
        self.data.mocap_pos[mocap_id] = [x, y, z]

    # -------------------------------------------------------------------------
    # 核心接口：根据名称设置姿态（四元数 w,x,y,z）
    # -------------------------------------------------------------------------
    def set_quaternion(self, name: str, w: float, x: float, y: float, z: float):
        if name not in self._mocap_map:
            raise KeyError(f"不存在 mocap: {name}")
        mocap_id, _ = self._mocap_map[name]
        self.data.mocap_quat[mocap_id] = [w, x, y, z]

    # -------------------------------------------------------------------------
    # 便捷接口：一次性设置 位姿 (pos + quat)
    # -------------------------------------------------------------------------
    def set_pose(self, name: str, pos: np.ndarray, quat: Optional[np.ndarray] = None):
        if name not in self._mocap_map:
            raise KeyError(f"不存在 mocap: {name}")
        mocap_id, _ = self._mocap_map[name]
        self.data.mocap_pos[mocap_id] = pos
        if quat is not None:
            self.data.mocap_quat[mocap_id] = quat

    # -------------------------------------------------------------------------
    # 显示 / 隐藏
    # -------------------------------------------------------------------------
    def show(self, name: str):
        if name not in self._mocap_map:
            raise KeyError(f"不存在 mocap: {name}")
        _, geom_id = self._mocap_map[name]
        self.model.geom_rgba[geom_id][3] = 1.0

    def hide(self, name: str):
        if name not in self._mocap_map:
            raise KeyError(f"不存在 mocap: {name}")
        _, geom_id = self._mocap_map[name]
        self.model.geom_rgba[geom_id][3] = 0.0

    # -------------------------------------------------------------------------
    # 批量接口
    # -------------------------------------------------------------------------
    def show_all(self):
        for name in self._mocap_map:
            self.show(name)

    def hide_all(self):
        for name in self._mocap_map:
            self.hide(name)

    # -------------------------------------------------------------------------
    # 获取当前位姿（调试用）
    # -------------------------------------------------------------------------
    def get_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        if name not in self._mocap_map:
            raise KeyError(f"不存在 mocap: {name}")
        mocap_id, _ = self._mocap_map[name]
        pos = self.data.mocap_pos[mocap_id].copy()
        quat = self.data.mocap_quat[mocap_id].copy()
        return pos, quat