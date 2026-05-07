
import pinocchio
import numpy as np
from numpy.linalg import norm, solve
from pathlib import Path
from utils import *

class PandaKinematics:
    def __init__(self, arm_path=None):
        if arm_path is None:
            arm_path = Path(__file__).resolve().parents[1] / "model/franka_panda_urdf/robots/panda_arm.urdf"
        arm_path = str(arm_path)
        self.model = pinocchio.RobotWrapper.BuildFromURDF(arm_path).model if arm_path.endswith(".urdf") else pinocchio.RobotWrapper.BuildFromMJCF(arm_path).model
        self.data = self.model.createData()
        self.FRAME_ID = self.model.getFrameId("link7")
        self.EE_FRAME_ID = self.model.getFrameId("ee_center_body") if self.model.existFrame("ee_center_body") else self.FRAME_ID
        print(f"✅ Panda 运动学类初始化完成，关节数: {self.model.nq}")

    def J(self, q):
        q = np.asarray(q, dtype=np.float64).flatten()
        assert len(q) == 7, "雅可比输入必须是7维关节角"
        pinocchio.computeJointJacobians(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        return pinocchio.getFrameJacobian(
            self.model,
            self.data,
            self.EE_FRAME_ID,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:, :7].copy()

    def fk(self, q):
        q = np.asarray(q).flatten()
        assert len(q) == 7, "正解输入必须是7维关节角"
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.EE_FRAME_ID]
        pos = T.translation.copy()
        rot = T.rotation.copy()
        quat = rot_to_quat(rot)
        return pos, quat

    def ik(self, current_q, target_rot, target_pos,
           eps=1e-4, IT_MAX=1000, DT=1e-1, damp=1e-6):
        current_q = np.asarray(current_q).flatten()
        assert len(current_q) == 7, "逆解输入必须是7维关节角"
        q = current_q.copy()
        oMdes = pinocchio.SE3(target_rot, np.array(target_pos))

        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)
            iMd = self.data.oMf[self.FRAME_ID].actInv(oMdes)
            err = pinocchio.log(iMd).vector
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break

            J = pinocchio.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.FRAME_ID,
                pinocchio.ReferenceFrame.LOCAL,
            )
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * DT)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            i += 1

        if success:
            print("✅ IK 收敛成功！")
        else:
            print("❌ IK 未收敛")
        
        return success, q.flatten().tolist()
