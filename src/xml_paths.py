from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PANDA_MODEL_DIR = ROOT_DIR / "model/franka_emika_panda"

PANDA_POS_SCENE_XML = str(PANDA_MODEL_DIR / "scene_pos.xml")
PANDA_TAU_SCENE_XML = str(PANDA_MODEL_DIR / "scene_tau.xml")
PANDA_ONLINE_SERVO_SCENE_XML = str(PANDA_MODEL_DIR / "scene_online_servo.xml")
PANDA_IBVS_SCENE_XML = str(PANDA_MODEL_DIR / "scene_with_apriltag.xml")
PANDA_IBVS_POS_SCENE_XML = str(PANDA_MODEL_DIR / "scene_with_apriltag_pos.xml")
PANDA_PBVS_SCENE_XML = str(PANDA_MODEL_DIR / "panda_pbvs.xml")
