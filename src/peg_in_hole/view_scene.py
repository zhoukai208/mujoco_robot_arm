import argparse
import time
import tempfile
from pathlib import Path

import mujoco
import mujoco.viewer


SCENE_XML = Path(__file__).resolve().with_name("scene.xml")
ROOT_DIR = Path(__file__).resolve().parents[2]
PANDA_MODEL_DIR = ROOT_DIR / "model/franka_emika_panda"
PANDA_INCLUDE_FROM_SRC = "../../model/franka_emika_panda/panda_vel.xml"
PANDA_INCLUDE_FROM_MODEL_DIR = "panda_vel.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="View the peg-in-hole MuJoCo scene.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run time in seconds. Use 0 to keep the viewer open until closed.",
    )
    return parser.parse_args()


def make_loadable_scene_xml():
    scene_text = SCENE_XML.read_text(encoding="utf-8")
    scene_text = scene_text.replace(
        f'<include file="{PANDA_INCLUDE_FROM_SRC}"/>',
        f'<include file="{PANDA_INCLUDE_FROM_MODEL_DIR}"/>',
    )

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix="peg_in_hole_",
        dir=PANDA_MODEL_DIR,
        delete=False,
        encoding="utf-8",
    )
    with temp_file:
        temp_file.write(scene_text)
    return Path(temp_file.name)


def main():
    args = parse_args()
    loadable_scene_xml = make_loadable_scene_xml()
    try:
        model = mujoco.MjModel.from_xml_path(str(loadable_scene_xml))
    finally:
        loadable_scene_xml.unlink(missing_ok=True)

    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    start_time = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.25]
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            if args.duration > 0 and time.time() - start_time >= args.duration:
                break


if __name__ == "__main__":
    main()
