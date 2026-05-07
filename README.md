# MuJoCo Robot Arm

Franka Panda MuJoCo simulation and control scripts. The active Python entry points live in `src/`; robot descriptions and meshes live in `model/`; runtime configuration lives in `config/`.

## Layout

- `src/`: control, planning, data collection, BC training/inference, and IBVS scripts.
- `config/target_pos.yaml`: multi-point target pose configuration used by trajectory scripts.
- `model/`: MuJoCo XML, assets, and Panda URDF files.
- `artifacts/bc_reach/`: generated BC datasets, checkpoints, and loss curves.
- `assets/`: committed checkpoints and other non-code assets.

## Run

Use the project virtual environment when available:

```bash
source .venv/bin/activate
python src/arm_move.py
```


Common entry points:

```bash
python src/arm_manual_control.py
python src/joint_impedance_control.py
python src/joint_admittance_control.py
python src/ee_impedance_control.py
python src/bc_train.py
python src/bc_grasp.py
```

Most scripts are intended to be run from the repository root so generated files and relative paths are predictable.
