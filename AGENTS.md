# Repository Guidelines

## Project Structure & Module Organization

This repository contains MuJoCo robot arm demos, controllers, and learning examples. Main Python scripts live in `src/`; shared helpers include `mujoco_viewer.py`, `utils.py`, planners, kinematics, and visual servoing modules. The peg-in-hole workflow is in `src/peg_in_hole/` with its own scene and task assets. MuJoCo XML, URDF/SRDF, meshes, and textures are under `model/`. Configuration YAML files are in `config/`. Generated datasets, plots, and model outputs belong in `artifacts/`; reusable checkpoints and extra resources are in `assets/`. Run scripts from the repository root so relative paths resolve correctly.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create and activate an environment.
- `pip install -r requirements.txt`: install MuJoCo, robotics, vision, and learning dependencies.
- `python src/arm_move.py`: run the target-pose trajectory demo using `config/target_pos.yaml`.
- `python src/traj_time_scaling_demo.py --methods linear trapezoidal quintic s_curve`: generate trajectory comparison artifacts.
- `python src/peg_in_hole/view_scene.py --duration 5`: smoke-test the peg-in-hole scene load.
- `python src/gen_grasp_data.py && python src/bc_train.py`: regenerate and train the behavior cloning reach workflow.

Some demos open MuJoCo or OpenCV windows; close the viewer or press `Ctrl+C` to stop.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Prefer explicit imports when practical, constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions/variables in `snake_case`. Keep path handling based on `pathlib.Path` or existing `ROOT_DIR`/`xml_paths.py` helpers. Follow the controller pattern: subclass `ArmBaseViewer`, put setup in `runBefore`, per-step behavior in `runFunc`, and guard demos with `if __name__ == '__main__':`.

## Testing Guidelines

There is no automated test suite configured yet. Validate changes with the narrowest runnable demo that exercises the touched code. For scene/model edits, use `python src/peg_in_hole/view_scene.py --duration 5` or the relevant viewer script. For trajectory logic, run `src/traj_time_scaling_demo.py` and inspect `artifacts/trajectory_time_scaling/`. Avoid committing large regenerated artifacts unless they are intentional.

## Commit & Pull Request Guidelines

Recent history uses short summaries in English or Chinese, for example `Split peg-in-hole task controllers`, `update readme`, or `优化轴孔装配`. Keep commits focused and mention the affected demo or subsystem. Pull requests should include a concise description, commands run, artifacts or checkpoints changed, and screenshots or notes for visual behavior changes. Link related issues when available.

## Security & Configuration Tips

Do not hard-code machine-specific absolute paths. Keep YAML configuration small and reviewable. Treat checkpoint files, datasets, and plots as large artifacts: add or update them only when they are needed to reproduce the documented behavior.
