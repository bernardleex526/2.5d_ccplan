# Reports

`baseline_small_world.md` / `metrics.json`: plan T25 baseline, measured on WSL2
(Ubuntu 22.04, Humble, Fortress 6.18 with Mesa llvmpipe rendering, RTX 4060 for
elevation_mapping_cupy) on `hill_world_small`, 2026-10-19, from one sequence:
test_mapping_loop -> test_global_elevation_map -> test_relocalization (3 starts,
worst value kept) -> test_gait_patrol (1 lap). Values only; thresholds are frozen
by the user at checkpoint 2 in `src/hill_patrol_bringup/config/thresholds.yaml`.

The full 1000-step world (`hill_world`) baseline is not included: a 450 m stair
route takes hours per run at the real-time factor reached under software
rendering. Run it on the native Ubuntu + RTX 3060 machine (plan K7).
