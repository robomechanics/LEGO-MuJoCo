# Experiment data notes

Record of Bigfoot MuJoCo search attempts: what we tried, what failed, and what to keep.

## Live tooling

| Script | Role |
|--------|------|
| `run_evolutionary_controller.py` | Final evolutionary strategy (ES) entrypoint |
| `test_sim_sweep.py` | Trial runner + metrics used by ES |
| `estimate_settle_quat.py` | Zero-amp settle; bake spawn quat into `Bigfoot/robot.xml` |
| `plot_es_foot_offsets.py` | Foot-offset exploration plots from ES history CSVs |
| `run_offset_control_grid.py` | Fixed-controller offset grid (used after walkgait_v12) |
| `test_sim_closedloop.py` | Interactive / recorded replay of a chosen controller |

## Kept ES artifacts (`data/es/`)

- **walkgait_v10** — Strong peak meters, but brittle: high `best_fit` with low population full-survival; robustness table showed 100%-robust neighbors beat the fitness champ for closedloop.
- **walkgait_v12** — Continued ES + offset-control grid around promising `(fx, fy)`.
- **walkgait_v13** — Latest ES snapshot (histories, logs, summaries, `_best/` dumps, offset plot).

## What we stopped continuing (and why)

### Early ES / frequency / “forward-only” runs
Runs tagged `0.524hz`, `fwdonly*`, `newmodel`, bare `walkgait` / `v2`–`v4` mostly optimized survival + raw travel. They often rewarded shuffle or sideways progress, locked onto foot-offset bounds, or did not transfer to closedloop. Superseded once gait quality (steps, clearance, slip) and anti-reverse terms were added.

### walkgait_v5–v9
Useful learning steps (30 s trials, exploration inject, yaw-locked settle) but intermediate. Fitness still preferred distance over robustness; v8/v9 showed elite loss / score-gap freeze. Histories deleted to cut clutter; lessons folded into later fitness/selection.

### walkgait_v11 (partial)
First robust-probe fitness attempt; aborted early. Population survival stayed thin despite `robust=100%` on the elite. Not continued as the main line.

### Parameter sweeps (`data/sweeps/`, amp/control/ridge/robust_controller scripts)
Grid/slice sweeps were exploratory. They seeded early ES ideas but did not produce the final gait; scripts and CSVs removed once ES + settle + closedloop became the workflow.

### Intermediate videos
Demo clips (`backwards`, `best_trial_0.524hz`, `es_best_sweep_seed`, `robust_controller_best`, offset-rank mp4s) were one-off checks. Kept `data/videos/bipedal-walking.mp4` as the representative recording.

## Practical takeaways (still true)

1. **Settle matters** — ES re-settles per trial; closedloop needs the same yaw-locked startup settle or baked quat for those offsets.
2. **Don’t trust `best_fit` alone** — Prefer high forward + straightness + local Kp/Kd survival for hardware/closedloop.
3. **Offset prior** — Successful stepping walkers clustered near small `|fx|` and `fy ≈ -0.02` to `-0.03` (exact values in v12/v13 summaries).
