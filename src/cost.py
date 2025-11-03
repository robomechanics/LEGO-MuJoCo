# sweep_feet_gap.py
# Minimal parameter sweep for feet gap using your existing Duplo class.
# Keeps changes local and easy to read.

import copy
import numpy as np
import os
from src.sim import ProgressCallback
from src.rduplo import Duplo  # <-- change to actual import path

# ---------- small helpers ----------

def make_args_base():
    """Return a minimal args dict you already use in main()."""
    args = {
        'name': 'feet-gap-sweep',
        'robot_dir': './robots',         # <-- set to your actual root
        'sim_time': 10.0,                 # seconds
        'video_dir': './videos',
        'video_fps': 30,
        'gui': False,
        'record': False,                 # keep False for speed while sweeping
        'ctrl_dict': {
            'Kp': 20,
            'Kd': 12,
            'leg_amp_deg': 30,
        },
        'design_params': {
            'geom_pos_offset': {
                # These will be filled by set_feet_gap()
                'hip_rod_1': [0, 0, 0],
                'leg_rod_1': [0.0, 0, 0],
                'motor_part1_1': [0, 0, 0],
                'motor_part2_1': [0, 0, 0],
                'arm_rod_1': [0, 0, 0],
                'battery_1': [0, 0, 0],
                'hip_rod_2': [0, 0, 0],
                'motor_part3_1': [0, 0, 0],
                'leg_rod_2': [0.0, 0, 0],
                'arm_rod_2': [0, 0, 0],
                'battery_2': [0, 0, 0],
            },
            'mesh_scale': {
                'part_1': [1, 1, 1],
                'hip': [1, 1, 1],
                'leg_rod': [1, 1, 1.0],
            },
            'body_quat': {
                'motor': [0.995, 0.067, 0.005, 0.079],
            }
        }
    }
    return args

def correct_args_quats(args_dict, avg_quat):
    # Correct the body_quat for all relevant parts
    for part,quat in args_dict['design_params']['body_quat'].items():
        args_dict['design_params']['body_quat'][part] = avg_quat

def set_feet_gap(args_dict, x_mm, y_mm):
    """
    Inject foot gap offsets (x,y) into the geom positions.
    Your original code offsets these six 'footstl_scaled_v4_*' geoms.
    """
    x = x_mm * 10e-4  # match your code’s scaling (meters)
    y = y_mm * 10e-4

    foot_keys_left  = ['footstl_scaled_v4_1','footstl_scaled_v4_2','footstl_scaled_v4_3']
    foot_keys_right = ['footstl_scaled_v4_4','footstl_scaled_v4_5','footstl_scaled_v4_6']

    gpo = args_dict['design_params']['geom_pos_offset']  # shorthand

    # apply or create entries if missing
    for k in foot_keys_left:
        gpo[k] = [ x, -y, 0]
    for k in foot_keys_right:
        gpo[k] = [-x, -y, 0]


def check_sway(x_mm, y_mm, base_args):
    """Run a single sim for given (x,y) and return if it swayed."""
    args = copy.deepcopy(base_args)
    set_feet_gap(args, x_mm, y_mm)
    args['ctrl_dict']['leg_amp_deg'] = 0
    args['sim_time'] = 8.0
    robot = Duplo(args)
    progress_cb = ProgressCallback(args['sim_time'])
    callbacks = {"progress_bar": progress_cb.update}

    robot.run_sim(callbacks=callbacks)

    swayed = bool(robot.sway)
    print(f'Sway check for x={x_mm}mm, y={y_mm}mm: {swayed}')

    robot.close()
    return (swayed, robot.mean_quat)

def run_once(x_mm, y_mm, base_args):
    """Run a single sim for given (x,y) and return simple metrics."""
    args = copy.deepcopy(base_args)
    set_feet_gap(args, x_mm, y_mm)

    robot = Duplo(args)
    progress_cb = ProgressCallback(args['sim_time'])
    callbacks = {"progress_bar": progress_cb.update}

    robot.run_sim(callbacks=callbacks)

    # --- simple metrics (keep it robust & cheap) ---
    # Try to read the forward displacement of the body you track ("motor"),
    # fallback to world COM if needed.
    try:
        motor_id = robot.model.body('motor').id
        x_disp = robot.data.body_xpos[motor_id][0]  # world x (meters)
    except Exception:
        # COM as fallback
        x_disp = float(robot.mass_center()[0])

    fell = bool(robot.check_fall())
    sway = bool(robot.sway)
    print(sway)

    robot.close()
    return {
        'x_mm': x_mm,
        'y_mm': y_mm,
        'distance_m': round(x_disp, 4),
        'fell': fell,
        'sway': sway,
    }

def main():
    base_args = make_args_base()

    # Define a tiny grid to keep it fast; expand later.
    x_list_mm = [20, 30, 40, 50, 60, 70, 80]  # wider gap as x increases
    y_list_mm = [40, 50, 60]          # closer to rod as y increases 

    results = []
    for y in y_list_mm:
        for x in x_list_mm:
            print(f"Running x={x}mm, y={y}mm …")
            (sway, avg_quat) = check_sway(x, y, base_args)
            if sway:
                print('Sway detected; grabbing average quats.')
                correct_args_quats(base_args, avg_quat)
            out = run_once(x, y, base_args)
            results.append(out)

    # Print a compact table
    print("\n=== Feet-gap sweep results ===")
    print(f"{'x(mm)':>6} {'y(mm)':>6} {'dist(m)':>8} {'fell':>5} {'sway':>5}")
    for r in results:
        print(f"{r['x_mm']:>6} {r['y_mm']:>6} {r['distance_m']:>8.3f} {str(r['fell']):>5} {str(r['sway']):>5}")

    viable = [r for r in results if not r['fell']]
    if viable:
        best = max(viable, key=lambda r: r['distance_m'])
        print("\nBest (no-fall, max distance):", best)
    else:
        print("\nNo stable runs (all fell); consider reducing leg_amp_deg or run longer t_wait.")

if __name__ == "__main__":
    main()
