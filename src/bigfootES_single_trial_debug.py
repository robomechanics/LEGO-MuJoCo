import copy
import numpy as np

from src.bigfootES import (
    evaluate_results,
    update_model,
    correct_args_quats,
)
from src.rduplo import Duplo
from src.sim import ProgressCallback


DEBUG_X_MM = -0.8
DEBUG_Y_MM = -13.0


def make_bigfoot_es_args():
    """Match bigfootES.py defaults as closely as possible."""
    return {
        'name': 'bigfoot-es-single-trial-debug',
        'robot_dir': './robots',
        'sim_time': 10.0,
        'video_dir': './videos',
        'video_fps': 30,
        'gui': False,
        'record': False,
        'ctrl_dict': {
            'Kp': 20,
            'Kd': 12,
            'leg_amp_deg': 30,
            'hip_omega': 0.7 * 2 * np.pi,
        },
        'design_params': {
            'geom_pos_offset': {
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
                'RightFoot': [0, 0, 0],
                'LeftFoot': [0, 0, 0],
            },
            'mesh_scale': {
                'part_1': [1, 1, 1],
                'hip': [1, 1, 1],
                'leg_rod': [1, 1, 1.0],
            },
            'body_quat': {
                'motor': [0.995, 0.067, 0.005, 0.079],
            },
            'meta': {'x_mm': DEBUG_X_MM, 'y_mm': DEBUG_Y_MM},
        }
    }


def make_rduplo_main_args():
    """Match src/rduplo.py main() defaults for the same x/y point."""
    return {
        'name': 'rduplo-single-trial-debug',
        'sim_time': 5.0,
        'com': False,
        'gui': False,
        'record': False,
        'robot_dir': 'robots',
        'video_dir': 'data/videos',
        'video_fps': 30,
        'ctrl_dict': {
            'Kp': 20,
            'Kd': 12,
            'leg_amp_deg': 30,
            'hip_omega': 0.7 * 2 * np.pi,
        },
        'design_params': {
            'body_pos_offset': {},
            'geom_pos_offset': {
                'LeftFoot': [DEBUG_X_MM * 1e-3, -DEBUG_Y_MM * 1e-3, 0.0],
                'RightFoot': [-DEBUG_X_MM * 1e-3, -DEBUG_Y_MM * 1e-3, 0.0],
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
                'leg_rod': [1, 1, 1.00],
            },
            'body_quat': {
                'motor': [0.967, -0.013, -0.003, 0.256],
            }
        }
    }


def summarize_run(label, args, robot, sway_result=None, applied_quat=None):
    stats = getattr(robot, 'stats', {})
    score = evaluate_results({
        'distance_m': float(stats.get('total_distance_m', 0.0)),
        'fell': bool(stats.get('fall', False)),
    })
    body_quat = args['design_params']['body_quat']['motor']
    print(f"\n=== {label} ===")
    print(
        f"sim_time={args['sim_time']}, "
        f"leg_amp_deg={args['ctrl_dict']['leg_amp_deg']}, "
        f"hip_omega={args['ctrl_dict']['hip_omega']:.4f}"
    )
    print(
        "body_quat.motor="
        f"[{body_quat[0]:.6f}, {body_quat[1]:.6f}, {body_quat[2]:.6f}, {body_quat[3]:.6f}]"
    )
    if sway_result is not None:
        print(f"sway_check_detected={sway_result}")
    if applied_quat is not None:
        print(
            "mean_quat_from_sway_check="
            f"[{applied_quat[0]:.6f}, {applied_quat[1]:.6f}, {applied_quat[2]:.6f}, {applied_quat[3]:.6f}]"
        )
    print(
        f"fell={bool(stats.get('fall', False))}, "
        f"distance_m={float(stats.get('total_distance_m', 0.0)):.6f}, "
        f"avg_speed_m_s={float(stats.get('avg_speed_m_s', 0.0)):.6f}, "
        f"final_yaw_deg={float(stats.get('final_yaw_deg', 0.0)):.6f}, "
        f"score={score:.6f}"
    )


def run_robot(args, reset_state):
    robot = Duplo(copy.deepcopy(args))
    if reset_state:
        if robot.model.nkey > 0:
            robot.data.qpos[:] = robot.model.key_qpos[0]
        else:
            robot.data.qpos[:] = 0
        robot.data.qvel[:] = 0
        robot.data.ctrl[:] = 0

    progress_cb = ProgressCallback(args['sim_time'])
    robot.run_sim(callbacks={"progress_bar": progress_cb.update})
    return robot


def run_bigfoot_es_style(theta_mm):
    args = make_bigfoot_es_args()
    update_model(args, theta_mm)

    sway_args = copy.deepcopy(args)
    sway_args['ctrl_dict']['leg_amp_deg'] = 0
    sway_args['sim_time'] = 8.0
    sway_robot = run_robot(sway_args, reset_state=False)
    swayed = bool(sway_robot.sway)
    mean_quat = sway_robot.mean_quat.copy() / np.linalg.norm(sway_robot.mean_quat)
    sway_robot.close()

    if swayed:
        correct_args_quats(args, mean_quat)

    robot = run_robot(args, reset_state=True)
    summarize_run("bigfootES style", args, robot, sway_result=swayed, applied_quat=mean_quat)
    robot.close()


def run_rduplo_main_style():
    args = make_rduplo_main_args()
    robot = run_robot(args, reset_state=False)
    summarize_run("rduplo.py main style", args, robot)
    robot.close()


def main():
    theta_mm = np.array([DEBUG_X_MM, DEBUG_Y_MM], dtype=float)
    print(f"Comparing single-trial behavior at x={theta_mm[0]} mm, y={theta_mm[1]} mm")
    run_rduplo_main_style()
    run_bigfoot_es_style(theta_mm)


if __name__ == "__main__":
    main()
