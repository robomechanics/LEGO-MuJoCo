# sweep_feet_gap.py
# Minimal parameter sweep for feet gap using your existing Duplo class.
# Keeps changes local and easy to read.

import copy
import mujoco
import numpy as np
import os
from src.sim import ProgressCallback
from src.rduplo import Duplo  # <-- change to actual import path

theta_bounds = {
    'x_mm': {'low': 0, 'high': 90},
    'y_mm': {'low': 0,  'high': 90},
}

def create_base_theta(args) -> np.ndarray:
    """Create array from args"""
    theta = []
    for key in args['design_params']['meta']:
        if key in theta_bounds:
            theta.append(args['design_params']['meta'][key])
    return np.array(theta)


def array_to_dict(array):
    """Convert a 2D numpy array of theta values to a list of dicts."""
    for row in array:
        theta_dict = {'x_mm': row[0], 'y_mm': row[1]}
    return theta_dict

def dict_to_array(dicts):
    """Convert a list of dicts to a 2D numpy array of theta values."""
    return np.array([[d['x_mm'], d['y_mm']] for d in dicts])
    

def make_args_base():
    args = {
        'name': 'feet-gap-sweep',
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
                'footstl_scaled_v4_1': [0, 0, 0],
                'footstl_scaled_v4_2': [0, 0, 0],
                'footstl_scaled_v4_3': [0, 0, 0],
                'footstl_scaled_v4_4': [0, 0, 0],
                'footstl_scaled_v4_5': [0, 0, 0],
                'footstl_scaled_v4_6': [0, 0, 0],
            },
            'mesh_scale': {
                'part_1': [1, 1, 1],
                'hip': [1, 1, 1],
                'leg_rod': [1, 1, 1.0],
            },
            'body_quat': {
                'motor': [0.995, 0.067, 0.005, 0.079],
            },
            'meta': {'x_mm': 50, 'y_mm': 50}
        }
    }
    return args

def correct_args_quats(args_dict, avg_quat):
    # Correct the body_quat for all relevant parts
    for part,quat in args_dict['design_params']['body_quat'].items():
        args_dict['design_params']['body_quat'][part] = avg_quat

def update_model(args_dict, theta):
    x = float(theta[0]) * 1e-3  # mm → m
    y = float(theta[1]) * 1e-3

    gpo = args_dict['design_params']['geom_pos_offset']

    gpo['ballfoot_1'] = [ +x, -y, 0.0 ]
    gpo['ballfoot_2'] = [ -x, -y, 0.0 ]

    gpo['footstl_scaled_v4_1'] = [ +x, -y, 0.0 ]
    gpo['footstl_scaled_v4_2'] = [ +x, -y, 0.0 ]
    gpo['footstl_scaled_v4_3'] = [ +x, -y, 0.0 ]
    gpo['footstl_scaled_v4_4'] = [ -x, -y, 0.0 ]
    gpo['footstl_scaled_v4_5'] = [ -x, -y, 0.0 ]
    gpo['footstl_scaled_v4_6'] = [ -x, -y, 0.0 ]


def evaluate_results(results):
    """Evaluate results dict and return a score."""
    score = 0
    wd = 10 # weight for displacement
    wfall = 50 # weight for falling
    score += wd * results['distance_m']
    score -= wfall * results['fell']
    return score

def es_step(pop=8, theta=None, theta_bounds=theta_bounds, scale=0.1):
    """Perform a single ES optimization step."""
    if theta is None or theta_bounds is None:
        raise ValueError("Both theta and theta_bounds must be provided for ES step.")

    # --- Step 1: compute per-parameter sigma from bounds ---
    order = list(theta_bounds.keys())  # canonical parameter order
    sigma_vec = np.array([
        (theta_bounds[k]['high'] - theta_bounds[k]['low']) * scale
        for k in order
    ], dtype=float)

    epsilons = np.random.randn(pop, len(sigma_vec))  # shape (pop, D)
    epsilons *= sigma_vec  # scale each dimension

    # --- Step 3: form antithetic samples ---
    


    lows = np.array([theta_bounds[k]['low'] for k in order])
    highs = np.array([theta_bounds[k]['high'] for k in order])

    theta_plus = np.clip(theta + epsilons, lows, highs)
    theta_minus = np.clip(theta - epsilons, lows, highs)

    deltaR = []
    base_args = make_args_base()
    for i in range(len(theta_plus)):
        results_plus = run_sim(theta_plus[i], base_args)
        results_minus = run_sim(theta_minus[i], base_args)
        dR = evaluate_results(results_plus) - evaluate_results(results_minus)
        deltaR.append(dR)

    print(f"Gen ΔR mean={np.mean(deltaR):.3f}, std={np.std(deltaR):.3f}")
    deltaR = np.array(deltaR)           # shape (pop,)
    grad = np.dot(deltaR, epsilons)     # weighted sum over εᵢ
    grad = grad / (2 * len(epsilons))   # divide by 2N
    grad = grad / sigma_vec             # scale back by σ (elementwise)
    alpha = 0.05   # learning rate (tune later)
    theta_new = theta + alpha * grad
    theta_new = np.clip(theta_new, lows, highs)

    return theta_new, grad, deltaR

    


def check_sway(base_args, theta):
    """Run a single sim for given (x,y) and return if it swayed."""
    # Make a deep copy so we never modify base_args in-place
    args = copy.deepcopy(base_args)
    update_model(args, theta)
    args['ctrl_dict']['leg_amp_deg'] = 0
    args['sim_time'] = 8.0

    robot = Duplo(args)
    progress_cb = ProgressCallback(args['sim_time'])
    callbacks = {"progress_bar": progress_cb.update}

    robot.run_sim(callbacks=callbacks)
    swayed = bool(robot.sway)
    q = robot.mean_quat / np.linalg.norm(robot.mean_quat)
    robot.close()

    # Only return the quaternion; let run_sim handle how to use it
    return swayed, q

def run_sim(theta, base_args):
    """Run a single sim for given (x,y) and return simple metrics."""
    # Always start from a pristine copy
    args = copy.deepcopy(base_args)
    update_model(args, theta)

    # Check sway but do NOT let it modify base_args
    swayed, q = check_sway(args, theta)
    if swayed:
        correct_args_quats(args, q)

    # Now create a *new* robot for the actual run
    robot = Duplo(args)

    # Ensure deterministic starting state
    if robot.model.nkey > 0:
        robot.data.qpos[:] = robot.model.key_qpos[0]
    else:
        robot.data.qpos[:] = 0  # fallback if no keyframe defined
    robot.data.qvel[:] = 0
    robot.data.ctrl[:] = 0

    progress_cb = ProgressCallback(args['sim_time'])
    callbacks = {"progress_bar": progress_cb.update}

    # Capture initial pose and forward direction
    motor_id = robot.model.body('motor').id
    p_start = robot.data.xpos[motor_id].copy()
    q = robot.data.xquat[motor_id].astype(np.float64)
    R_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R_flat, q)
    f_world = R_flat.reshape(3, 3)[:, 0]

    # Run the sim
    robot.run_sim(callbacks=callbacks)

    # Measure displacement along forward axis
    p_end = robot.data.xpos[motor_id].copy()
    forward_disp = np.dot(p_end - p_start, f_world)

    fell = bool(robot.check_fall())
    robot.close()

    return {
        'x_mm': float(theta[0]),
        'y_mm': float(theta[1]),
        'distance_m': round(forward_disp, 4),
        'fell': fell,
    }


def main():
    base_args = make_args_base()
    theta = create_base_theta(base_args)
    theta_opt, rewards = train_es(theta, theta_bounds, max_gens=10)

    print("\nRunning final validation sim...")
    final_results = run_sim(theta_opt, base_args)
    final_score = evaluate_results(final_results)
    print(f"Final optimized reward: {final_score:.3f}")


def train_es(theta_init, theta_bounds, max_gens=20, tol_theta=0.01, tol_reward=0.1):
    # """
    # Run multiple ES steps until convergence or max generations.
    # """
    theta = np.array(theta_init, dtype=float)
    args = make_args_base()
    prev_reward = 0.0
    reward_history = []
    order = list(theta_bounds.keys())

    for gen in range(max_gens):
        print(f"\n=== Generation {gen} ===")
        scale = 0.1
        # Perform one ES update
        theta_new, grad, deltaR = es_step(pop=16, theta=theta, theta_bounds=theta_bounds, scale=scale)
        scale = scale * 0.95  # decay scale over generations
        # Compute convergence metrics
        delta_theta = np.linalg.norm(theta_new - theta)
        mean_reward = np.mean(deltaR)
        reward_history.append(mean_reward)

        print(f"Δθ = {delta_theta:.4f}, mean ΔR = {mean_reward:.4f}")

        # Check for convergence (after first gen)
        if gen > 0:
            delta_reward = abs(mean_reward - prev_reward)
            if delta_theta < tol_theta or delta_reward < tol_reward:
                print(f"Converged at generation {gen}")
                break

        # Update state for next gen
        theta = theta_new
        prev_reward = mean_reward

    print(f"\nFinal optimized theta: x={theta[0]:.2f}mm, y={theta[1]:.2f}mm")
    print(f"Reward history: {np.round(reward_history, 3)}")
    return theta, reward_history



if __name__ == "__main__":
    main()
