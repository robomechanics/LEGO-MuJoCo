# sweep_feet_gap.py
# Minimal parameter sweep for feet gap using your existing Duplo class.
# Keeps changes local and easy to read.

import copy
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.sim import ProgressCallback
from src.rduplo import Duplo  # <-- change to actual import path

theta_bounds = {
    'x_mm': {'low': -90, 'high': 90},
    'y_mm': {'low': -90,  'high': 90},
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
            'meta': {'x_mm': 0, 'y_mm': -13}
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

    
    gpo['LeftFoot'] = [ x, -y, 0.0 ]
    gpo['RightFoot'] = [ -x, -y, 0.0 ]


def evaluate_results(results):
    """Evaluate results dict and return a score."""
    score = 0
    wd = 10 # weight for displacement
    wfall = 30 # weight for falling
    score += wd * results['distance_m']
    score -= wfall * results['fell']
    return score

def es_step(pop=8, theta=None, theta_bounds=theta_bounds, scale=0.2, alpha=0.1):
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
    trials = []
    base_args = make_args_base()
    for i in range(len(theta_plus)):
        results_plus = run_sim(theta_plus[i], base_args)
        results_minus = run_sim(theta_minus[i], base_args)
        score_plus = evaluate_results(results_plus)
        score_minus = evaluate_results(results_minus)
        dR = score_plus - score_minus
        deltaR.append(dR)
        trials.append({
            'x_mm': results_plus['x_mm'],
            'y_mm': results_plus['y_mm'],
            'score': score_plus,
            'sample_type': 'plus',
        })
        trials.append({
            'x_mm': results_minus['x_mm'],
            'y_mm': results_minus['y_mm'],
            'score': score_minus,
            'sample_type': 'minus',
        })

    print(f"Gen ΔR mean={np.mean(deltaR):.3f}, std={np.std(deltaR):.3f}")
    deltaR = np.array(deltaR)           # shape (pop,)
    grad = np.dot(deltaR, epsilons)     # weighted sum over εᵢ
    grad = grad / (2 * len(epsilons))   # divide by 2N
    grad = grad / sigma_vec             # scale back by σ (elementwise)
    theta_new = theta + alpha * grad
    theta_new = np.clip(theta_new, lows, highs)

    return theta_new, grad, deltaR, trials

    


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

    robot.run_sim(callbacks=callbacks)
    robot.close()

    return {
        'x_mm': float(theta[0]),
        'y_mm': float(theta[1]),
        'distance_m': round(robot.stats['total_distance_m'], 4),
        'fell': robot.stats['fall'],
    }


def main():
    base_args = make_args_base()
    theta = create_base_theta(base_args)
    theta_opt, history, all_trials = train_es(theta, theta_bounds, max_gens=10)

    print("\nRunning final validation sim...")
    final_results = run_sim(theta_opt, base_args)
    final_score = evaluate_results(final_results)
    print(f"Final optimized reward: {final_score:.3f}")

    plot_trials(all_trials, output_path="data/es_trials.png")
    plot_reward_history(history, output_path="data/es_history.png")


def plot_trials(trials: list, output_path: str = "data/es_trials.png") -> None:
    """Scatter plot of all ES trials: x/y offsets coloured red→green by reward."""
    x = np.array([t['x_mm'] for t in trials])
    y = np.array([t['y_mm'] for t in trials])
    scores = np.array([t['score'] for t in trials])

    best_idx = int(np.argmax(scores))
    bx, by, bs = x[best_idx], y[best_idx], scores[best_idx]

    fig, ax = plt.subplots(figsize=(8, 7))

    norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
    sc = ax.scatter(x, y, c=scores, cmap='RdYlGn', norm=norm,
                    s=70, edgecolors='k', linewidths=0.5, zorder=2)

    ax.scatter(bx, by, s=250, marker='*', color='gold',
               edgecolors='k', linewidths=1.0, zorder=3, label='Best')

    # Callout offset away from plot edges
    offset_x = 15 if bx < 0 else -15
    offset_y = 15 if by < 0 else -15
    ax.annotate(
        f"Best\nx = {bx:.1f} mm\ny = {by:.1f} mm\nR = {bs:.2f}",
        xy=(bx, by),
        xytext=(bx + offset_x, by + offset_y),
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor='gold', linewidth=1.5, alpha=0.95),
        arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1.5),
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Reward', fontsize=11)

    ax.set_xlabel('X offset (mm)', fontsize=11)
    ax.set_ylabel('Y offset (mm)', fontsize=11)
    ax.set_title(f'ES Trial Rewards — {len(trials)} evaluations', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ES plot] Saved to {output_path}")


def plot_reward_history(history: list, output_path: str = "data/es_history.png") -> None:
    """Plot actual score statistics and ES directional signal over generations."""
    gens = np.array([h['gen'] for h in history], dtype=int)
    mean_scores = np.array([h['mean_score'] for h in history], dtype=float)
    best_scores = np.array([h['best_score'] for h in history], dtype=float)
    mean_delta_r = np.array([h['mean_deltaR'] for h in history], dtype=float)
    scales = np.array([h['scale'] for h in history], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(gens, mean_scores, marker='o', label='Mean score')
    axes[0].plot(gens, best_scores, marker='*', markersize=12, label='Best score')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('ES Reward History')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(gens, mean_delta_r, marker='s', label='Mean ΔR')
    axes[1].plot(gens, scales, marker='^', label='Exploration scale')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Signal / Scale')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ES history plot] Saved to {output_path}")


def plot_trials_by_generation(trials: list, output_path: str = "data/es_trials_by_generation.png") -> None:
    """Scatter one panel per generation so the search trajectory is easier to read."""
    if not trials:
        return

    gens = sorted({int(t['gen']) for t in trials})
    ncols = min(3, len(gens))
    nrows = int(np.ceil(len(gens) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    scores = np.array([t['score'] for t in trials], dtype=float)
    norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
    flat_axes = axes.flatten()
    sc = None

    for ax, gen in zip(flat_axes, gens):
        gen_trials = [t for t in trials if int(t['gen']) == gen]
        x = np.array([t['x_mm'] for t in gen_trials], dtype=float)
        y = np.array([t['y_mm'] for t in gen_trials], dtype=float)
        gen_scores = np.array([t['score'] for t in gen_trials], dtype=float)

        sc = ax.scatter(x, y, c=gen_scores, cmap='RdYlGn', norm=norm,
                        s=70, edgecolors='k', linewidths=0.5)

        best_idx = int(np.argmax(gen_scores))
        ax.scatter(x[best_idx], y[best_idx], s=220, marker='*', color='gold',
                   edgecolors='k', linewidths=1.0, zorder=3)

        ax.set_title(f'Generation {gen}')
        ax.set_xlabel('X offset (mm)')
        ax.set_ylabel('Y offset (mm)')
        ax.grid(True, alpha=0.3)

    for ax in flat_axes[len(gens):]:
        ax.axis('off')

    if sc is not None:
        cbar = fig.colorbar(sc, ax=flat_axes.tolist(), shrink=0.95)
        cbar.set_label('Reward')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ES generation plot] Saved to {output_path}")


def train_es(theta_init, theta_bounds, max_gens=20, tol_theta=0.01, tol_reward=0.1):
    theta = np.array(theta_init, dtype=float)
    scale = 0.10
    scale_decay = 0.95
    alpha = 0.1
    history = []
    all_trials = []
    best_overall = None

    for gen in range(max_gens):
        print(f"\n=== Generation {gen} ===")
        theta_new, grad, deltaR, trials = es_step(pop=32, theta=theta,
                                                  theta_bounds=theta_bounds,
                                                  scale=scale,
                                                  alpha=alpha)
        for trial in trials:
            trial['gen'] = gen
        all_trials.extend(trials)

        scores = np.array([trial['score'] for trial in trials], dtype=float)
        best_trial = max(trials, key=lambda trial: trial['score'])
        if best_overall is None or best_trial['score'] > best_overall['score']:
            best_overall = dict(best_trial)

        delta_theta = np.linalg.norm(theta_new - theta)
        mean_score = float(np.mean(scores))
        best_score = float(np.max(scores))
        mean_delta_r = float(np.mean(deltaR))
        history.append({
            'gen': gen,
            'scale': scale,
            'delta_theta': float(delta_theta),
            'mean_score': mean_score,
            'best_score': best_score,
            'mean_deltaR': mean_delta_r,
            'best_x_mm': float(best_trial['x_mm']),
            'best_y_mm': float(best_trial['y_mm']),
        })

        print(
            f"scale={scale:.4f}, Δθ={delta_theta:.4f}, "
            f"mean score={mean_score:.4f}, best score={best_score:.4f}, "
            f"mean ΔR={mean_delta_r:.4f}"
        )

        # if gen > 0:
        #     prev_mean_score = history[-2]['mean_score']
        #     delta_reward = abs(mean_score - prev_mean_score)
        #     if delta_theta < tol_theta or delta_reward < tol_reward:
        #         print(f"Converged at generation {gen}")
        #         theta = theta_new
        #         break

        theta = theta_new
        scale *= scale_decay

    print(f"\nFinal optimized theta: x={theta[0]:.2f}mm, y={theta[1]:.2f}mm")
    if best_overall is not None:
        print(
            "Best trial seen: "
            f"x={best_overall['x_mm']:.2f}mm, "
            f"y={best_overall['y_mm']:.2f}mm, "
            f"score={best_overall['score']:.3f}"
        )
    print(f"Mean score history: {np.round([h['mean_score'] for h in history], 3)}")
    print(f"Best score history: {np.round([h['best_score'] for h in history], 3)}")
    plot_trials_by_generation(all_trials, output_path="data/es_trials_by_generation.png")
    return theta, history, all_trials



if __name__ == "__main__":
    main()
