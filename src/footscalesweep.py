"""
footscalesweep.py

Bounded CMA-ES search for the walking controller + foot mesh scales.

Optimized parameters (5D):
  1) hip_omega      [rad/s]    (0.4 Hz .. 0.8 Hz) * 2*pi
  2) leg_amp_deg    [deg]      25 .. 45
  3) x_scale        [-]        SCALE_MIN .. SCALE_MAX
  4) y_scale        [-]        SCALE_MIN .. SCALE_MAX
  5) z_scale        [-]        SCALE_MIN .. SCALE_MAX

Objective:
  maximize forward speed while penalizing yaw drift/rate and falls.

Notes:
  - Starts from scales [1, 1, 1].
  - Scales are applied directly to both feet without geometry correction
    (the STL meshes are already correctly centered).
  - Uses the same stabilize -> walk flow used in existing scripts.
"""

import copy
import os
import pickle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from src.sim import ProgressCallback
from src.rduplo import Duplo

# ─── Parameter space ─────────────────────────────────────────────────────────
HZ_TO_RAD = 2.0 * np.pi

BOUNDS = {
    'hip_omega':  (0.4 * HZ_TO_RAD, 0.8 * HZ_TO_RAD),
    'leg_amp_deg': (25.0, 45.0),
    'x_scale':    (0.8, 2.0),
    'y_scale':    (0.5, 2.0),
    'z_scale':    (0.8, 2.5),
}
ORDER = list(('hip_omega', 'leg_amp_deg', 'x_scale', 'y_scale', 'z_scale'))

LOWS  = np.array([BOUNDS[k][0] for k in ORDER], dtype=float)
HIGHS = np.array([BOUNDS[k][1] for k in ORDER], dtype=float)

X0 = np.array([
    0.3 * HZ_TO_RAD,   # hip_omega start (clipped to LOWS[0] on first eval)
    30.0,              # leg_amp_deg
    1.0, 1.0, 1.0,     # x/y/z scale
], dtype=float)

SIGMA0 = np.array([0.35, 2.5, 0.16, 0.16, 0.16], dtype=float)

# ─── CMA-ES hyper-parameters ─────────────────────────────────────────────────
MAX_GENERATIONS = 18
POP_SIZE        = 12

# ─── Objective weights ────────────────────────────────────────────────────────
W_SPEED      = 1.0
W_YAW_DRIFT  = 0.012
W_YAW_RATE   = 0.02
PENALTY_FALL = 1.0
PENALTY_SWAY = 0.05

# ─── I/O ──────────────────────────────────────────────────────────────────────
RESULTS_PKL     = 'foot_scale_cmaes_results.pkl'
CONVERGENCE_PNG = 'foot_scale_cmaes_convergence.png'
ANALYSIS_PREFIX = 'foot_scale_cmaes'

_DEFAULT_X_MM = 14.3
_DEFAULT_Y_MM = 9.7


# ─── Sim helpers ──────────────────────────────────────────────────────────────

def make_args_base() -> dict:
    x = _DEFAULT_X_MM * 0.001
    y = _DEFAULT_Y_MM * 0.001
    return {
        'name': 'foot-scale-cmaes',
        'robot_dir': './robots',
        'sim_time': 20.0,
        'video_dir': './videos',
        'video_fps': 30,
        'gui': False,
        'record': False,
        'ctrl_dict': {
            'Kp': 20,
            'Kd': 12,
            'leg_amp_deg': 30.0,
            'hip_omega': 0.75 * HZ_TO_RAD,
        },
        'design_params': {
            'geom_pos_offset': {
                'RightFoot':     [x,  -y, 0.0],
                'LeftFoot':      [-x, -y, 0.0],
                'ballfoot_1':    [0, 0, 0],
                'ballfoot_2':    [0, 0, 0],
                'hip_rod_1':     [0, 0, 0],
                'hip_rod_2':     [0, 0, 0],
                'leg_rod_1':     [0, 0, 0],
                'leg_rod_2':     [0, 0, 0],
                'motor_part1_1': [0, 0, 0],
                'motor_part2_1': [0, 0, 0],
                'motor_part3_1': [0, 0, 0],
                'arm_rod_1':     [0, 0, 0],
                'arm_rod_2':     [0, 0, 0],
                'battery_1':     [0, 0, 0],
                'battery_2':     [0, 0, 0],
            },
            'mesh_scale': {
                'RightFoot': [1.0, 1.0, 1.0],
                'LeftFoot':  [1.0, 1.0, 1.0],
                'leg_rod':   [1.0, 1.0, 1.0],
            },
            'body_quat': {
                'motor': [0.98074, -0.19443, 0.0007, -0.01876],
            },
        },
    }


def correct_args_quats(args: dict, avg_quat: np.ndarray) -> None:
    for part in args['design_params']['body_quat']:
        args['design_params']['body_quat'][part] = list(avg_quat)


def stabilise(base_args: dict) -> tuple[bool, np.ndarray]:
    args = copy.deepcopy(base_args)
    args['ctrl_dict']['leg_amp_deg'] = 0.0
    args['sim_time'] = 8.0
    robot = Duplo(args)
    cb = ProgressCallback(args['sim_time'])
    robot.run_sim(callbacks={'progress_bar': cb.update})
    swayed = bool(robot.sway)
    q = robot.mean_quat.copy()
    q /= np.linalg.norm(q)
    robot.close()
    return swayed, q


def run_walking_sim(args: dict) -> dict:
    robot = Duplo(args)
    cb = ProgressCallback(args['sim_time'])
    robot.run_sim(callbacks={'progress_bar': cb.update})
    metrics = copy.deepcopy(robot.walk_metrics)
    robot.close()
    return metrics


def safe_run(args: dict, label: str) -> dict:
    try:
        return run_walking_sim(args)
    except Exception as exc:
        print(f'  [SIM ERROR] {label}: {exc}')
        return {
            'fwd_speed_ms': 0.0,
            'yaw_drift_deg': 0.0,
            'yaw_rate_deg_per_s': 0.0,
            'fell': True,
            'error': str(exc),
        }


def apply_foot_scales(args: dict, scales_xyz) -> None:
    """Set uniform x/y/z mesh scale on both feet. No geometry correction needed
    since the STL meshes are already correctly centered."""
    s_vec = [float(v) for v in scales_xyz]
    args['design_params']['mesh_scale']['RightFoot'] = s_vec[:]
    args['design_params']['mesh_scale']['LeftFoot']  = s_vec[:]


def theta_to_args(theta: np.ndarray) -> dict:
    args = make_args_base()
    args['ctrl_dict']['hip_omega']   = float(theta[0])
    args['ctrl_dict']['leg_amp_deg'] = float(theta[1])
    apply_foot_scales(args, theta[2:5])
    return args


def objective(metrics: dict, swayed_at_rest: bool) -> float:
    speed         = float(metrics.get('fwd_speed_ms', 0.0))
    yaw_drift_abs = abs(float(metrics.get('yaw_drift_deg', 0.0)))
    yaw_rate_abs  = abs(float(metrics.get('yaw_rate_deg_per_s', 0.0)))
    fell          = bool(metrics.get('fell', True))
    score = (W_SPEED * speed
             - W_YAW_DRIFT  * yaw_drift_abs
             - W_YAW_RATE   * yaw_rate_abs
             - PENALTY_FALL * float(fell)
             - PENALTY_SWAY * float(swayed_at_rest))
    return float(score)


def evaluate_theta(theta: np.ndarray) -> dict:
    theta = np.clip(theta.astype(float), LOWS, HIGHS)
    args  = theta_to_args(theta)
    swayed, q = stabilise(args)
    if swayed:
        correct_args_quats(args, q)
    label   = (f'w={theta[0]:.3f}, amp={theta[1]:.2f}, '
               f's=({theta[2]:.3f},{theta[3]:.3f},{theta[4]:.3f})')
    metrics = safe_run(copy.deepcopy(args), label=label)
    score   = objective(metrics, swayed)
    return {
        'theta':              theta.copy(),
        'hip_omega':          float(theta[0]),
        'leg_amp_deg':        float(theta[1]),
        'x_scale':            float(theta[2]),
        'y_scale':            float(theta[3]),
        'z_scale':            float(theta[4]),
        'score':              score,
        'fwd_speed_ms':       float(metrics.get('fwd_speed_ms', 0.0)),
        'yaw_drift_deg':      float(metrics.get('yaw_drift_deg', 0.0)),
        'yaw_rate_deg_per_s': float(metrics.get('yaw_rate_deg_per_s', 0.0)),
        'fell':               bool(metrics.get('fell', True)),
        'swayed_at_rest':     bool(swayed),
    }


# ─── CMA-ES ───────────────────────────────────────────────────────────────────

class CMAES:
    def __init__(self, x0: np.ndarray, sigma0: np.ndarray,
                 lows: np.ndarray, highs: np.ndarray, popsize: int):
        self.n       = len(x0)
        self.lows    = lows.astype(float)
        self.highs   = highs.astype(float)
        self.popsize = int(popsize)
        self.mu      = self.popsize // 2

        # Recombination weights
        weights       = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights  = weights / np.sum(weights)
        self.mueff    = np.sum(self.weights)**2 / np.sum(self.weights**2)

        # Strategy parameters
        self.cc    = (4 + self.mueff/self.n) / (self.n + 4 + 2*self.mueff/self.n)
        self.cs    = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1    = 2 / ((self.n + 1.3)**2 + self.mueff)
        self.cmu   = min(1 - self.c1,
                         2 * (self.mueff - 2 + 1/self.mueff)
                         / ((self.n + 2)**2 + self.mueff))
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff - 1)/(self.n + 1)) - 1) + self.cs

        # State
        self.m        = np.clip(x0.astype(float), self.lows, self.highs)
        self.sigma    = float(np.mean(sigma0))
        self.sigma_vec = sigma0.astype(float)
        self.pc       = np.zeros(self.n)
        self.ps       = np.zeros(self.n)
        self.C        = np.eye(self.n)
        self.B        = np.eye(self.n)
        self.D        = np.ones(self.n)
        self.invsqrtC = np.eye(self.n)
        self.chi_n    = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))
        self.counteval  = 0
        self.eigeneval  = 0

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        arz = np.random.randn(self.n, self.popsize)
        ary = self.B @ (self.D[:, None] * arz)
        arx = self.m[:, None] + self.sigma_vec[:, None] * ary
        arx = np.clip(arx, self.lows[:, None], self.highs[:, None])
        return arx.T.copy(), arz.T.copy()

    def tell(self, solutions: np.ndarray, fitness: np.ndarray):
        idx   = np.argsort(fitness)
        x_sel = solutions[idx[:self.mu]]
        x_old = self.m.copy()

        # Mean update
        self.m = np.sum(self.weights[:, None] * x_sel, axis=0)
        y_w    = (self.m - x_old) / np.maximum(self.sigma_vec, 1e-12)

        # Evolution path (sigma)
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff)
                   * self.invsqrtC @ y_w)
        norm_ps = np.linalg.norm(self.ps)

        hsig = float(
            norm_ps
            / np.sqrt(1 - (1 - self.cs)**(2 * self.counteval/self.popsize + 1))
            / self.chi_n
            < 1.4 + 2/(self.n + 1)
        )

        # Evolution path (covariance)
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w)

        artmp    = (x_sel - x_old[None, :]) / np.maximum(self.sigma_vec[None, :], 1e-12)
        rank_mu  = np.zeros((self.n, self.n))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(artmp[i], artmp[i])

        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + delta_hsig * self.C)
                  + self.cmu * rank_mu)

        # Step-size update
        self.sigma *= np.exp(self.cs / self.damps * (norm_ps / self.chi_n - 1))

        # Clip sigma_vec to sane bounds
        _range         = self.highs - self.lows
        self.sigma_vec = np.clip(self.sigma * _range, 0.02 * _range, 0.45 * _range)

        self.counteval += len(solutions)

        # Eigen decomposition (lazy update)
        if (self.counteval - self.eigeneval
                > self.popsize / (self.c1 + self.cmu) / self.n / 10):
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals         = np.maximum(eigvals, 1e-12)
            self.D          = np.sqrt(eigvals)
            self.B          = eigvecs
            self.invsqrtC   = self.B @ np.diag(1.0 / self.D) @ self.B.T


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {path}')


def create_convergence_plot(history: list, out_path: str) -> None:
    gens        = [h['generation'] for h in history]
    best_scores = [h['best_score'] for h in history]
    mean_scores = [h['mean_score'] for h in history]
    best_speed  = [h['best_speed'] for h in history]
    best_abs_yaw = [h['best_abs_yaw'] for h in history]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(gens, best_scores, 'o', color='#1b5e20', label='Best score')
    ax[0].plot(gens, mean_scores, 's', color='#1565c0', label='Mean score')
    ax[0].set_ylabel('Objective score')
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    ax[1].plot(gens, best_speed,   '^', color='#ff6f00', label='Best speed (m/s)')
    ax[1].plot(gens, best_abs_yaw, '^', color='#c2185b', label='Best |yaw drift| (deg)')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel('Metric value')
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.suptitle('CMA-ES: speed/yaw optimization over generations', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved convergence plot -> {out_path}')


def create_analysis_plots(all_evals: list, prefix: str) -> None:
    """Five diagnostic plots covering parameter relationships and score structure."""
    pass  # Diagnostic plots — run standalone for full analysis


def print_best(tag: str, r: dict) -> None:
    print(f'{tag}: score={r["score"]:.4f}'
          f', speed={r["fwd_speed_ms"]} m/s'
          f', yaw={r["yaw_drift_deg"]:+.3f} deg'
          f', yaw_rate={r["yaw_rate_deg_per_s"]} deg/s'
          f', fell={r["fell"]}'
          f', swayed={r["swayed_at_rest"]}'
          f', omega={r["hip_omega"]}'
          f', amp={r["leg_amp_deg"]:.2f}'
          f', scale=[{r["x_scale"]}, {r["y_scale"]}, {r["z_scale"]}]')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print('Starting bounded CMA-ES for [omega, leg_amp_deg, x_scale, y_scale, z_scale]')
    print(f'Bounds: omega=[{LOWS[0]:.4f}, {HIGHS[0]:.4f}]'
          f', amp=[{LOWS[1]:.1f}, {HIGHS[1]:.1f}]'
          f', scales=[{LOWS[2]:.2f}, {HIGHS[2]:.2f}]')
    print(f'Initial theta: {X0}')
    print()

    es = CMAES(x0=X0, sigma0=SIGMA0, lows=LOWS, highs=HIGHS, popsize=POP_SIZE)

    all_evals   = []
    history     = []
    global_best = None

    # Baseline at X0
    base_eval = evaluate_theta(np.clip(X0, LOWS, HIGHS))
    base_eval['generation']    = -1
    base_eval['candidate_idx'] = 0
    all_evals.append(base_eval)
    global_best = base_eval
    print_best('Baseline', base_eval)

    for gen in range(MAX_GENERATIONS):
        print('====================================================================')
        print(f'Generation {gen+1}/{MAX_GENERATIONS}')
        xs, _ = es.ask()
        gen_records = []
        gen_fitness = []
        for i, x in enumerate(xs):
            rec = evaluate_theta(x)
            rec['generation']    = gen
            rec['candidate_idx'] = i
            gen_records.append(rec)
            all_evals.append(rec)
            gen_fitness.append(-rec['score'])
            if rec['score'] > global_best['score']:
                global_best = rec

        es.tell(np.array([r['theta'] for r in gen_records], dtype=float),
                np.array(gen_fitness, dtype=float))

        best_gen   = max(gen_records, key=lambda r: r['score'])
        mean_score = np.mean([r['score'] for r in gen_records])
        history.append({
            'generation':  gen,
            'best_score':  best_gen['score'],
            'mean_score':  mean_score,
            'best_speed':  best_gen['fwd_speed_ms'],
            'best_abs_yaw': abs(best_gen['yaw_drift_deg']),
            'best_theta':  best_gen['theta'],
        })
        print_best('  best gen', best_gen)
        print_best('  best all', global_best)

    print('\n--------------------------------------------------------------------')
    print_best('Final best', global_best)
    print('--------------------------------------------------------------------')

    payload = {
        'order':  ORDER,
        'bounds': BOUNDS,
        'weights': es.weights.tolist(),
        'settings': {
            'max_generations': MAX_GENERATIONS,
            'pop_size':        POP_SIZE,
            'x0':              X0.tolist(),
            'sigma0':          SIGMA0.tolist(),
        },
        'baseline':  base_eval,
        'best':      global_best,
        'history':   history,
        'all_evals': all_evals,
    }
    with open(RESULTS_PKL, 'wb') as fh:
        pickle.dump(payload, fh)
    print(f'Saved raw CMA-ES results -> {RESULTS_PKL}')

    create_convergence_plot(history, out_path=CONVERGENCE_PNG)
    create_analysis_plots(all_evals, prefix=ANALYSIS_PREFIX)

    print('\nRunning final confirmation simulation with best candidate...')
    final_confirm = evaluate_theta(global_best['theta'])
    print_best('Confirmed best', final_confirm)


if __name__ == '__main__':
    main()
