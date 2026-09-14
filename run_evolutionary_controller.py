#!/usr/bin/env python3
"""Evolutionary strategy search for robust bigfoot gait controllers."""

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import test_sim_sweep as sweep

FIXED_FREQ_HZ = 0.524

# Evolvable controller + foot/hip placement offsets (same mapping as test_sim_sweep).
# foot_x capped at ±78 mm so search cannot lock onto extreme knife-edge stances.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "amp_deg": (22.0, 50.0),
    "start_amp_mult": (0.8, 2.5),
    "start_freq_mult": (0.8, 2.8),
    "Kp": (18.0, 90.0),
    "Kd": (1.5, 12.0),
    "foot_x": (-0.078, 0.078),
    # fy > 0 shifts feet/CoM forward (toward Pi cover) and biases reverse heading; keep <= 0.
    "foot_y": (-0.065, 0.0),
}

PARAM_KEYS = list(PARAM_BOUNDS.keys())
CONTROL_KEYS = ("amp_deg", "start_amp_mult", "start_freq_mult", "Kp", "Kd")
PERTURB_KEYS = ("Kp", "Kd")

# Grid 5/5 occupancy island (fx≈−1 mm, fy=−37 mm). Used by --freeze-feet.
OCCUPANCY_ISLAND = {
    "foot_x": -0.0010440627665987783,
    "foot_y": -0.037142857142857144,
}

# Filled in run_evolution when --freeze-feet is set; clip/mutate copy these in.
_FROZEN: dict[str, float] = {}

# Default seed: best full-survival walker from amp_mult high-limits sweep.
DEFAULT_SWEEP_CSV = Path("data/sweeps/amp_mult_sweep_freq0.405_high_limits.csv")

# Seed: v12 best_ever (fit≈284, 100% Kp/Kd robust, ~15 m / ~11 m forward).
SEED_INDIVIDUAL = {
    "amp_deg": 48.940606327751546,
    "start_amp_mult": 1.2840040496029064,
    "start_freq_mult": 1.958686763739469,
    "Kp": 29.207580386062894,
    "Kd": 8.100165117179767,
    "foot_x": 0.005198001880078662,
    "foot_y": -0.02284166878890382,
}

# v12 feet island used when snapping offset grids / freeze defaults.
V12_FEET = {
    "foot_x": 0.005198001880078662,
    "foot_y": -0.02284166878890382,
}


def load_best_from_sweep_csv(csv_path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise ValueError(f"No rows in sweep CSV: {csv_path}")

    def sort_key(row: dict) -> tuple:
        fell = row.get("Fell") in ("True", True, "true")
        survival = float(row.get("Survival_Time_s", 0.0))
        distance = float(row.get("Distance_Traversed", 0.0))
        return (0 if fell else 1, survival, distance)

    best = max(rows, key=sort_key)
    return {
        "amp_deg": float(best["Amplitude_Deg"]),
        "start_amp_mult": float(best["Start_Amp_Mult"]),
        "start_freq_mult": float(best["Start_Freq_Mult"]),
        "Kp": float(best["Kp"]),
        "Kd": float(best["Kd"]),
        "foot_x": float(best.get("Foot_X", best.get("foot_x", 0.0))),
        "foot_y": float(best.get("Foot_Y", best.get("foot_y", 0.0))),
    }


@dataclass
class Individual:
    genes: dict[str, float]
    fitness: float = float("-inf")
    survival_s: float = 0.0
    distance_m: float = 0.0
    forward_m: float = 0.0
    backward_m: float = 0.0
    walked_backward: bool = False
    fell: bool = True
    generation: int = -1
    alternating_steps: int = 0
    min_swing_clearance: float = 0.0
    slip_ratio: float = 0.0
    gait_quality_pass: bool = False
    robust_rate: float = 0.0

    def to_params(self) -> dict:
        params = dict(self.genes)
        params["freq_hz"] = FIXED_FREQ_HZ
        return params

    def as_row(self) -> dict:
        row = {key: self.genes[key] for key in PARAM_KEYS}
        row.update(
            {
                "freq_hz": FIXED_FREQ_HZ,
                "fitness": self.fitness,
                "Survival_Time_s": self.survival_s,
                "Distance_Traversed": self.distance_m,
                "Forward_Progress": self.forward_m,
                "Backward_Progress": self.backward_m,
                "Walked_Backward": self.walked_backward,
                "Fell": self.fell,
                "Generation": self.generation,
                "Alternating_Steps": self.alternating_steps,
                "Min_Swing_Clearance": self.min_swing_clearance,
                "Slip_Ratio": self.slip_ratio,
                "Gait_Quality_Pass": self.gait_quality_pass,
                "Robust_Rate": self.robust_rate,
            }
        )
        return row


@dataclass
class EvolutionState:
    generation: int = 0
    sigma: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    best_ever: Individual | None = None


def clip_genes(genes: dict[str, float]) -> dict[str, float]:
    clipped = {}
    for key, (lo, hi) in PARAM_BOUNDS.items():
        clipped[key] = float(np.clip(genes[key], lo, hi))
    clipped.update(_FROZEN)
    return clipped


def random_genes(rng: np.random.Generator) -> dict[str, float]:
    genes = {
        key: float(rng.uniform(lo, hi))
        for key, (lo, hi) in PARAM_BOUNDS.items()
        if key not in _FROZEN
    }
    genes.update(_FROZEN)
    return genes


def mutate(
    parent: dict[str, float],
    sigma: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    child = {}
    for key in PARAM_KEYS:
        if key in _FROZEN:
            child[key] = _FROZEN[key]
        else:
            child[key] = parent[key] + rng.normal(0.0, sigma[key])
    return clip_genes(child)


def crossover(
    parent_a: dict[str, float],
    parent_b: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    child = {}
    for key in PARAM_KEYS:
        if key in _FROZEN:
            child[key] = _FROZEN[key]
            continue
        alpha = rng.uniform(0.0, 1.0)
        child[key] = alpha * parent_a[key] + (1.0 - alpha) * parent_b[key]
    return clip_genes(child)


def initial_sigmas(fraction: float = 0.12) -> dict[str, float]:
    return {
        key: (hi - lo) * fraction
        for key, (lo, hi) in PARAM_BOUNDS.items()
    }


def compute_fitness(
    result: dict,
    *,
    robust_rate: float = 0.0,
    heading_max_deg: float = 60.0,
) -> float:
    """Survival + yaw-aligned displacement, gated by local Kp/Kd robustness.

    Displacement bonuses only fully pay out when nearby gain perturbations also
    survive; brittle distance peaks get crushed relative to solid walkers.
    Heading above heading_max_deg is taxed so occupancy-island crab-walks
    cannot outscore a straighter gait at the same stance.
    """
    survival = float(result["Survival_Time_s"])
    distance = float(result.get("Distance_Traversed", 0.0))
    forward = float(result.get("Forward_Progress", 0.0))
    backward = float(result.get("Backward_Progress", max(0.0, -forward)))
    walked_backward = bool(result.get("Walked_Backward", forward < -0.01))
    fell = bool(result["Fell"])

    alternating_steps = float(result.get("Alternating_Steps", 0))
    min_clearance = float(result.get("Min_Swing_Clearance", 0.0))
    slip_ratio = float(result.get("Slip_Ratio", 0.0))
    if not np.isfinite(slip_ratio):
        slip_ratio = 10.0
    slip_ratio = float(np.clip(slip_ratio, 0.0, 10.0))
    gait_pass = bool(result.get("Gait_Quality_Pass", False))
    robust_rate = float(np.clip(robust_rate, 0.0, 1.0))

    heading_deg = float(result.get("Heading_Change_Deg", 0.0))
    if not np.isfinite(heading_deg):
        heading_deg = 180.0
    heading_abs = abs(heading_deg)
    heading_ok = heading_abs <= float(heading_max_deg)
    yaw_mult = float(np.clip(np.cos(np.deg2rad(heading_deg)), 0.0, 1.0))
    straightness = (
        float(np.clip(forward / max(distance, 1e-6), 0.0, 1.0)) if forward > 0.0 else 0.0
    )
    forward_walk_mult = yaw_mult * (0.25 + 0.75 * straightness)

    step_scale = float(np.clip(alternating_steps / 10.0, 0.0, 1.0))
    quality = 0.35 + 0.65 * step_scale
    clearance_term = float(np.clip((min_clearance - 0.02) / 0.04, 0.0, 1.0))

    # Robustness gate on meters: 0 probes passed → keep ~25% of travel payout.
    robust_gate = 0.25 + 0.75 * robust_rate

    score = survival
    full_survived = (not fell) and survival >= sweep.FULL_SURVIVAL_S
    if full_survived:
        score += 12.0
        # Explicit payoff for a wide survival basin (beats raw distance farming).
        score += 35.0 * robust_rate

    score += 12.0 * max(0.0, distance) * forward_walk_mult * quality * robust_gate
    score += 8.0 * max(0.0, forward) * yaw_mult * quality * robust_gate

    score += 4.0 * step_scale
    score += 5.0 * clearance_term
    score -= 2.0 * slip_ratio
    if gait_pass:
        score += 8.0

    forward_pos = max(0.0, forward)
    justified_steps = forward_pos / 0.05
    wasted_steps = max(0.0, alternating_steps - justified_steps)
    score -= 0.6 * wasted_steps

    score -= 6.0 * backward
    if walked_backward:
        score -= 4.0
    if fell and survival < 6.0:
        score -= 4.0
    if full_survived and heading_ok:
        score += 10.0 * (1.0 - heading_abs / max(float(heading_max_deg), 1e-6))
    elif full_survived:
        score -= 8.0
    return score


def probe_local_robustness(
    ctx: sweep.SimulationContext,
    genes: dict[str, float],
    perturb_fraction: float = 0.08,
) -> float:
    """Cheap in-loop robustness: Kp± and Kd± one-sided probes (2 extra trials).

    Returns fraction of probes that full-survive. Call only after a nominal
    full-survival trial so we do not triple-cost every faller.
    """
    full = 0
    n = 0
    for key in PERTURB_KEYS:
        lo, hi = PARAM_BOUNDS[key]
        delta = (hi - lo) * perturb_fraction
        for sign in (-1.0, 1.0):
            # One direction per gain to keep cost at 2 trials total.
            if (key == "Kp" and sign < 0.0) or (key == "Kd" and sign > 0.0):
                continue
            n += 1
            params = dict(genes)
            params["freq_hz"] = FIXED_FREQ_HZ
            params[key] = float(np.clip(genes[key] + sign * delta, lo, hi))
            result = sweep.run_single_trial(ctx, params, verbose=False)
            if (not result["Fell"]) and float(result["Survival_Time_s"]) >= sweep.FULL_SURVIVAL_S:
                full += 1
    return full / max(n, 1)


def evaluate_individual(
    ctx: sweep.SimulationContext,
    genes: dict[str, float],
    generation: int,
    *,
    probe_robustness: bool = True,
    perturb_fraction: float = 0.08,
    heading_max_deg: float = 60.0,
) -> Individual:
    params = dict(genes)
    params["freq_hz"] = FIXED_FREQ_HZ
    result = sweep.run_single_trial(ctx, params, verbose=False)
    survival = float(result["Survival_Time_s"])
    fell = bool(result["Fell"])
    full_survived = (not fell) and survival >= sweep.FULL_SURVIVAL_S

    robust_rate = 0.0
    if probe_robustness and full_survived:
        robust_rate = probe_local_robustness(ctx, genes, perturb_fraction=perturb_fraction)

    return Individual(
        genes=genes,
        fitness=compute_fitness(
            result, robust_rate=robust_rate, heading_max_deg=heading_max_deg
        ),
        survival_s=survival,
        distance_m=float(result["Distance_Traversed"]),
        forward_m=float(result["Forward_Progress"]),
        backward_m=float(result["Backward_Progress"]),
        walked_backward=bool(result["Walked_Backward"]),
        fell=fell,
        generation=generation,
        alternating_steps=int(result.get("Alternating_Steps", 0)),
        min_swing_clearance=float(result.get("Min_Swing_Clearance", 0.0)),
        slip_ratio=float(result.get("Slip_Ratio", 0.0)) if np.isfinite(float(result.get("Slip_Ratio", 0.0))) else 10.0,
        gait_quality_pass=bool(result.get("Gait_Quality_Pass", False)),
        robust_rate=robust_rate,
    )


def seed_population(
    population_size: int,
    rng: np.random.Generator,
    sigma: dict[str, float],
    seed_genes: dict[str, float],
) -> list[Individual]:
    population: list[Individual] = []

    seed = clip_genes(seed_genes)
    population.append(Individual(genes=seed))

    for _ in range(max(0, population_size // 4 - 1)):
        population.append(Individual(genes=mutate(seed, sigma, rng)))

    while len(population) < population_size:
        population.append(Individual(genes=random_genes(rng)))

    return population


def select_parents(population: list[Individual], mu: int) -> list[Individual]:
    return sorted(population, key=lambda ind: ind.fitness, reverse=True)[:mu]


# Keep foot-offset mutation from collapsing once a local gait island is found.
OFFSET_KEYS = ("foot_x", "foot_y")
MIN_SIGMA_FRACTION = 0.02
MIN_SIGMA_FRACTION_OFFSETS = 0.06
MAX_SIGMA_FRACTION = 0.40


def rank_select_parents(
    population: list[Individual],
    mu: int,
    rng: np.random.Generator,
) -> list[Individual]:
    """Linear ranking selection — shrinks the effect of huge raw fitness gaps."""
    ranked = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    mu = min(mu, len(ranked))
    if mu <= 0:
        return []
    weights = np.arange(len(ranked), 0, -1, dtype=float)
    chosen: list[Individual] = []
    remaining = list(range(len(ranked)))
    rem_w = weights.copy()
    for _ in range(mu):
        rem_w = rem_w / rem_w.sum()
        pick = int(rng.choice(len(remaining), p=rem_w))
        chosen.append(ranked[remaining[pick]])
        del remaining[pick]
        rem_w = np.delete(rem_w, pick)
    return chosen


def scale_sigmas(sigma: dict[str, float], scale: float) -> dict[str, float]:
    scaled = {}
    for key, value in sigma.items():
        lo, hi = PARAM_BOUNDS[key]
        span = hi - lo
        min_frac = MIN_SIGMA_FRACTION_OFFSETS if key in OFFSET_KEYS else MIN_SIGMA_FRACTION
        min_sigma = span * min_frac
        max_sigma = span * MAX_SIGMA_FRACTION
        scaled[key] = float(np.clip(value * scale, min_sigma, max_sigma))
    return scaled


def genes_on_bound(genes: dict[str, float], tol_frac: float = 0.02) -> list[str]:
    """Return param names sitting near their box bounds (local-trap signal)."""
    hit = []
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key in _FROZEN:
            continue
        span = hi - lo
        if span <= 0:
            continue
        tol = span * tol_frac
        value = genes[key]
        if value <= lo + tol or value >= hi - tol:
            hit.append(key)
    return hit


def adapt_sigmas(
    sigma: dict[str, float],
    success_rate: float,
    target_rate: float = 0.2,
    grow: float = 1.15,
    shrink: float = 0.85,
) -> dict[str, float]:
    updated = {}
    for key, value in sigma.items():
        lo, hi = PARAM_BOUNDS[key]
        span = hi - lo
        min_frac = MIN_SIGMA_FRACTION_OFFSETS if key in OFFSET_KEYS else MIN_SIGMA_FRACTION
        min_sigma = span * min_frac
        max_sigma = span * MAX_SIGMA_FRACTION
        if success_rate > target_rate:
            next_sigma = value * grow
        else:
            next_sigma = value * shrink
        updated[key] = float(np.clip(next_sigma, min_sigma, max_sigma))
    return updated


def summarize_region(top_individuals: list[Individual]) -> dict:
    summary = {"count": len(top_individuals), "parameters": {}}
    for key in PARAM_KEYS:
        values = np.array([ind.genes[key] for ind in top_individuals], dtype=float)
        summary["parameters"][key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    summary["mean_fitness"] = float(np.mean([ind.fitness for ind in top_individuals]))
    summary["mean_survival_s"] = float(np.mean([ind.survival_s for ind in top_individuals]))
    summary["mean_distance_m"] = float(np.mean([ind.distance_m for ind in top_individuals]))
    summary["full_survival_rate"] = float(
        np.mean([1.0 if (not ind.fell and ind.survival_s >= sweep.FULL_SURVIVAL_S) else 0.0 for ind in top_individuals])
    )
    return summary


def evaluate_robustness(
    ctx: sweep.SimulationContext,
    genes: dict[str, float],
    perturb_fraction: float = 0.08,
) -> dict:
    nominal = dict(genes)
    nominal["freq_hz"] = FIXED_FREQ_HZ
    variants = [nominal]

    for key in PERTURB_KEYS:
        lo, hi = PARAM_BOUNDS[key]
        span = hi - lo
        delta = span * perturb_fraction
        low = dict(nominal)
        high = dict(nominal)
        low[key] = float(np.clip(nominal[key] - delta, lo, hi))
        high[key] = float(np.clip(nominal[key] + delta, lo, hi))
        variants.extend([low, high])

    survival_times = []
    full_survivals = 0
    for params in variants:
        result = sweep.run_single_trial(ctx, params, verbose=False)
        survival = float(result["Survival_Time_s"])
        survival_times.append(survival)
        if not result["Fell"] and survival >= sweep.FULL_SURVIVAL_S:
            full_survivals += 1

    arr = np.asarray(survival_times, dtype=float)
    return {
        "perturbation_count": len(variants),
        "full_survival_count": full_survivals,
        "full_survival_rate": full_survivals / len(variants),
        "mean_survival_s": float(np.mean(arr)),
        "min_survival_s": float(np.min(arr)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def controller_stem(genes: dict[str, float]) -> str:
    """Label like es_amp48.94_kp29.21_kd8.10_sam1.28_sfm1.96 (0.00 rounding)."""
    return (
        f"es_amp{genes['amp_deg']:.2f}"
        f"_kp{genes['Kp']:.2f}"
        f"_kd{genes['Kd']:.2f}"
        f"_sam{genes['start_amp_mult']:.2f}"
        f"_sfm{genes['start_freq_mult']:.2f}"
    )


def save_named_controllers(
    individuals: list[Individual],
    out_dir: Path,
    *,
    extra: dict | None = None,
) -> list[Path]:
    """Write one JSON per unique controller label under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for ind in individuals:
        stem = controller_stem(ind.genes)
        if stem in seen:
            continue
        seen.add(stem)
        path = out_dir / f"{stem}.json"
        payload = {
            "label": stem,
            "freq_hz": FIXED_FREQ_HZ,
            "genes": {key: float(ind.genes[key]) for key in PARAM_KEYS},
            "metrics": {
                "fitness": float(ind.fitness),
                "survival_s": float(ind.survival_s),
                "distance_m": float(ind.distance_m),
                "forward_m": float(ind.forward_m),
                "fell": bool(ind.fell),
                "gait_quality_pass": bool(ind.gait_quality_pass),
                "robust_rate": float(ind.robust_rate),
                "generation": int(ind.generation),
            },
        }
        if extra:
            payload.update(extra)
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)
        print(
            f"  saved {path.name}  fit={ind.fitness:.2f} "
            f"fx={ind.genes['foot_x']*1000:+.1f}mm fy={ind.genes['foot_y']*1000:+.1f}mm"
        )
    return written


def plot_offset_history(history_csv: Path, out_png: Path, bins: int = 10) -> None:
    try:
        from plot_es_foot_offsets import load_rows, plot_es_offsets
    except Exception as exc:  # pragma: no cover
        print(f"Offset plot skipped ({exc})")
        return
    if not history_csv.exists():
        print(f"Offset plot skipped — missing {history_csv}")
        return
    rows = [r for r in load_rows(history_csv) if float(r["Survival_Time_s"]) > 0.05]
    if not rows:
        print("Offset plot skipped — no usable history rows")
        return
    plot_es_offsets(rows, out_png, n_bins=max(3, bins))


def run_evolution(args: argparse.Namespace) -> dict:
    global _FROZEN
    rng = np.random.default_rng(args.seed)
    ctx = sweep.load_simulation(args.model_xml)
    _FROZEN = {}
    if args.freeze_feet:
        _FROZEN = {
            "foot_x": float(args.freeze_foot_x),
            "foot_y": float(args.freeze_foot_y),
        }
    state = EvolutionState(sigma=initial_sigmas(args.initial_sigma_fraction))

    if args.sweep_csv is not None:
        seed_genes = load_best_from_sweep_csv(args.sweep_csv)
        print(f"Seeding theta_0 from sweep best: {args.sweep_csv}")
    else:
        seed_genes = dict(SEED_INDIVIDUAL)
        print("Seeding theta_0 from built-in SEED_INDIVIDUAL (v12 best region)")
    seed_genes = clip_genes(seed_genes)

    print(
        "  "
        f"amp={seed_genes['amp_deg']:.2f} sam={seed_genes['start_amp_mult']:.3f} "
        f"sfm={seed_genes['start_freq_mult']:.3f} "
        f"Kp={seed_genes['Kp']:.1f} Kd={seed_genes['Kd']:.1f} "
        f"fx={seed_genes['foot_x']:.3f} fy={seed_genes['foot_y']:.3f}"
    )

    population = seed_population(args.population_size, rng, state.sigma, seed_genes)
    generation_rows: list[dict] = []
    gens_since_best_improve = 0
    wakeup_cooldown = 0

    if _FROZEN:
        feet_msg = (
            f"FROZEN feet fx={_FROZEN['foot_x']*1000:.1f}mm "
            f"fy={_FROZEN['foot_y']*1000:.1f}mm (control knobs only)"
        )
    else:
        feet_msg = f"evolving foot_x{PARAM_BOUNDS['foot_x']} foot_y{PARAM_BOUNDS['foot_y']}"
    print(
        f"ES start: pop={args.population_size}, mu={args.parents}, "
        f"generations={args.generations}, freq={FIXED_FREQ_HZ:.3f} Hz, "
        f"{feet_msg}"
    )
    print(
        f"Exploration: sigma0={args.initial_sigma_fraction:.2f}*span, "
        f"random_inject={args.random_inject_rate:.0%}, "
        f"rank_select=True, local_refine_on_plateau=True, "
        f"robust_probe=True, heading_max={args.heading_max_deg:.0f}deg, "
        f"stagnation_patience={args.stagnation_patience}, "
        f"wakeup_fit_ratio={args.wakeup_fitness_ratio:.2f}"
    )

    for generation in range(args.generations):
        state.generation = generation
        evaluated: list[Individual] = []
        prev_best_ever_fit = (
            float("-inf") if state.best_ever is None else state.best_ever.fitness
        )
        for individual in population:
            evaluated_ind = evaluate_individual(
                ctx,
                individual.genes,
                generation,
                perturb_fraction=args.perturb_fraction,
                heading_max_deg=args.heading_max_deg,
            )
            evaluated.append(evaluated_ind)
            if state.best_ever is None or evaluated_ind.fitness > state.best_ever.fitness:
                state.best_ever = evaluated_ind

        evaluated.sort(key=lambda ind: ind.fitness, reverse=True)
        # Rank selection softens the cliff between elite (~120) and explorers (~30).
        parents = rank_select_parents(evaluated, args.parents, rng)
        best = evaluated[0]
        full_survivors = sum(1 for ind in evaluated if not ind.fell and ind.survival_s >= sweep.FULL_SURVIVAL_S)

        if state.best_ever is not None and state.best_ever.fitness > prev_best_ever_fit + 1e-9:
            gens_since_best_improve = 0
            wakeup_cooldown = 0
        else:
            gens_since_best_improve += 1

        bound_hits = genes_on_bound(best.genes)
        ever_fit = state.best_ever.fitness if state.best_ever is not None else best.fitness
        # True collapse: population champion much worse than archived best_ever.
        collapsed = best.fitness < args.wakeup_fitness_ratio * ever_fit
        # Plateau: elite is present but we are not improving — need local refine, not spam.
        plateau = (
            gens_since_best_improve >= args.stagnation_patience
            and best.fitness >= args.wakeup_fitness_ratio * ever_fit
        )
        edge_trap = len(bound_hits) >= 2 and best.fitness < 0.5 * ever_fit
        wakeup = False
        plateau_refine = False
        if wakeup_cooldown > 0:
            wakeup_cooldown -= 1
        elif collapsed or edge_trap:
            wakeup = True
            wakeup_cooldown = args.stagnation_patience
        elif plateau:
            plateau_refine = True
            wakeup_cooldown = max(3, args.stagnation_patience // 2)

        gen_summary = {
            "generation": generation,
            "best_fitness": best.fitness,
            "best_survival_s": best.survival_s,
            "best_distance_m": best.distance_m,
            "best_forward_m": best.forward_m,
            "best_backward_m": best.backward_m,
            "best_walked_backward": best.walked_backward,
            "best_fell": best.fell,
            "population_full_survivors": full_survivors,
            "population_walked_backward": sum(1 for ind in evaluated if ind.walked_backward),
            "mean_fitness": float(np.mean([ind.fitness for ind in evaluated])),
            "best_robust_rate": best.robust_rate,
            "mean_robust_rate": float(np.mean([ind.robust_rate for ind in evaluated])),
            "gens_since_best_improve": gens_since_best_improve,
            "wakeup": wakeup,
            "plateau_refine": plateau_refine,
            **{f"sigma_{key}": state.sigma[key] for key in PARAM_KEYS},
            **{f"best_{key}": best.genes[key] for key in PARAM_KEYS},
        }
        state.history.append(gen_summary)
        generation_rows.extend(ind.as_row() for ind in evaluated)
        write_csv(args.history_csv, generation_rows)

        print(
            f"Gen {generation:03d} | best_fit={best.fitness:.2f} "
            f"survival={best.survival_s:.2f}s dist={best.distance_m:.3f}m "
            f"fwd={best.forward_m:.3f}m back={best.backward_m:.3f}m "
            f"backwards={best.walked_backward} fell={best.fell} | "
            f"pop_full_surv={full_survivors}/{len(evaluated)} | "
            f"amp={best.genes['amp_deg']:.1f} sam={best.genes['start_amp_mult']:.2f} "
            f"sfm={best.genes['start_freq_mult']:.2f} "
            f"Kp={best.genes['Kp']:.1f} Kd={best.genes['Kd']:.1f} "
            f"fx={best.genes['foot_x']:.3f} fy={best.genes['foot_y']:.3f} | "
            f"steps={best.alternating_steps} clr={best.min_swing_clearance:.3f} "
            f"slip={best.slip_ratio:.2f} gait_pass={best.gait_quality_pass} "
            f"robust={best.robust_rate:.0%}"
        )

        if generation == args.generations - 1:
            break

        assert state.best_ever is not None
        elite = dict(state.best_ever.genes)

        if wakeup:
            state.sigma = initial_sigmas(min(args.initial_sigma_fraction * 1.35, 0.35))
            print(
                f"  WAKEUP(collapse): stagn={gens_since_best_improve} "
                f"gen_best={best.fitness:.1f} best_ever={ever_fit:.1f} "
                f"bounds={bound_hits or '-'} → hard reseed"
            )
        elif plateau_refine:
            print(
                f"  PLATEAU: stagn={gens_since_best_improve} best_ever={ever_fit:.1f} "
                f"→ local refine around elite + protected explorers"
            )

        offspring: list[Individual] = []
        parent_best_fitness = best.fitness

        # Single elite copy (enough to not forget; not enough to freeze the whole pop).
        offspring.append(Individual(genes=elite))

        if wakeup:
            # Lost the peak: reintroduce seed + elite mutants + heavy random.
            offspring.append(Individual(genes=dict(seed_genes)))
            for i in range(max(6, args.population_size // 3)):
                base = elite if (i % 2 == 0) else seed_genes
                offspring.append(Individual(genes=mutate(base, state.sigma, rng)))
            inject_rate = max(args.random_inject_rate, 0.40)
        elif plateau_refine:
            # Score gap is huge — hill-climb with SMALL steps; explorers are quarantined
            # in their own quota so they don't need to beat 120 to stay alive one gen.
            local_sigma = scale_sigmas(state.sigma, 0.22)
            wide_sigma = scale_sigmas(state.sigma, 1.15)
            n_local = max(8, args.population_size // 2)
            n_wide = max(4, args.population_size // 5)
            for _ in range(n_local):
                offspring.append(Individual(genes=mutate(elite, local_sigma, rng)))
            for _ in range(n_wide):
                offspring.append(Individual(genes=mutate(elite, wide_sigma, rng)))
            inject_rate = max(args.random_inject_rate, 0.30)
        else:
            inject_rate = args.random_inject_rate

        n_random = int(round(args.population_size * max(0.0, inject_rate)))
        n_random = min(n_random, max(0, args.population_size - len(offspring)))
        for _ in range(n_random):
            offspring.append(Individual(genes=random_genes(rng)))

        while len(offspring) < args.population_size:
            parent = parents[rng.integers(0, len(parents))]
            if rng.random() < args.crossover_rate:
                other = parents[rng.integers(0, len(parents))]
                child_genes = crossover(parent.genes, other.genes, rng)
            else:
                child_genes = mutate(parent.genes, state.sigma, rng)
            offspring.append(Individual(genes=child_genes))

        offspring = offspring[: args.population_size]
        child_evaluated = [
            evaluate_individual(
                ctx,
                ind.genes,
                generation + 1,
                perturb_fraction=args.perturb_fraction,
                heading_max_deg=args.heading_max_deg,
            )
            for ind in offspring
        ]
        improvements = sum(1 for child in child_evaluated if child.fitness > parent_best_fitness)
        success_rate = improvements / max(len(child_evaluated), 1)
        if not wakeup and not plateau_refine:
            state.sigma = adapt_sigmas(state.sigma, success_rate)
        population = child_evaluated

    assert state.best_ever is not None
    top_k = sorted(evaluated, key=lambda ind: ind.fitness, reverse=True)[: args.top_k_region]
    region = summarize_region(top_k)

    print("\nEvaluating robustness on top candidates...")
    robust_rows = []
    for rank, individual in enumerate(top_k[: min(5, len(top_k))], 1):
        robustness = evaluate_robustness(ctx, individual.genes, perturb_fraction=args.perturb_fraction)
        robustness["rank"] = rank
        robustness["fitness"] = float(individual.fitness)
        robustness["survival_s"] = float(individual.survival_s)
        robustness["distance_m"] = float(individual.distance_m)
        robustness["genes"] = {key: float(individual.genes[key]) for key in PARAM_KEYS}
        robust_rows.append(robustness)
        print(
            f"  #{rank}: fit={individual.fitness:.2f} survival={individual.survival_s:.2f}s "
            f"dist={individual.distance_m:.3f}m | robust_rate={robustness['full_survival_rate']:.2f} "
            f"mean_pert_surv={robustness['mean_survival_s']:.2f}s"
        )

    best_robust = max(
        robust_rows,
        key=lambda row: row["full_survival_rate"] * row["mean_survival_s"] * max(row["distance_m"], 0.1),
    )

    payload = {
        "fixed_frequency_hz": FIXED_FREQ_HZ,
        "evolving_foot_offsets": not bool(_FROZEN),
        "frozen_offsets": dict(_FROZEN) if _FROZEN else None,
        "heading_max_deg": args.heading_max_deg,
        "seed_theta_0": seed_genes,
        "param_bounds": PARAM_BOUNDS,
        "generations": args.generations,
        "population_size": args.population_size,
        "parents": args.parents,
        "seed": args.seed,
        "best_ever": state.best_ever.as_row(),
        "promising_region_top_k": region,
        "generation_history": state.history,
        "robustness_evaluations": robust_rows,
        "recommended_working_region": {
            "center": {key: region["parameters"][key]["mean"] for key in PARAM_KEYS},
            "spread": {key: region["parameters"][key]["std"] for key in PARAM_KEYS},
            "best_robust_genes": best_robust["genes"],
        },
    }

    write_csv(args.history_csv, generation_rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(payload, indent=2))

    # Archive unique top controllers + best_ever / best_robust with fixed naming.
    best_dir = args.summary_json.parent / f"{args.summary_json.stem}_best"
    print(f"\nSaving named controllers to {best_dir}/")
    archive: list[Individual] = []
    if state.best_ever is not None:
        archive.append(state.best_ever)
    for row in robust_rows:
        genes = row["genes"]
        archive.append(
            Individual(
                genes=genes,
                fitness=float(row["fitness"]),
                survival_s=float(row["survival_s"]),
                distance_m=float(row["distance_m"]),
                fell=False,
                generation=-1,
                robust_rate=float(row["full_survival_rate"]),
            )
        )
    archive.extend(top_k)
    # Prefer higher fitness when labels collide.
    archive.sort(key=lambda ind: ind.fitness, reverse=True)
    named_paths = save_named_controllers(
        archive,
        best_dir,
        extra={"source_summary": str(args.summary_json)},
    )
    payload["named_controllers"] = [str(p) for p in named_paths]
    args.summary_json.write_text(json.dumps(payload, indent=2))

    offset_png = args.summary_json.with_name(args.summary_json.stem.replace("summary", "offsets") + ".png")
    if "offsets" not in offset_png.name:
        offset_png = args.summary_json.parent / f"{args.summary_json.stem}_offsets.png"
    print(f"\nPlotting foot-offset exploration → {offset_png}")
    plot_offset_history(args.history_csv, offset_png, bins=10)

    print("\n=== ES complete ===")
    print(f"Best ever fitness: {state.best_ever.fitness:.2f}")
    print(
        f"Best ever: survival={state.best_ever.survival_s:.2f}s "
        f"dist={state.best_ever.distance_m:.3f}m fell={state.best_ever.fell}"
    )
    print(f"Best ever label: {controller_stem(state.best_ever.genes)}")
    print("Promising region (top-k mean ± std):")
    for key in PARAM_KEYS:
        stats = region["parameters"][key]
        print(f"  {key:16s}: {stats['mean']:.4f} ± {stats['std']:.4f}  [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"\nSaved history to {args.history_csv}")
    print(f"Saved summary to {args.summary_json}")
    print(f"Saved named controllers to {best_dir}")
    print(f"Saved offset plot to {offset_png}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=35)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--top-k-region", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crossover-rate", type=float, default=0.35)
    parser.add_argument(
        "--initial-sigma-fraction",
        type=float,
        default=0.20,
        help="Initial mutation sigma as a fraction of each parameter's bound span.",
    )
    parser.add_argument(
        "--random-inject-rate",
        type=float,
        default=0.20,
        help="Fraction of each generation drawn uniformly from PARAM_BOUNDS (exploration).",
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=8,
        help="Gens without beating best_ever before a WAKEUP reseed/boost.",
    )
    parser.add_argument(
        "--wakeup-fitness-ratio",
        type=float,
        default=0.55,
        help="If gen best_fit < ratio * best_ever for long enough, trigger WAKEUP.",
    )
    parser.add_argument("--perturb-fraction", type=float, default=0.08)
    parser.add_argument(
        "--heading-max-deg",
        type=float,
        default=60.0,
        help="Full-survival walks with |heading| above this are taxed (crab-walk penalty).",
    )
    parser.add_argument(
        "--freeze-feet",
        action="store_true",
        help="Evolve only amp/sam/sfm/Kp/Kd; pin foot_x/foot_y to --freeze-foot-*.",
    )
    parser.add_argument(
        "--freeze-foot-x",
        type=float,
        default=V12_FEET["foot_x"],
        help="Pinned foot_x when --freeze-feet (default: v12 best feet, +5.2 mm).",
    )
    parser.add_argument(
        "--freeze-foot-y",
        type=float,
        default=V12_FEET["foot_y"],
        help="Pinned foot_y when --freeze-feet (default: v12 best feet, -22.8 mm).",
    )
    parser.add_argument("--model-xml", default=os.environ.get("MODEL_XML_PATH", "Bigfoot/scene.xml"))
    parser.add_argument(
        "--sweep-csv",
        type=Path,
        default=None,
        help="Optional: load theta_0 from the best row in this sweep CSV.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("data/es/bigfoot_es_history.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/es/bigfoot_es_summary.json"),
    )
    args = parser.parse_args()
    if args.sweep_csv is not None and str(args.sweep_csv).strip() == "":
        args.sweep_csv = None
    if args.freeze_feet:
        if args.history_csv == Path("data/es/bigfoot_es_history.csv"):
            args.history_csv = Path("data/es/bigfoot_es_history_inner.csv")
        if args.summary_json == Path("data/es/bigfoot_es_summary.json"):
            args.summary_json = Path("data/es/bigfoot_es_summary_inner.json")
    run_evolution(args)


if __name__ == "__main__":
    main()
