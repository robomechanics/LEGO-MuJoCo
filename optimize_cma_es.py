#!/usr/bin/env python3
"""CMA-ES search over foot geometry and actuation parameters.

This script is meant as a local optimizer that sits on top of the current
geometry generator and simulation stack:
    - geometry XML generation uses `run_sweep.generate_modified_xml`
    - rollouts use `test_sim_sweep.run_single_trial`
    - objective defaults to mean `Walk_Score`

Example:
    python3 optimize_cma_es.py \
      --generations 20 \
      --population-size 12 \
      --max-workers 12 \
      --trials-per-eval 1 \
      --objective walk_score \
      --active-params curve_x curve_y box_x box_y Kp Kd amp_deg freq_hz \
      --results-csv data/cma_es/cma_es_results.csv \
      --best-json data/cma_es/cma_es_best.json

Dependency:
    pip install cma
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cma
except ImportError:  # pragma: no cover - runtime dependency check
    cma = None

import run_sweep as sweep_runner
import test_sim_sweep as sim_sweep
import sweep_config


GEOMETRY_PARAM_NAMES = {"curve_x", "curve_y", "box_x", "box_y"}
DEFAULT_ACTIVE_PARAMS = (
    "curve_x",
    "curve_y",
    "box_x",
    "box_y",
    "Kp",
    "Kd",
    "start_amp_mult",
    "start_freq_mult",
    "amp_deg",
    "freq_hz",
    "ramp_time",
)

RESULT_METRICS = (
    "Distance_Traversed",
    "Forward_Progress",
    "Lateral_Drift",
    "Instantaneous_Forward_Progress",
    "Instantaneous_Forward_Absolute",
    "Instantaneous_Lateral_Progress",
    "Path_Length",
    "Heading_Change_Deg",
    "Right_Landings",
    "Left_Landings",
    "Landing_Count",
    "Alternating_Steps",
    "Right_Max_Swing_Clearance",
    "Left_Max_Swing_Clearance",
    "Min_Swing_Clearance",
    "Right_Contact_Slip",
    "Left_Contact_Slip",
    "Total_Contact_Slip",
    "Slip_Ratio",
    "Walk_Score",
    "CoT",
)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lower: float
    upper: float
    initial: float


@dataclass(frozen=True)
class WorkerConfig:
    specs: tuple[ParamSpec, ...]
    trials_per_eval: int
    objective: str
    static_model_xml: str | None


def default_results_csv() -> Path:
    return Path("data/cma_es/cma_es_results.csv")


def default_best_json() -> Path:
    return Path("data/cma_es/cma_es_best.json")


def _distribution_bound(name: str, fallback_low: float, fallback_high: float, fallback_mean: float) -> tuple[float, float, float]:
    spec = sweep_config.NORMAL_TRIAL_DISTRIBUTIONS.get(name)
    if spec is None:
        return fallback_low, fallback_high, fallback_mean
    return (
        float(spec.get("min", fallback_low)),
        float(spec.get("max", fallback_high)),
        float(spec.get("mean", fallback_mean)),
    )


def build_param_library() -> dict[str, ParamSpec]:
    fixed = dict(sweep_config.FIXED_TRIAL_PARAMS)
    geometry_base = dict(sweep_config.GEOMETRY_BASE)

    kp_low, kp_high, kp_mean = _distribution_bound("Kp", 20.0, 45.0, float(fixed.get("Kp", 45.0)))
    kd_low, kd_high, kd_mean = _distribution_bound("Kd", 2.0, 15.0, float(fixed.get("Kd", 7.0)))
    sam_low, sam_high, sam_mean = _distribution_bound("start_amp_mult", 0.8, 1.8, float(fixed.get("start_amp_mult", 1.2)))
    sfm_low, sfm_high, sfm_mean = _distribution_bound("start_freq_mult", 0.7, 1.4, float(fixed.get("start_freq_mult", 0.9)))
    ramp_low, ramp_high, ramp_mean = _distribution_bound("ramp_time", 0.0, 3.0, float(fixed.get("ramp_time", 0.0)))

    library = {
        "curve_x": ParamSpec(
            "curve_x",
            lower=float(min(sweep_config.SWEEP_VALUES["curve_x"])),
            upper=float(max(sweep_config.SWEEP_VALUES["curve_x"])),
            initial=float(geometry_base["curve_x"]),
        ),
        "curve_y": ParamSpec(
            "curve_y",
            lower=float(min(sweep_config.SWEEP_VALUES["curve_y"])),
            upper=float(max(sweep_config.SWEEP_VALUES["curve_y"])),
            initial=float(geometry_base["curve_y"]),
        ),
        "box_x": ParamSpec(
            "box_x",
            lower=float(min(sweep_config.SWEEP_VALUES["box_x"])),
            upper=float(max(sweep_config.SWEEP_VALUES["box_x"])),
            initial=float(geometry_base["box_x"]),
        ),
        "box_y": ParamSpec(
            "box_y",
            lower=float(min(sweep_config.SWEEP_VALUES["box_y"])),
            upper=float(max(sweep_config.SWEEP_VALUES["box_y"])),
            initial=float(geometry_base["box_y"]),
        ),
        "Kp": ParamSpec("Kp", kp_low, kp_high, kp_mean),
        "Kd": ParamSpec("Kd", kd_low, kd_high, kd_mean),
        "start_amp_mult": ParamSpec("start_amp_mult", sam_low, sam_high, sam_mean),
        "start_freq_mult": ParamSpec("start_freq_mult", sfm_low, sfm_high, sfm_mean),
        "amp_deg": ParamSpec("amp_deg", 10.0, 60.0, float(fixed.get("amp_deg", 35.0))),
        "freq_hz": ParamSpec("freq_hz", 0.2, 1.2, float(fixed.get("freq_hz", 0.50))),
        "ramp_time": ParamSpec("ramp_time", ramp_low, ramp_high, ramp_mean),
        "foot_x": ParamSpec("foot_x", -0.08, 0.08, float(fixed.get("foot_x", 0.0))),
        "foot_y": ParamSpec("foot_y", -0.08, 0.08, float(fixed.get("foot_y", 0.0))),
        "torque_limit": ParamSpec("torque_limit", 5.0, 25.0, float(fixed.get("torque_limit", 25.0))),
    }
    return library


def resolve_active_specs(active_names: list[str]) -> tuple[ParamSpec, ...]:
    library = build_param_library()
    missing = [name for name in active_names if name not in library]
    if missing:
        raise ValueError(f"Unsupported active parameter(s): {', '.join(sorted(missing))}")
    return tuple(library[name] for name in active_names)


def decode_vector(vector: np.ndarray, specs: tuple[ParamSpec, ...]) -> tuple[dict[str, float], dict[str, float]]:
    geometry = sweep_runner.resolve_geometry()
    trial_params = dict(sweep_config.FIXED_TRIAL_PARAMS)
    for value, spec in zip(vector, specs):
        clipped = float(np.clip(value, spec.lower, spec.upper))
        if spec.name in GEOMETRY_PARAM_NAMES:
            geometry[spec.name] = clipped
        else:
            trial_params[spec.name] = clipped
    return geometry, trial_params


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in RESULT_METRICS:
        values = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
        metrics[metric] = float(np.mean(values)) if values else float("nan")
    metrics["Pass_Rate"] = float(np.mean([1.0 if row["Gait_Quality_Pass"] else 0.0 for row in rows]))
    metrics["Fall_Rate"] = float(np.mean([1.0 if row["Fell"] else 0.0 for row in rows]))
    return metrics


def compute_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "walk_score":
        return metrics["Walk_Score"]
    if objective == "distance":
        return metrics["Distance_Traversed"]
    if objective == "forward_progress":
        return metrics["Forward_Progress"]
    if objective == "pass_rate":
        return metrics["Pass_Rate"]
    if objective == "composite":
        cot = metrics["CoT"]
        cot_penalty = min(cot, 10.0) if math.isfinite(cot) else 10.0
        return (
            1.0 * metrics["Walk_Score"]
            + 1.5 * metrics["Pass_Rate"]
            + 0.5 * metrics["Forward_Progress"]
            - 0.25 * metrics["Lateral_Drift"]
            - 0.10 * metrics["Heading_Change_Deg"]
            - 0.20 * cot_penalty
        )
    raise ValueError(f"Unsupported objective: {objective}")


def evaluate_vector(vector: np.ndarray, worker_config: WorkerConfig) -> dict[str, Any]:
    geometry, trial_params = decode_vector(vector, worker_config.specs)

    if worker_config.static_model_xml is None:
        with tempfile.TemporaryDirectory(prefix="cma_es_eval_") as temp_dir:
            temp_path = Path(temp_dir)
            output_xml = temp_path / "modified_model.xml"
            mesh_out_dir = temp_path / "foot_section_out"
            sweep_runner.generate_modified_xml(geometry, output_xml, mesh_out_dir)
            rows = run_trials(output_xml, trial_params, worker_config.trials_per_eval)
    else:
        rows = run_trials(Path(worker_config.static_model_xml), trial_params, worker_config.trials_per_eval)

    metrics = mean_metrics(rows)
    score = compute_score(metrics, worker_config.objective)
    loss = -score

    result = {
        "loss": loss,
        "score": score,
        "geometry": geometry,
        "trial_params": trial_params,
        "metrics": metrics,
    }
    return result


def run_trials(model_xml_path: Path, trial_params: dict[str, float], trials_per_eval: int) -> list[dict[str, Any]]:
    ctx = sim_sweep.load_simulation(model_xml_path)
    rows = []
    for _ in range(trials_per_eval):
        row = sim_sweep.run_single_trial(ctx, trial_params, verbose=False)
        rows.append(row)
    return rows


def flatten_result_record(
    *,
    generation: int,
    candidate_index: int,
    vector: np.ndarray,
    worker_config: WorkerConfig,
    result: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "Generation": generation,
        "Candidate_Index": candidate_index,
        "Objective": worker_config.objective,
        "Loss": result["loss"],
        "Score": result["score"],
    }
    for spec, value in zip(worker_config.specs, vector):
        record[f"Opt_{spec.name}"] = float(np.clip(value, spec.lower, spec.upper))
    for key, value in result["geometry"].items():
        record[f"Geometry_{key}"] = value
    for key, value in result["trial_params"].items():
        record[f"Trial_{key}"] = value
    for key, value in result["metrics"].items():
        record[f"Metric_{key}"] = value
    return record


def write_records(csv_path: Path, records: list[dict[str, Any]], append: bool) -> None:
    if not records:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not csv_path.exists() or csv_path.stat().st_size == 0
    mode = "a" if append else "w"
    with csv_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def make_static_model_if_possible(specs: tuple[ParamSpec, ...]) -> tuple[tempfile.TemporaryDirectory[str] | None, str | None]:
    if any(spec.name in GEOMETRY_PARAM_NAMES for spec in specs):
        return None, None

    temp_dir = tempfile.TemporaryDirectory(prefix="cma_es_static_geometry_")
    temp_path = Path(temp_dir.name)
    output_xml = temp_path / "modified_model.xml"
    mesh_out_dir = temp_path / "foot_section_out"
    sweep_runner.generate_modified_xml(sweep_runner.resolve_geometry(), output_xml, mesh_out_dir)
    return temp_dir, str(output_xml)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-params",
        nargs="+",
        default=list(DEFAULT_ACTIVE_PARAMS),
        help="Parameters to optimize. Supports geometry and actuation parameters.",
    )
    parser.add_argument(
        "--objective",
        choices=("walk_score", "distance", "forward_progress", "pass_rate", "composite"),
        default="walk_score",
        help="Objective to maximize. `walk_score` is the default local-search target.",
    )
    parser.add_argument("--generations", type=int, default=20, help="Number of CMA-ES generations.")
    parser.add_argument("--population-size", type=int, default=12, help="Population size per generation.")
    parser.add_argument(
        "--trials-per-eval",
        type=int,
        default=1,
        help="Number of rollouts averaged for each candidate. Useful if you later add noise.",
    )
    parser.add_argument(
        "--sigma-frac",
        type=float,
        default=0.15,
        help="Initial CMA-ES sigma as a fraction of the mean parameter range.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional CMA-ES seed. Omit for non-deterministic search.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel workers for evaluating each generation. Defaults to min(population, cpu_count).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=default_results_csv(),
        help="CSV path for logging every candidate evaluation.",
    )
    parser.add_argument(
        "--best-json",
        type=Path,
        default=default_best_json(),
        help="JSON path for the best candidate summary.",
    )
    return parser.parse_args()


def main() -> None:
    if cma is None:
        raise SystemExit(
            "Missing dependency `cma`. Install it in the active environment with:\n"
            "  pip install cma"
        )

    args = parse_args()
    specs = resolve_active_specs(args.active_params)
    initial = np.array([spec.initial for spec in specs], dtype=float)
    mean_range = float(np.mean([spec.upper - spec.lower for spec in specs]))
    sigma0 = max(1e-6, args.sigma_frac * mean_range)

    static_model_dir, static_model_xml = make_static_model_if_possible(specs)
    worker_config = WorkerConfig(
        specs=specs,
        trials_per_eval=max(1, int(args.trials_per_eval)),
        objective=args.objective,
        static_model_xml=static_model_xml,
    )

    options = {
        "bounds": [[spec.lower for spec in specs], [spec.upper for spec in specs]],
        "popsize": int(args.population_size),
        "verbose": -9,
    }
    if args.seed is not None:
        options["seed"] = int(args.seed)

    max_workers = args.max_workers
    if max_workers is None:
        max_workers = min(int(args.population_size), os.cpu_count() or 1)
    max_workers = max(1, int(max_workers))

    print(
        f"Starting CMA-ES with {len(specs)} parameter(s), "
        f"population={args.population_size}, generations={args.generations}, "
        f"trials_per_eval={worker_config.trials_per_eval}, workers={max_workers}, "
        f"objective={args.objective}"
    )
    print("Active parameters:", ", ".join(spec.name for spec in specs))

    es = cma.CMAEvolutionStrategy(initial.tolist(), sigma0, options)
    best_record: dict[str, Any] | None = None

    try:
        args.results_csv.parent.mkdir(parents=True, exist_ok=True)
        if args.results_csv.exists():
            args.results_csv.unlink()

        for generation in range(1, int(args.generations) + 1):
            vectors = [np.asarray(candidate, dtype=float) for candidate in es.ask()]
            if max_workers == 1:
                results = [evaluate_vector(vector, worker_config) for vector in vectors]
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(evaluate_vector, vectors, [worker_config] * len(vectors)))

            losses = [result["loss"] for result in results]
            es.tell([vector.tolist() for vector in vectors], losses)

            records = [
                flatten_result_record(
                    generation=generation,
                    candidate_index=index,
                    vector=vector,
                    worker_config=worker_config,
                    result=result,
                )
                for index, (vector, result) in enumerate(zip(vectors, results), start=1)
            ]
            write_records(args.results_csv, records, append=True)

            generation_best = max(results, key=lambda item: item["score"])
            if best_record is None or generation_best["score"] > best_record["score"]:
                best_record = generation_best

            print(
                f"Generation {generation:>3}: "
                f"best_score={generation_best['score']:.4f} "
                f"best_dist={generation_best['metrics']['Distance_Traversed']:.4f} "
                f"best_forward={generation_best['metrics']['Forward_Progress']:.4f} "
                f"pass_rate={generation_best['metrics']['Pass_Rate']:.2f}"
            )

        if best_record is None:
            raise RuntimeError("CMA-ES finished without any evaluated candidates.")

        args.best_json.parent.mkdir(parents=True, exist_ok=True)
        with args.best_json.open("w") as handle:
            json.dump(best_record, handle, indent=2, sort_keys=True)

        print(f"Wrote evaluation log to {args.results_csv}")
        print(f"Wrote best candidate summary to {args.best_json}")
        print("Best geometry:", json.dumps(best_record["geometry"], sort_keys=True))
        print("Best trial params:", json.dumps(best_record["trial_params"], sort_keys=True))
        print("Best metrics:", json.dumps(best_record["metrics"], sort_keys=True))
    finally:
        if static_model_dir is not None:
            static_model_dir.cleanup()


if __name__ == "__main__":
    main()
