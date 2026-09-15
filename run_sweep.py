#!/usr/bin/env python3
"""Run a geometry sweep over any configured set of foot parameters."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import itertools
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco.viewer

import gen_new_xml_v3 as mesh_gen
import plot_sweep_results
import test_sim_sweep as sim_sweep
import sweep_config as config

ROOT_DIR = Path(__file__).resolve().parent
mesh_gen.VERBOSE = False
PREVIEW_FIRST_TRIAL = bool(getattr(config, "PREVIEW_FIRST_TRIAL", False))
PREVIEW_TRIAL_DURATION = float(getattr(config, "PREVIEW_TRIAL_DURATION", 20.0))
CACHE_LOCK_TIMEOUT_S = 15.0
CACHE_LOCK_POLL_S = 0.25
CACHE_LOCK_WARN_AFTER_S = 3.0


def vprint(*args, **kwargs) -> None:
    if config.RUNNER_VERBOSE:
        print(*args, **kwargs)


def resolve_geometry(overrides: dict[str, float] | None = None) -> dict[str, float]:
    geometry = dict(config.GEOMETRY_BASE)
    if overrides:
        geometry.update(overrides)
    return geometry


def generate_modified_xml(geometry: dict[str, float], output_xml: Path, out_dir: Path) -> None:
    sections = mesh_gen.generate_feet(
        scad_file=Path(mesh_gen.DEFAULT_SCAD_FILE),
        out_dir=out_dir,
        curve_x=geometry["curve_x"],
        curve_y=geometry["curve_y"],
        curve_z=geometry.get("curve_z", mesh_gen.DEFAULT_CURVE_Z),
        box_x=geometry["box_x"],
        box_y=geometry["box_y"],
        box_z=geometry.get("box_z", mesh_gen.DEFAULT_BOX_Z),
        fn=mesh_gen.DEFAULT_FN,
    )

    mesh_gen.inject_feet_preserving_dynamics(
        entry_xml=Path(mesh_gen.DEFAULT_ENTRY_XML),
        sections=sections,
        output_xml=output_xml,
    )


def cached_modified_xml(geometry: dict[str, float], cache_root: Path) -> Path:
    """Reuse each mesh shape across offset/actuation trials, including workers."""
    shape = {
        key: geometry.get(key)
        for key in ("curve_x", "curve_y", "curve_z", "box_x", "box_y", "box_z")
    }
    key = hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:20]
    directory = cache_root / key
    directory.mkdir(parents=True, exist_ok=True)
    output_xml = directory / "modified_model.xml"
    with (directory / "build.lock").open("w") as lock:
        start = time.monotonic()
        warned = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                elapsed = time.monotonic() - start
                if not warned and elapsed >= CACHE_LOCK_WARN_AFTER_S:
                    print(
                        f"Waiting for geometry cache lock {directory / 'build.lock'} "
                        f"({elapsed:.1f}s elapsed)...",
                        flush=True,
                    )
                    warned = True
                if elapsed >= CACHE_LOCK_TIMEOUT_S:
                    raise TimeoutError(
                        f"Timed out after {elapsed:.1f}s waiting for geometry cache lock "
                        f"{directory / 'build.lock'}. Another run_sweep process may still be "
                        "generating this mesh, or a previous run may have been interrupted "
                        "while holding the lock."
                    ) from exc
                time.sleep(CACHE_LOCK_POLL_S)
        if not (directory / "ready").exists():
            generate_modified_xml(geometry, output_xml, directory / "feet")
            (directory / "ready").touch()
    return output_xml


def active_normal_distributions(distributions: dict) -> dict:
    return {
        name: spec
        for name, spec in distributions.items()
        if spec.get("use", True)
    }


def recenter_swept_distributions(
    distributions: dict,
    point_values: dict,
    axes: tuple[str, ...],
) -> dict:
    recentered = {}
    for name, spec in distributions.items():
        next_spec = dict(spec)
        if name in axes and name in point_values and "mean" in next_spec:
            old_mean = float(next_spec["mean"])
            new_mean = float(point_values[name])
            mean_delta = new_mean - old_mean
            next_spec["mean"] = new_mean
            if "min" in next_spec:
                next_spec["min"] = float(next_spec["min"]) + mean_delta
            if "max" in next_spec:
                next_spec["max"] = float(next_spec["max"]) + mean_delta
        recentered[name] = next_spec
    return recentered


def sampled_geometry(base_geometry: dict[str, float], parameter_row: dict) -> dict[str, float]:
    geometry = dict(base_geometry)
    for axis_name in config.GEOMETRY_SHAPE_SWEEP_AXES:
        if axis_name in parameter_row:
            geometry[axis_name] = float(parameter_row[axis_name])
    return geometry


def geometry_group_key(geometry: dict[str, float]) -> str:
    shape = {
        key: geometry.get(key)
        for key in ("curve_x", "curve_y", "curve_z", "box_x", "box_y", "box_z")
    }
    return json.dumps(shape, sort_keys=True)


def group_parameter_rows_by_geometry(
    base_geometry: dict[str, float],
    parameter_rows: list[dict],
) -> list[tuple[dict[str, float], list[dict]]]:
    groups: dict[str, tuple[dict[str, float], list[dict]]] = {}
    for row in parameter_rows:
        geometry = sampled_geometry(base_geometry, row)
        key = geometry_group_key(geometry)
        if key not in groups:
            groups[key] = (geometry, [])
        groups[key][1].append(row)
    return list(groups.values())


def build_jobs() -> list[dict]:
    axes = config.canonicalize_axes(config.SWEEP_AXES)
    values_by_axis = [config.SWEEP_VALUES[axis_name] for axis_name in axes]
    geometry_axes = set(config.geometry_axis_names(axes))
    trial_axes = set(config.trial_axis_names(axes))

    jobs = []
    for point_index, indexed_values in enumerate(
        itertools.product(*(enumerate(values) for values in values_by_axis)),
        start=1,
    ):
        axis_indices = {
            axis_name: int(index)
            for axis_name, (index, _value) in zip(axes, indexed_values)
        }
        axis_values = {
            axis_name: float(value)
            for axis_name, (_index, value) in zip(axes, indexed_values)
        }
        geometry_overrides = {
            axis_name: value
            for axis_name, value in axis_values.items()
            if axis_name in geometry_axes
        }
        trial_overrides = {
            axis_name: value
            for axis_name, value in axis_values.items()
            if axis_name in trial_axes
        }
        geometry = resolve_geometry(geometry_overrides)
        # Foot offsets are applied once by the simulator to the foot geoms.
        # Include unswept defaults too, so placement stays defined by GEOMETRY_BASE.
        trial_overrides.update({
            axis_name: geometry[axis_name]
            for axis_name in config.FOOT_OFFSET_SWEEP_AXES
        })
        jobs.append(
            {
                "point_index": point_index,
                "pair_index": point_index,
                "axes": axes,
                "axis_indices": axis_indices,
                "axis_values": axis_values,
                "geometry_overrides": geometry_overrides,
                "trial_overrides": trial_overrides,
                "geometry": geometry,
            }
        )

    return jobs


def format_axis_values(axis_values: dict[str, float]) -> str:
    return ", ".join(f"{axis_name}={value}" for axis_name, value in axis_values.items())


def axis_label_for_logs(axes: tuple[str, ...]) -> str:
    return "(" + ", ".join(axes) + ")"


def json_dumps_compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def legacy_axis_metadata(job: dict) -> dict[str, str]:
    axes = job["axes"]
    axis_values = job["axis_values"]
    axis_indices = job["axis_indices"]
    metadata: dict[str, str] = {}

    if len(axes) >= 1:
        axis_name = axes[0]
        metadata.update(
            {
                "Sweep_X_Param": axis_name,
                "Sweep_X_Value": str(axis_values[axis_name]),
                "Sweep_X_Index": str(axis_indices[axis_name]),
            }
        )
    if len(axes) >= 2:
        axis_name = axes[1]
        metadata.update(
            {
                "Sweep_Y_Param": axis_name,
                "Sweep_Y_Value": str(axis_values[axis_name]),
                "Sweep_Y_Index": str(axis_indices[axis_name]),
            }
        )
    return metadata


def generic_axis_metadata(job: dict) -> dict[str, str]:
    axes = job["axes"]
    axis_values = job["axis_values"]
    axis_indices = job["axis_indices"]
    metadata = {
        "Sweep_Axes": ",".join(axes),
        "Sweep_Dim_Count": str(len(axes)),
    }
    for axis_name in axes:
        metadata.update(
            {
                f"Sweep_{axis_name}_Value": str(axis_values[axis_name]),
                f"Sweep_{axis_name}_Index": str(axis_indices[axis_name]),
            }
        )
    return metadata


def curve_legacy_index(job: dict, axis_name: str) -> str:
    axis_indices = job["axis_indices"]
    if axis_name not in axis_indices:
        return "-1"
    return str(axis_indices[axis_name])


def resolve_total_worker_budget(total_points: int) -> int:
    configured = getattr(config, "MAX_WORKERS", None)
    if configured is None:
        requested = os.cpu_count() or 1
    else:
        requested = int(configured)
    return max(1, requested)


def resolve_worker_plan(total_points: int) -> tuple[int, int]:
    total_budget = resolve_total_worker_budget(total_points)
    configured_trial_workers = getattr(config, "TRIAL_WORKERS_PER_PAIR", None)

    if configured_trial_workers is None:
        point_workers = min(total_points, total_budget)
        trial_workers = max(1, total_budget // point_workers)
        return point_workers, trial_workers

    trial_workers = max(1, min(int(configured_trial_workers), total_budget))
    point_workers = max(1, min(total_points, total_budget // trial_workers))
    return max(1, point_workers), trial_workers


def append_rows(rows: list[dict], results_csv: Path) -> int:
    if not rows:
        return 0

    previous_line_count = 0
    if results_csv.exists():
        with results_csv.open(newline="") as handle:
            previous_line_count = sum(1 for _ in handle)
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with results_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    with results_csv.open(newline="") as handle:
        new_line_count = sum(1 for _ in handle)
    expected_new_lines = len(rows) + (1 if write_header else 0)
    actual_new_lines = new_line_count - previous_line_count
    if actual_new_lines != expected_new_lines:
        raise RuntimeError(
            f"CSV append verification failed for {results_csv}: expected "
            f"{expected_new_lines} new line(s), found {actual_new_lines}."
        )
    return len(rows)


def plot_results_if_enabled(results_csv: Path) -> None:
    if not getattr(config, "PLOT_RESULTS_AT_END", False):
        return
    output_path = Path(
        getattr(
            config,
            "PLOT_RESULTS_PATH",
            Path(getattr(config, "SWEEP_DIR", results_csv.parent)) / "sweep_results_plot.png",
        )
    ).resolve()
    rows = plot_sweep_results.load_rows(results_csv)
    axis_spec = plot_sweep_results.resolve_sweep_axes(rows, list(rows[0].keys()))
    active_axes, ignored_axes = plot_sweep_results.resolve_active_axes(axis_spec.axis_names, [])
    rows = plot_sweep_results.filter_default_axis_values(rows, axis_spec, ignored_axes)
    plot_sweep_results.plot_scatter(
        rows=rows,
        axis_spec=axis_spec,
        active_axes=active_axes,
        min_distance=plot_sweep_results.default_success_min_distance(),
        output_path=output_path,
        colorbar_metric=getattr(config, "PLOT_COLORBAR", "distance"),
    )
    vprint(f"Wrote sweep plot to: {output_path}")


def split_parameter_rows(parameter_rows: list[dict], chunk_count: int) -> list[tuple[int, list[dict]]]:
    if chunk_count <= 1 or len(parameter_rows) <= 1:
        return [(0, parameter_rows)]

    chunk_count = min(chunk_count, len(parameter_rows))
    base_size, remainder = divmod(len(parameter_rows), chunk_count)
    chunks = []
    start = 0
    for chunk_index in range(chunk_count):
        chunk_size = base_size + (1 if chunk_index < remainder else 0)
        if chunk_size == 0:
            continue
        end = start + chunk_size
        chunks.append((start, parameter_rows[start:end]))
        start = end
    return chunks


def run_trial_chunks(
    model_xml_path: Path,
    parameter_rows: list[dict],
    save_all_results: bool,
    metadata: dict,
    trial_workers: int,
) -> list[dict]:
    chunks = split_parameter_rows(parameter_rows, trial_workers)
    if len(chunks) == 1:
        chunk_offset, chunk_rows = chunks[0]
        return sim_sweep.run_parameter_chunk(
            model_xml_path=model_xml_path,
            parameter_rows=chunk_rows,
            save_all_results=save_all_results,
            metadata=metadata,
            verbose=False,
            progress_every=100,
            progress_offset=chunk_offset,
            total_trials_override=len(parameter_rows),
        )

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                sim_sweep.run_parameter_chunk,
                model_xml_path,
                chunk_rows,
                save_all_results,
                metadata,
                False,
                100,
                chunk_offset,
                len(parameter_rows),
            )
            for chunk_offset, chunk_rows in chunks
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def build_metadata(
    job: dict,
    run_index: int,
    num_trials: int,
    geometry_override: dict[str, float] | None = None,
) -> dict[str, str]:
    geometry = geometry_override or job["geometry"]
    return {
        **generic_axis_metadata(job),
        **legacy_axis_metadata(job),
        "Curve_X": str(geometry["curve_x"]),
        "Curve_Y": str(geometry["curve_y"]),
        "Curve_Z": str(geometry.get("curve_z", mesh_gen.DEFAULT_CURVE_Z)),
        "Box_X": str(geometry["box_x"]),
        "Box_Y": str(geometry["box_y"]),
        "Box_Z": str(geometry.get("box_z", mesh_gen.DEFAULT_BOX_Z)),
        "Sweep_Trial_Overrides_JSON": json_dumps_compact(job["trial_overrides"]),
        "Mesh_X": str(geometry["curve_x"]),
        "Mesh_Y": str(geometry["curve_y"]),
        "Mesh_X_Index": curve_legacy_index(job, "curve_x"),
        "Mesh_Y_Index": curve_legacy_index(job, "curve_y"),
        "Point_Index": str(job["point_index"]),
        "Pair_Index": str(job["pair_index"]),
        "Run_Index": str(run_index),
        "Trials_Per_Point": str(num_trials),
        "Trials_Per_Pair": str(num_trials),
        "Mesh_Generator": mesh_gen.__name__,
        "Mesh_Generator_Entry_XML": str(mesh_gen.DEFAULT_ENTRY_XML),
        "Mesh_Generator_SCAD": str(mesh_gen.DEFAULT_SCAD_FILE),
    }


def process_point(
    job: dict,
    trial_workers: int,
    num_trials: int,
    save_all_results: bool,
    fixed_params: dict,
    normal_distributions: dict,
) -> dict:
    geometry = job["geometry"]

    with tempfile.TemporaryDirectory(prefix=f"geometry_sweep_point_{job['point_index']:03d}_") as temp_dir:
        temp_path = Path(temp_dir)
        rows: list[dict] = []
        runs_per_point = getattr(config, "RUNS_PER_POINT", getattr(config, "RUNS_PER_PAIR", 1))
        for run_index in range(1, runs_per_point + 1):
            point_fixed_params = {**fixed_params, **job["trial_overrides"]}
            point_values = {**point_fixed_params, **job["geometry"], **job["axis_values"]}
            point_distributions = recenter_swept_distributions(
                normal_distributions,
                point_values,
                job["axes"],
            )
            parameter_rows = sim_sweep.generate_parameter_samples(
                num_trials,
                fixed_params=point_fixed_params,
                normal_distributions=point_distributions,
            )
            for group_index, (trial_geometry, group_rows) in enumerate(
                group_parameter_rows_by_geometry(geometry, parameter_rows),
                start=1,
            ):
                output_xml = temp_path / f"modified_model_run{run_index}_group{group_index}.xml"
                mesh_out_dir = temp_path / f"foot_section_out_run{run_index}_group{group_index}"
                try:
                    cache_root = (
                        getattr(config, "GEOMETRY_CACHE_DIR", None)
                        if getattr(config, "SAVE_MESH_CACHE", True)
                        else None
                    )
                    if cache_root:
                        output_xml = cached_modified_xml(trial_geometry, Path(cache_root).resolve())
                    else:
                        generate_modified_xml(trial_geometry, output_xml, mesh_out_dir)
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    return {
                        "point_index": job["point_index"],
                        "pair_index": job["pair_index"],
                        "axes": job["axes"],
                        "axis_values": job["axis_values"],
                        "rows": [],
                        "skipped": True,
                        "skip_reason": str(exc),
                    }

                metadata = build_metadata(job, run_index, num_trials, trial_geometry)
                rows.extend(
                    run_trial_chunks(
                        model_xml_path=output_xml,
                        parameter_rows=group_rows,
                        save_all_results=save_all_results,
                        metadata=metadata,
                        trial_workers=trial_workers,
                    )
                )

        return {
            "point_index": job["point_index"],
            "pair_index": job["pair_index"],
            "axes": job["axes"],
            "axis_values": job["axis_values"],
            "rows": rows,
            "skipped": False,
            "skip_reason": "",
        }


def preview_target_value(axis_name: str, fixed_params: dict, normal_distributions: dict) -> float:
    if axis_name in config.SWEEP_BASE:
        return float(config.SWEEP_BASE[axis_name])
    if axis_name in fixed_params:
        return float(fixed_params[axis_name])
    if axis_name in normal_distributions:
        return float(normal_distributions[axis_name]["mean"])
    raise KeyError(f"No preview target value found for sweep axis '{axis_name}'.")


def select_center_preview_job(
    jobs: list[dict],
    fixed_params: dict,
    normal_distributions: dict,
) -> dict:
    def normalized_error(job: dict) -> float:
        total = 0.0
        for axis_name, value in job["axis_values"].items():
            target = preview_target_value(axis_name, fixed_params, normal_distributions)
            scale = max(abs(target), 1.0)
            total += ((float(value) - target) / scale) ** 2
        return total

    return min(jobs, key=normalized_error)


def center_trial_params(job: dict, fixed_params: dict, normal_distributions: dict) -> dict:
    params = {**fixed_params, **job["trial_overrides"]}
    for key, spec in normal_distributions.items():
        params[key] = float(spec["mean"])
    return params


def preview_first_trial(
    job: dict,
    num_trials: int,
    fixed_params: dict,
    normal_distributions: dict,
) -> None:
    with tempfile.TemporaryDirectory(prefix="geometry_sweep_preview_") as temp_dir:
        temp_path = Path(temp_dir)
        output_xml = temp_path / "modified_model.xml"
        mesh_out_dir = temp_path / "foot_section_out"

        first_params = center_trial_params(job, fixed_params, normal_distributions)
        preview_geometry = sampled_geometry(job["geometry"], first_params)
        generate_modified_xml(preview_geometry, output_xml, mesh_out_dir)
        vprint(
            "Previewing center sweep trial before multiprocessing starts. "
            f"Point {job['point_index']} ({format_axis_values(job['axis_values'])}), "
            f"params={sim_sweep.build_replay_params(first_params)}. "
            "Close the MuJoCo window to continue the sweep."
        )
        ctx = sim_sweep.load_simulation(output_xml)
        with mujoco.viewer.launch_passive(ctx.model, ctx.data) as viewer:
            sim_sweep.run_single_trial(
                ctx,
                first_params,
                verbose=True,
                viewer=viewer,
                iteration_duration=PREVIEW_TRIAL_DURATION,
                stop_on_fall=True,
            )


def main() -> None:
    output_xml = Path(config.OUTPUT_XML).resolve()
    results_csv = Path(config.RESULTS_CSV).resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    if config.OVERWRITE_RESULTS_CSV and results_csv.exists():
        results_csv.unlink()

    jobs = build_jobs()
    axes = config.canonicalize_axes(config.SWEEP_AXES)
    total_points = len(jobs)
    runs_per_point = getattr(config, "RUNS_PER_POINT", getattr(config, "RUNS_PER_PAIR", 1))
    total_runs = total_points * runs_per_point
    point_workers, trial_workers = resolve_worker_plan(total_points)
    num_trials = config.trials_for_axes(axes)
    save_all_results = True
    fixed_params = dict(config.FIXED_TRIAL_PARAMS)
    normal_distributions = active_normal_distributions(dict(config.NORMAL_TRIAL_DISTRIBUTIONS))

    if not config.USE_RAMPED_START:
        normal_distributions.pop("ramp_time", None)
    vprint(
        f"Starting geometry sweep on {axis_label_for_logs(axes)}: "
        f"{' x '.join(str(len(config.SWEEP_VALUES[axis_name])) for axis_name in axes)} grid, "
        f"{runs_per_point} run(s) per point "
        f"({total_runs} total test_sim_sweep.py runs)."
    )
    vprint(
        f"Trial configuration: {num_trials} trials per geometry point | "
        f"fixed params={fixed_params} | randomized={sorted(normal_distributions)} | "
        f"startup settle={sim_sweep.STARTUP_SETTLE_S:.1f}s"
    )
    vprint(
        f"Worker plan: {point_workers} point worker(s), "
        f"{trial_workers} trial worker(s) per point "
        f"(budget {point_workers * trial_workers}/{resolve_total_worker_budget(total_points)})."
    )
    vprint(f"Writing combined results to: {results_csv}")

    if PREVIEW_FIRST_TRIAL and jobs:
        preview_job = select_center_preview_job(jobs, fixed_params, normal_distributions)
        preview_first_trial(preview_job, num_trials, fixed_params, normal_distributions)

    completed_points = 0
    merged_rows = 0
    skipped_points = 0
    passed_trials = 0
    nonfallen_trials = 0
    last_successful_job = None

    with ProcessPoolExecutor(max_workers=point_workers) as executor:
        future_to_job = {
            executor.submit(
                process_point,
                job,
                trial_workers,
                num_trials,
                save_all_results,
                fixed_params,
                normal_distributions,
            ): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except (RuntimeError, ValueError, TimeoutError) as exc:
                result = {"skipped": True, "skip_reason": str(exc)}
            completed_points += 1
            if result.get("skipped"):
                skipped_points += 1
                vprint(
                    f"[{completed_points}/{total_points}] Skipped point {job['point_index']} "
                    f"({format_axis_values(job['axis_values'])}) before simulation trials.\n"
                    f"Reason: {result['skip_reason']}"
                )
                continue
            appended = append_rows(result["rows"], results_csv)
            merged_rows += appended
            passed_trials += sum(bool(row["Gait_Quality_Pass"]) for row in result["rows"])
            nonfallen_trials += sum(not bool(row["Fell"]) for row in result["rows"])
            with results_csv.with_suffix(".status.json").open("w") as status_file:
                json.dump({"completed_points": completed_points, "total_points": total_points,
                           "rows": merged_rows, "skipped": skipped_points,
                           "gait_passes": passed_trials, "nonfallen": nonfallen_trials}, status_file)
            last_successful_job = job
            vprint(
                f"[{completed_points}/{total_points}] Finished point {job['point_index']} "
                f"({format_axis_values(job['axis_values'])}) -> merged {appended} row(s); "
                f"gait passes={passed_trials}, nonfallen={nonfallen_trials}/{merged_rows}"
            )

    if last_successful_job is not None and getattr(config, "WRITE_FINAL_XML_SNAPSHOT", True):
        last_geometry = last_successful_job["geometry"]
        vprint(
            "Writing final XML snapshot for the last successful grid point "
            f"({format_axis_values(last_successful_job['axis_values'])}) to {output_xml}"
        )
        generate_modified_xml(last_geometry, output_xml, Path(mesh_gen.DEFAULT_OUT_DIR).resolve())
    elif last_successful_job is not None:
        vprint("Skipping final XML snapshot because WRITE_FINAL_XML_SNAPSHOT is disabled.")
    else:
        vprint("No valid geometry points were generated; final XML snapshot was not written.")

    vprint(
        f"\nDone. Combined results are in {results_csv} ({merged_rows} row(s)); "
        f"skipped {skipped_points}/{total_points} invalid geometry point(s)."
    )
    if merged_rows:
        plot_results_if_enabled(results_csv)


if __name__ == "__main__":
    main()
