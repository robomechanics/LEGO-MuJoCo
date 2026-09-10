#!/usr/bin/env python3
"""Run a geometry sweep over any configured set of foot parameters."""

from __future__ import annotations

import csv
import itertools
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco.viewer

import gen_new_xml_v2 as mesh_gen
import test_sim_sweep as sim_sweep
import sweep_config as config

ROOT_DIR = Path(__file__).resolve().parent
mesh_gen.VERBOSE = False
PREVIEW_FIRST_TRIAL = True


def vprint(*args, **kwargs) -> None:
    if config.RUNNER_VERBOSE:
        print(*args, **kwargs)


def resolve_geometry(overrides: dict[str, float] | None = None) -> dict[str, float]:
    geometry = dict(config.GEOMETRY_BASE)
    if overrides:
        geometry.update(overrides)
    return geometry


def generate_modified_xml(geometry: dict[str, float], output_xml: Path, out_dir: Path) -> None:
    scad_file = Path(mesh_gen.SCAD_DIR) / "shell_feet_generator.scad"

    # Public geometry convention: +x forward, +y robot-left/lateral.
    # The OpenSCAD local X axis is the front/back slicing axis, so pass the
    # sweep geometry through directly.
    if hasattr(mesh_gen, "generate_full_feet") and getattr(mesh_gen, "SUFFIX_TO_SECTION", None) == {"1": "full"}:
        sections = mesh_gen.generate_full_feet(
            scad_file,
            out_dir,
            geometry["curve_x"],
            geometry["curve_y"],
            mesh_gen.Z,
            geometry["box_x"],
            geometry["box_y"],
            mesh_gen.FN,
            shell_thickness=getattr(mesh_gen, "WALL_THICKNESS", None),
        )
    else:
        sections = mesh_gen.generate_all_sections(
            scad_file,
            out_dir,
            geometry["curve_x"],
            geometry["curve_y"],
            mesh_gen.Z,
            geometry["box_x"],
            geometry["box_y"],
            mesh_gen.FN,
            swap_front_back=mesh_gen.SWAP_FRONT_BACK,
        )

    mesh_gen.inject_feet_into_model(
        Path(mesh_gen.ENTRY_XML),
        sections,
        output_xml,
        left_correction=mesh_gen.parse_correction_string(mesh_gen.LEFT_CORRECTION),
        right_correction=mesh_gen.parse_correction_string(mesh_gen.RIGHT_CORRECTION),
        left_offset=mesh_gen.LEFT_OFFSET,
        right_offset=mesh_gen.RIGHT_OFFSET,
        offset_frame=mesh_gen.OFFSET_FRAME,
    )


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

    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with results_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


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


def build_metadata(job: dict, run_index: int, num_trials: int) -> dict[str, str]:
    geometry = job["geometry"]
    return {
        **generic_axis_metadata(job),
        **legacy_axis_metadata(job),
        "Curve_X": str(geometry["curve_x"]),
        "Curve_Y": str(geometry["curve_y"]),
        "Box_X": str(geometry["box_x"]),
        "Box_Y": str(geometry["box_y"]),
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
        "Mesh_Generator_Entry_XML": str(mesh_gen.ENTRY_XML),
        "Mesh_Generator_SCAD": str(Path(mesh_gen.SCAD_DIR) / "feet_generator.scad"),
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
        output_xml = temp_path / "modified_model.xml"
        mesh_out_dir = temp_path / "foot_section_out"

        try:
            generate_modified_xml(geometry, output_xml, mesh_out_dir)
        except (ValueError, RuntimeError) as exc:
            return {
                "point_index": job["point_index"],
                "pair_index": job["pair_index"],
                "axes": job["axes"],
                "axis_values": job["axis_values"],
                "rows": [],
                "skipped": True,
                "skip_reason": str(exc),
            }

        rows: list[dict] = []
        runs_per_point = getattr(config, "RUNS_PER_POINT", getattr(config, "RUNS_PER_PAIR", 1))
        for run_index in range(1, runs_per_point + 1):
            point_fixed_params = {**fixed_params, **job["trial_overrides"]}
            parameter_rows = sim_sweep.generate_parameter_samples(
                num_trials,
                fixed_params=point_fixed_params,
                normal_distributions=normal_distributions,
            )
            metadata = build_metadata(job, run_index, num_trials)
            rows.extend(
                run_trial_chunks(
                    model_xml_path=output_xml,
                    parameter_rows=parameter_rows,
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

        generate_modified_xml(job["geometry"], output_xml, mesh_out_dir)
        point_fixed_params = {**fixed_params, **job["trial_overrides"]}
        parameter_rows = sim_sweep.generate_parameter_samples(
            1,
            fixed_params=point_fixed_params,
            normal_distributions=normal_distributions,
        )
        if not parameter_rows:
            vprint("No first-trial parameters were generated; skipping preview.")
            return

        first_params = parameter_rows[0]
        vprint(
            "Previewing first sweep trial before multiprocessing starts. "
            f"Point {job['point_index']} ({format_axis_values(job['axis_values'])}), "
            f"params={sim_sweep.build_replay_params(first_params)}. "
            "Close the MuJoCo window to continue the sweep."
        )
        ctx = sim_sweep.load_simulation(output_xml)
        with mujoco.viewer.launch_passive(ctx.model, ctx.data) as viewer:
            sim_sweep.run_single_trial(ctx, first_params, verbose=True, viewer=viewer)


def main() -> None:
    output_xml = Path(config.OUTPUT_XML).resolve()
    results_csv = Path(config.RESULTS_CSV).resolve()

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
    normal_distributions = dict(config.NORMAL_TRIAL_DISTRIBUTIONS)

    if not config.USE_RAMPED_START:
        normal_distributions.pop("ramp_time", None)
    for axis_name in axes:
        normal_distributions.pop(axis_name, None)

    vprint(
        f"Starting geometry sweep on {axis_label_for_logs(axes)}: "
        f"{' x '.join(str(len(config.SWEEP_VALUES[axis_name])) for axis_name in axes)} grid, "
        f"{runs_per_point} run(s) per point "
        f"({total_runs} total test_sim_sweep.py runs)."
    )
    vprint(
        f"Trial configuration: {num_trials} trials per geometry point | "
        f"fixed params={fixed_params} | randomized={sorted(normal_distributions)}"
    )
    vprint(
        f"Worker plan: {point_workers} point worker(s), "
        f"{trial_workers} trial worker(s) per point "
        f"(budget {point_workers * trial_workers}/{resolve_total_worker_budget(total_points)})."
    )
    vprint(f"Writing combined results to: {results_csv}")

    if PREVIEW_FIRST_TRIAL and jobs:
        preview_first_trial(jobs[0], num_trials, fixed_params, normal_distributions)

    completed_points = 0
    merged_rows = 0
    skipped_points = 0
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
            result = future.result()
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
            last_successful_job = job
            vprint(
                f"[{completed_points}/{total_points}] Finished point {job['point_index']} "
                f"({format_axis_values(job['axis_values'])}) -> merged {appended} row(s)"
            )

    if last_successful_job is not None:
        last_geometry = last_successful_job["geometry"]
        vprint(
            "Writing final XML snapshot for the last successful grid point "
            f"({format_axis_values(last_successful_job['axis_values'])}) to {output_xml}"
        )
        generate_modified_xml(last_geometry, output_xml, Path(mesh_gen.OUT_DIR).resolve())
    else:
        vprint("No valid geometry points were generated; final XML snapshot was not written.")

    vprint(
        f"\nDone. Combined results are in {results_csv} ({merged_rows} row(s)); "
        f"skipped {skipped_points}/{total_points} invalid geometry point(s)."
    )


if __name__ == "__main__":
    main()
