#!/usr/bin/env python3
"""Generate modified_model.xml variants for X/Y foot meshes and run test_sim_sweep."""

import csv
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gen_new_xml as mesh_gen
import xy_sweep_config as config

ROOT_DIR = Path(__file__).resolve().parent
mesh_gen.VERBOSE = False


def vprint(*args, **kwargs) -> None:
    if config.RUNNER_VERBOSE:
        print(*args, **kwargs)


def generate_modified_xml(x_value: float, y_value: float, output_xml: Path, out_dir: Path) -> None:
    scad_file = Path(mesh_gen.SCAD_DIR) / "feet_generator.scad"

    sections = mesh_gen.generate_all_sections(
        scad_file,
        out_dir,
        x_value,
        y_value,
        mesh_gen.Z,
        mesh_gen.BOX_X,
        mesh_gen.BOX_Y,
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


def run_test_sim_sweep(
    x_value: float,
    y_value: float,
    x_index: int,
    y_index: int,
    pair_index: int,
    run_index: int,
    output_xml: Path,
    results_csv: Path,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_XML_PATH": str(output_xml),
            "SWEEP_RESULTS_CSV": str(results_csv),
            "SWEEP_APPEND_RESULTS": "1",
            "SWEEP_MESH_X": str(x_value),
            "SWEEP_MESH_Y": str(y_value),
            "SWEEP_MESH_X_INDEX": str(x_index),
            "SWEEP_MESH_Y_INDEX": str(y_index),
            "SWEEP_PAIR_INDEX": str(pair_index),
            "SWEEP_RUN_INDEX": str(run_index),
            "TEST_SIM_SWEEP_VERBOSE": "0",
            "SWEEP_SAVE_ALL_RESULTS": "1",
        }
    )
    if config.TEST_SIM_NUM_TRIALS is not None:
        env["NUM_TRIALS"] = str(config.TEST_SIM_NUM_TRIALS)
    if config.MIN_DISTANCE is not None:
        env["MIN_DISTANCE"] = str(config.MIN_DISTANCE)

    subprocess.run(
        [sys.executable, str(ROOT_DIR / "test_sim_sweep.py")],
        check=True,
        cwd=ROOT_DIR,
        env=env,
    )


def build_jobs() -> list[dict]:
    jobs = []
    pair_index = 0
    for x_index, x_value in enumerate(config.X_VALUES):
        for y_index, y_value in enumerate(config.Y_VALUES):
            pair_index += 1
            jobs.append(
                {
                    "x_index": x_index,
                    "y_index": y_index,
                    "x_value": x_value,
                    "y_value": y_value,
                    "pair_index": pair_index,
                }
            )
    return jobs


def resolve_max_workers(total_pairs: int) -> int:
    configured = getattr(config, "MAX_WORKERS", None)
    if configured is None:
        requested = os.cpu_count() or 1
    else:
        requested = int(configured)
    return max(1, min(requested, total_pairs))


def read_results_rows(results_csv: Path) -> list[dict]:
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return []

    with results_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_rows(rows: list[dict], results_csv: Path) -> int:
    if not rows:
        return 0

    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with results_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def process_pair(job: dict) -> dict:
    pair_index = job["pair_index"]
    x_value = job["x_value"]
    y_value = job["y_value"]
    x_index = job["x_index"]
    y_index = job["y_index"]

    with tempfile.TemporaryDirectory(prefix=f"xy_sweep_pair_{pair_index:03d}_") as temp_dir:
        temp_path = Path(temp_dir)
        output_xml = temp_path / "modified_model.xml"
        mesh_out_dir = temp_path / "foot_section_out"
        results_csv = temp_path / "pair_results.csv"

        generate_modified_xml(x_value, y_value, output_xml, mesh_out_dir)

        for run_index in range(1, config.RUNS_PER_PAIR + 1):
            run_test_sim_sweep(
                x_value,
                y_value,
                x_index,
                y_index,
                pair_index,
                run_index,
                output_xml,
                results_csv,
            )

        rows = read_results_rows(results_csv)
        return {
            "pair_index": pair_index,
            "x_value": x_value,
            "y_value": y_value,
            "rows": rows,
        }


def main() -> None:
    output_xml = Path(config.OUTPUT_XML).resolve()
    results_csv = Path(config.RESULTS_CSV).resolve()

    if config.OVERWRITE_RESULTS_CSV and results_csv.exists():
        results_csv.unlink()

    jobs = build_jobs()
    total_pairs = len(jobs)
    total_runs = total_pairs * config.RUNS_PER_PAIR
    max_workers = resolve_max_workers(total_pairs)

    vprint(
        f"Starting X/Y mesh sweep: {len(config.X_VALUES)} x-values, "
        f"{len(config.Y_VALUES)} y-values, {config.RUNS_PER_PAIR} runs per pair "
        f"({total_runs} total test_sim_sweep.py runs)."
    )
    vprint(f"Using {max_workers} worker process(es).")
    vprint(f"Writing combined results to: {results_csv}")

    completed_pairs = 0
    merged_rows = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(process_pair, job): job for job in jobs}

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            completed_pairs += 1
            appended = append_rows(result["rows"], results_csv)
            merged_rows += appended
            vprint(
                f"[{completed_pairs}/{total_pairs}] Finished pair {job['pair_index']} "
                f"(X={job['x_value']}, Y={job['y_value']}) -> merged {appended} row(s)"
            )

    if jobs:
        last_job = jobs[-1]
        vprint(
            f"Writing final XML snapshot for the last grid point "
            f"(X={last_job['x_value']}, Y={last_job['y_value']}) to {output_xml}"
        )
        generate_modified_xml(
            last_job["x_value"],
            last_job["y_value"],
            output_xml,
            Path(mesh_gen.OUT_DIR).resolve(),
        )

    vprint(f"\nDone. Combined results are in {results_csv} ({merged_rows} row(s)).")


if __name__ == "__main__":
    main()
