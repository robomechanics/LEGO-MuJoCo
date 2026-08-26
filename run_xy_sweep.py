#!/usr/bin/env python3
"""Generate modified_model.xml variants for X/Y foot meshes and run test_sim_sweep."""

import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gen_new_xml as mesh_gen
import test_sim_sweep as sim_sweep
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


def resolve_total_worker_budget(total_pairs: int) -> int:
    configured = getattr(config, "MAX_WORKERS", None)
    if configured is None:
        requested = os.cpu_count() or 1
    else:
        requested = int(configured)
    return max(1, requested)


def resolve_worker_plan(total_pairs: int) -> tuple[int, int]:
    total_budget = resolve_total_worker_budget(total_pairs)
    configured_trial_workers = getattr(config, "TRIAL_WORKERS_PER_PAIR", None)

    if configured_trial_workers is None:
        pair_workers = min(total_pairs, total_budget)
        trial_workers = max(1, total_budget // pair_workers)
        return pair_workers, trial_workers

    trial_workers = max(1, min(int(configured_trial_workers), total_budget))
    pair_workers = max(1, min(total_pairs, total_budget // trial_workers))
    if pair_workers == 0:
        pair_workers = 1
    return pair_workers, trial_workers


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


def split_parameter_rows(parameter_rows: list[dict], chunk_count: int) -> list[tuple[int, list[dict]]]:
    if chunk_count <= 1 or len(parameter_rows) <= 1:
        return [(0, parameter_rows)]

    chunk_count = min(chunk_count, len(parameter_rows))
    base, remainder = divmod(len(parameter_rows), chunk_count)
    chunks = []
    start = 0
    for chunk_idx in range(chunk_count):
        chunk_size = base + (1 if chunk_idx < remainder else 0)
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

    results = []
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


def process_pair(job: dict, trial_workers: int, num_trials: int, save_all_results: bool) -> dict:
    pair_index = job["pair_index"]
    x_value = job["x_value"]
    y_value = job["y_value"]
    x_index = job["x_index"]
    y_index = job["y_index"]

    with tempfile.TemporaryDirectory(prefix=f"xy_sweep_pair_{pair_index:03d}_") as temp_dir:
        temp_path = Path(temp_dir)
        output_xml = temp_path / "modified_model.xml"
        mesh_out_dir = temp_path / "foot_section_out"

        generate_modified_xml(x_value, y_value, output_xml, mesh_out_dir)

        rows = []
        for run_index in range(1, config.RUNS_PER_PAIR + 1):
            parameter_rows = sim_sweep.generate_lhs_samples(num_trials)
            metadata = {
                "Mesh_X": str(x_value),
                "Mesh_Y": str(y_value),
                "Mesh_X_Index": str(x_index),
                "Mesh_Y_Index": str(y_index),
                "Pair_Index": str(pair_index),
                "Run_Index": str(run_index),
                "Mesh_Generator": mesh_gen.__name__,
                "Mesh_Generator_Entry_XML": str(mesh_gen.ENTRY_XML),
                "Mesh_Generator_SCAD": str(Path(mesh_gen.SCAD_DIR) / "feet_generator.scad"),
            }
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
    pair_workers, trial_workers = resolve_worker_plan(total_pairs)
    num_trials = int(config.TEST_SIM_NUM_TRIALS) if config.TEST_SIM_NUM_TRIALS is not None else 500
    save_all_results = True

    vprint(
        f"Starting X/Y mesh sweep: {len(config.X_VALUES)} x-values, "
        f"{len(config.Y_VALUES)} y-values, {config.RUNS_PER_PAIR} runs per pair "
        f"({total_runs} total test_sim_sweep.py runs)."
    )
    vprint(
        f"Worker plan: {pair_workers} pair worker(s), "
        f"{trial_workers} trial worker(s) per pair "
        f"(budget {pair_workers * trial_workers}/{resolve_total_worker_budget(total_pairs)})."
    )
    vprint(f"Writing combined results to: {results_csv}")

    completed_pairs = 0
    merged_rows = 0

    with ProcessPoolExecutor(max_workers=pair_workers) as executor:
        future_to_job = {
            executor.submit(process_pair, job, trial_workers, num_trials, save_all_results): job
            for job in jobs
        }

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
