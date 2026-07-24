# sweeps through values of x/y and runs randomized trials N times. see xy_sweep_config.py to change parameters.

#!/usr/bin/env python3
"""Generate modified_model.xml for each X/Y foot mesh pair and run test_sim_sweep."""

import os
import subprocess
import sys
from pathlib import Path

import gen_new_xml as mesh_gen
import xy_sweep_config as config

ROOT_DIR = Path(__file__).resolve().parent
mesh_gen.VERBOSE = False


def vprint(*args, **kwargs) -> None:
    if config.RUNNER_VERBOSE:
        print(*args, **kwargs)


def generate_modified_xml(x_value: float, y_value: float, output_xml: Path) -> None:
    scad_file = Path(mesh_gen.SCAD_DIR) / "feet_generator.scad"
    out_dir = Path(mesh_gen.OUT_DIR).resolve()

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


def main() -> None:
    output_xml = Path(config.OUTPUT_XML).resolve()
    results_csv = Path(config.RESULTS_CSV).resolve()

    if config.OVERWRITE_RESULTS_CSV and results_csv.exists():
        results_csv.unlink()

    total_pairs = len(config.X_VALUES) * len(config.Y_VALUES)
    total_runs = total_pairs * config.RUNS_PER_PAIR
    completed_runs = 0

    vprint(
        f"Starting X/Y mesh sweep: {len(config.X_VALUES)} x-values, "
        f"{len(config.Y_VALUES)} y-values, {config.RUNS_PER_PAIR} runs per pair "
        f"({total_runs} total test_sim_sweep.py runs)."
    )
    vprint(f"Overwriting generated XML at: {output_xml}")
    vprint(f"Appending results to: {results_csv}")

    pair_index = 0
    for x_index, x_value in enumerate(config.X_VALUES):
        for y_index, y_value in enumerate(config.Y_VALUES):
            pair_index += 1
            vprint(
                f"\n[{pair_index}/{total_pairs}] Generating mesh XML "
                f"for X={x_value}, Y={y_value}"
            )
            generate_modified_xml(x_value, y_value, output_xml)
            vprint(f"[{pair_index}/{total_pairs}] XML ready: {output_xml}")

            for run_index in range(1, config.RUNS_PER_PAIR + 1):
                completed_runs += 1
                vprint(
                    f"[{completed_runs}/{total_runs}] Running test_sim_sweep.py "
                    f"for pair {pair_index}, repeat {run_index}"
                )
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
                vprint(
                    f"[{completed_runs}/{total_runs}] Finished test_sim_sweep.py "
                    f"for pair {pair_index}, repeat {run_index}"
                )

    vprint(f"\nDone. Combined results are in {results_csv}")


if __name__ == "__main__":
    main()
