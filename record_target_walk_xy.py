#!/usr/bin/env python3
"""Search randomized trials, then record the top eligible walks by distance or CoT."""

# example run
'''
python3 record_target_walk_xy.py \
  --x-percent 6 \
  --y-percent -10 \
  --attempts 500 \
  --top-n 5 \
  --rank-by cot \
  --video-out data/videos/x6_y-10_cot.mp4
'''

import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np

import run_xy_sweep as xy_sweep
import test_sim_sweep as sweep
import xy_sweep_config as sweep_config


def build_tracking_camera(ctx: sweep.SimulationContext, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = ctx.torso_body_id
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def resolved_record_size(ctx: sweep.SimulationContext, width: int, height: int) -> tuple[int, int]:
    max_width = int(ctx.model.vis.global_.offwidth)
    max_height = int(ctx.model.vis.global_.offheight)
    return min(width, max_width), min(height, max_height)


def percent_to_value(base: float, percent_delta: float) -> float:
    return round(base * (1.0 + percent_delta / 100.0), 6)


def run_trial(
    ctx: sweep.SimulationContext,
    params: dict,
    video_fps: int | None = None,
    width: int = 960,
    height: int = 540,
    camera_distance: float = 3.5,
    camera_azimuth: float = 130.0,
    camera_elevation: float = -15.0,
) -> tuple[dict, list[np.ndarray]]:
    mujoco.mj_resetData(ctx.model, ctx.data)
    sweep.apply_foot_offsets(ctx, params["foot_x"], params["foot_y"])
    mujoco.mj_forward(ctx.model, ctx.data)

    renderer = None
    camera = None
    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    output_interval = None if video_fps is None else 1.0 / video_fps

    if video_fps is not None:
        width, height = resolved_record_size(ctx, width, height)
        renderer = mujoco.Renderer(ctx.model, height, width)
        camera = build_tracking_camera(
            ctx,
            distance=camera_distance,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
        )

    start_x = ctx.data.xpos[ctx.torso_body_id][0]
    start_y = ctx.data.xpos[ctx.torso_body_id][1]
    hip_omega = params["freq_hz"] * 2 * np.pi
    leg_amp_rad = np.deg2rad(params["amp_deg"])
    cmd_buffer = [0.0] * sweep.CMD_DELAY_STEPS
    fell = False
    max_steps = int(sweep.ITERATION_DURATION / ctx.model.opt.timestep)
    energy_used = 0.0

    def maybe_record_frame() -> None:
        nonlocal next_frame_time
        if renderer is None or output_interval is None or camera is None:
            return
        while ctx.data.time >= next_frame_time:
            renderer.update_scene(ctx.data, camera=camera)
            frames.append(renderer.render().copy())
            next_frame_time += output_interval

    maybe_record_frame()

    for _ in range(max_steps):
        t = ctx.data.time
        target_pos_rad, target_vel_rad = sweep.calculate_sine_reference(
            t,
            hip_omega,
            leg_amp_rad,
            params["start_amp_mult"],
            params["start_freq_mult"],
        )

        current_pos = ctx.data.qpos[ctx.qpos_idx]
        current_vel = ctx.data.qvel[ctx.qvel_idx]

        ramp = min(1.0, t / sweep.RAMP_TIME) if sweep.USE_RAMP and sweep.RAMP_TIME > 0 else 1.0
        tau = (
            params["Kp"] * ramp * (target_pos_rad - current_pos)
            + params["Kd"] * ramp * (target_vel_rad - current_vel)
        )
        tau = np.clip(tau, -sweep.TORQUE_LIMIT, sweep.TORQUE_LIMIT)
        energy_used += abs(tau * current_vel) * ctx.model.opt.timestep

        cmd_buffer.append(tau)
        ctx.data.ctrl[0] = cmd_buffer.pop(0)
        mujoco.mj_step(ctx.model, ctx.data)
        maybe_record_frame()

        if sweep.check_has_fallen(ctx):
            fell = True
            break

    final_x = ctx.data.xpos[ctx.torso_body_id][0]
    final_y = ctx.data.xpos[ctx.torso_body_id][1]
    distance = float(np.sqrt((final_x - start_x) ** 2 + (final_y - start_y) ** 2))
    cot = (energy_used / (ctx.total_mass * ctx.gravity * distance)) if distance > 1e-6 else float("inf")

    if renderer is not None:
        renderer.close()

    result_row = {
        "Foot_X": round(params["foot_x"], 4),
        "Foot_Y": round(params["foot_y"], 4),
        "Kp": round(params["Kp"], 2),
        "Kd": round(params["Kd"], 2),
        "Start_Amp_Mult": round(params["start_amp_mult"], 3),
        "Start_Freq_Mult": round(params["start_freq_mult"], 3),
        "Amplitude_Deg": round(params["amp_deg"], 2),
        "Frequency_Hz": round(params["freq_hz"], 3),
        "Fell": fell,
        "Distance_Traversed": round(distance, 4),
        "CoT": round(cot, 4),
    }
    return result_row, frames


def export_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames were recorded for the selected trial.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def trial_matches(result_row: dict, target_distance: float, tolerance: float) -> bool:
    return (not result_row["Fell"]) and abs(result_row["Distance_Traversed"] - target_distance) <= tolerance


def is_eligible_trial(result_row: dict, target_distance: float | None, tolerance: float) -> bool:
    if result_row["Fell"]:
        return False
    if target_distance is None:
        return True
    return trial_matches(result_row, target_distance, tolerance)


def rank_trial_key(result_row: dict, rank_by: str) -> float:
    if rank_by == "distance":
        return float(result_row["Distance_Traversed"])
    cot = float(result_row["CoT"])
    if not math.isfinite(cot):
        return float("inf")
    return cot


def output_path_for_rank(base_path: Path, rank: int, total: int) -> Path:
    if total <= 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_rank{rank:02d}{base_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-percent", type=float, required=True, help="Percent change from BASE_X to generate.")
    parser.add_argument("--y-percent", type=float, required=True, help="Percent change from BASE_Y to generate.")
    parser.add_argument("--attempts", type=int, default=500, help="Maximum randomized trials to search.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top eligible trials to record.",
    )
    parser.add_argument(
        "--rank-by",
        choices=("distance", "cot"),
        default="distance",
        help="Metric used to rank eligible trials. Distance prefers larger values; CoT prefers smaller values.",
    )
    parser.add_argument(
        "--target-distance",
        type=float,
        default=None,
        help="Desired walking distance in meters. If omitted, all non-falling trials are eligible.",
    )
    parser.add_argument("--tolerance", type=float, default=0.1, help="Allowed absolute error around target distance.")
    parser.add_argument("--video-out", type=Path, default=Path("data/videos/target_walk_xy.mp4"), help="Output video path.")
    parser.add_argument("--video-fps", type=int, default=30, help="Recorded video FPS.")
    parser.add_argument("--width", type=int, default=960, help="Recorded video width.")
    parser.add_argument("--height", type=int, default=540, help="Recorded video height.")
    parser.add_argument("--camera-distance", type=float, default=3.5, help="Tracking camera distance.")
    parser.add_argument("--camera-azimuth", type=float, default=130.0, help="Tracking camera azimuth.")
    parser.add_argument("--camera-elevation", type=float, default=-15.0, help="Tracking camera elevation.")
    parser.add_argument("--params-out", type=Path, default=None, help="Optional JSON path for the matched trial parameters.")
    parser.add_argument("--print-every", type=int, default=25, help="Progress print interval during headless search.")
    args = parser.parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1.")

    mesh_x = percent_to_value(sweep_config.BASE_X, args.x_percent)
    mesh_y = percent_to_value(sweep_config.BASE_Y, args.y_percent)
    parameter_rows = sweep.generate_lhs_samples(args.attempts)

    with tempfile.TemporaryDirectory(prefix="target_walk_xy_") as temp_dir:
        temp_path = Path(temp_dir)
        output_xml = temp_path / "modified_model.xml"
        mesh_out_dir = temp_path / "foot_section_out"

        print(
            f"Generating model for mesh pair X={mesh_x} ({args.x_percent:+g}%), "
            f"Y={mesh_y} ({args.y_percent:+g}%)"
        )
        xy_sweep.generate_modified_xml(mesh_x, mesh_y, output_xml, mesh_out_dir)
        ctx = sweep.load_simulation(output_xml)

        attempted_trials: list[dict] = []
        best_overall_index = None
        best_overall_result = None

        for idx, params in enumerate(parameter_rows, 1):
            result_row, _ = run_trial(ctx, params)
            if idx == 1 or idx % args.print_every == 0:
                progress_message = (
                    f"[{idx}/{args.attempts}] dist={result_row['Distance_Traversed']:.4f} "
                    f"cot={result_row['CoT']:.4f} fell={result_row['Fell']}"
                )
                if args.target_distance is not None:
                    progress_message += (
                        f" target={args.target_distance:.4f} tol={args.tolerance:.4f}"
                    )
                print(progress_message)
            if (not result_row["Fell"]) and (
                best_overall_result is None
                or result_row["Distance_Traversed"] > best_overall_result["Distance_Traversed"]
            ):
                best_overall_index = idx
                best_overall_result = result_row
            if is_eligible_trial(result_row, args.target_distance, args.tolerance):
                attempted_trials.append(
                    {
                        "attempt_index": idx,
                        "trial_parameters": params,
                        "search_result": result_row,
                    }
                )

        if not attempted_trials:
            if best_overall_result is not None and best_overall_index is not None:
                print(
                    f"Best non-falling trial was {best_overall_index}/{args.attempts}: "
                    f"distance={best_overall_result['Distance_Traversed']:.4f}, "
                    f"CoT={best_overall_result['CoT']:.4f}"
                )
            if args.target_distance is None:
                raise SystemExit(f"No non-falling trial found after {args.attempts} attempts.")
            raise SystemExit(
                f"No non-falling trial matched target distance {args.target_distance:.4f} "
                f"within tolerance {args.tolerance:.4f} after {args.attempts} attempts."
            )

        reverse_rank = args.rank_by == "distance"
        ranked_trials = sorted(
            attempted_trials,
            key=lambda trial: rank_trial_key(trial["search_result"], args.rank_by),
            reverse=reverse_rank,
        )
        selected_trials = ranked_trials[: args.top_n]
        if len(selected_trials) < args.top_n:
            print(
                f"Only {len(selected_trials)} eligible trials found; recording all of them."
            )

        selection_label = "matching" if args.target_distance is not None else "eligible"
        print(
            f"Selected top {len(selected_trials)} {selection_label} trial(s) ranked by {args.rank_by}."
        )
        for rank, trial in enumerate(selected_trials, 1):
            trial_result = trial["search_result"]
            print(
                f"  rank {rank}: attempt {trial['attempt_index']}/{args.attempts} "
                f"distance={trial_result['Distance_Traversed']:.4f} CoT={trial_result['CoT']:.4f}"
            )

        print("Re-running selected trial(s) with recording enabled...")

        record_ctx = sweep.load_simulation(output_xml)
        resolved_width, resolved_height = resolved_record_size(record_ctx, args.width, args.height)
        print(
            f"Recording at {resolved_width}x{resolved_height} "
            f"(model offscreen limit {record_ctx.model.vis.global_.offwidth}x"
            f"{record_ctx.model.vis.global_.offheight})"
        )

        recorded_payloads: list[dict] = []
        for rank, selected_trial in enumerate(selected_trials, 1):
            trial_video_out = output_path_for_rank(args.video_out, rank, len(selected_trials))
            print(
                f"Recording rank {rank}/{len(selected_trials)} "
                f"from attempt {selected_trial['attempt_index']} to {trial_video_out}"
            )
            recorded_result, frames = run_trial(
                record_ctx,
                selected_trial["trial_parameters"],
                video_fps=args.video_fps,
                width=resolved_width,
                height=resolved_height,
                camera_distance=args.camera_distance,
                camera_azimuth=args.camera_azimuth,
                camera_elevation=args.camera_elevation,
            )
            export_video(frames, trial_video_out, args.video_fps)
            print(f"Saved video to {trial_video_out}")
            recorded_payloads.append(
                {
                    "rank": rank,
                    "attempt_index": selected_trial["attempt_index"],
                    "search_result": selected_trial["search_result"],
                    "recorded_result": recorded_result,
                    "trial_parameters": selected_trial["trial_parameters"],
                    "video_out": str(trial_video_out),
                }
            )

        if args.params_out is not None:
            args.params_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "x_percent": args.x_percent,
                "y_percent": args.y_percent,
                "mesh_x": mesh_x,
                "mesh_y": mesh_y,
                "attempts": args.attempts,
                "top_n": args.top_n,
                "rank_by": args.rank_by,
                "target_distance": args.target_distance,
                "tolerance": args.tolerance,
                "recordings": recorded_payloads,
            }
            args.params_out.write_text(json.dumps(payload, indent=2))
            print(f"Saved selected parameters to {args.params_out}")


if __name__ == "__main__":
    main()
