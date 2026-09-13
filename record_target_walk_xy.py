#!/usr/bin/env python3
"""Record the top saved gait passes from a run_sweep CSV, without resampling.

Example:
    MUJOCO_GL=egl python3 record_target_walk_xy.py --top-n 5 --rank-by distance
"""
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np

import run_sweep as sweep_runner
import test_sim_sweep as sweep
import sweep_config

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


def run_trial(ctx, params, video_fps=None, width=960, height=540,
              camera_distance=3.5, camera_azimuth=130.0, camera_elevation=-15.0):
    """Use precisely the sweep dynamics; the callback only renders frames."""
    frames = []
    renderer = None
    next_frame_time = 0.0
    try:
        if video_fps is not None:
            if video_fps <= 0:
                raise ValueError("Video FPS must be positive.")
            width, height = resolved_record_size(ctx, width, height)
            renderer = mujoco.Renderer(ctx.model, height, width)
            camera = build_tracking_camera(ctx, camera_distance, camera_azimuth, camera_elevation)

        def record_frame(state):
            nonlocal next_frame_time
            if renderer is None:
                return
            while state.data.time >= next_frame_time:
                renderer.update_scene(state.data, camera=camera)
                frames.append(renderer.render().copy())
                next_frame_time += 1.0 / video_fps

        result = sweep.run_single_trial(ctx, params, verbose=False, frame_callback=record_frame)
        return result, frames
    finally:
        if renderer is not None:
            renderer.close()


def csv_bool(value):
    return str(value).strip().lower() in {"true", "1"}


def select_trials(csv_path, top_n, rank_by, target_distance=None, tolerance=0.1):
    """Read a bounded snapshot of a possibly still-growing result file."""
    eligible = []
    metric = {"distance": "Distance_Traversed", "cot": "CoT", "walk_score": "Walk_Score"}[rank_by]
    with csv_path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size

        def snapshot_lines():
            while handle.tell() < size:
                line = handle.readline(size - handle.tell())
                if not line.endswith(b"\n"):
                    return  # The sweep has not finished writing this row.
                yield line.decode("utf-8")

        reader = csv.DictReader(snapshot_lines())
        for row in reader:
            if not csv_bool(row.get("Gait_Quality_Pass")) or csv_bool(row.get("Fell")):
                continue
            try:
                score = float(row[metric])
                distance = float(row["Distance_Traversed"])
            except (ValueError, TypeError, KeyError):
                continue  # Incomplete trailing row of an active sweep.
            if not math.isfinite(score):
                continue
            if target_distance is not None and abs(distance - target_distance) > tolerance:
                continue
            eligible.append(row)
    eligible.sort(key=lambda row: float(row[metric]), reverse=rank_by != "cot")
    return eligible[:top_n]


def saved_trial(row):
    """Never substitute today's defaults for missing historical parameters."""
    params = json.loads(row["Replay_Params_JSON"])
    required = {"foot_x", "foot_y", "torque_limit", "Kp", "Kd", "start_amp_mult",
                "start_freq_mult", "ramp_time", "amp_deg", "freq_hz"}
    missing = required - params.keys()
    if missing:
        raise ValueError(f"Saved trial is missing replay parameters: {sorted(missing)}")
    params = {name: float(params[name]) for name in required}
    geometry = {name: float(row[column]) for name, column in
                (("curve_x", "Curve_X"), ("curve_y", "Curve_Y"), ("box_x", "Box_X"), ("box_y", "Box_Y"))}
    if not all(math.isfinite(v) for v in (*params.values(), *geometry.values())):
        raise ValueError("Saved trial contains non-finite parameters.")
    return geometry, params


def resolve_model(geometry, cache_dir, temporary_dir):
    # Prefer the exact XML and meshes used by the sweep, not current generator defaults.
    key = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:20]
    cached = cache_dir / key / "modified_model.xml"
    if cached.exists() and (cached.parent / "ready").exists():
        return cached, "sweep_cache"
    output = temporary_dir / "modified_model.xml"
    print("Cached model unavailable; regenerating saved shape with the current generator.")
    sweep_runner.generate_modified_xml(geometry, output, temporary_dir / "feet")
    return output, "regenerated"


def replay_differences(saved, actual):
    differences = {}
    for key in ("Fell", "Gait_Quality_Pass"):
        if csv_bool(saved[key]) != bool(actual[key]):
            differences[key] = {"saved": saved[key], "replayed": actual[key]}
    for key in ("Distance_Traversed", "Forward_Progress", "Walk_Score", "CoT", "Alternating_Steps"):
        expected, observed = float(saved[key]), float(actual[key])
        if not math.isclose(expected, observed, rel_tol=1e-7, abs_tol=1e-8):
            differences[key] = {"saved": expected, "replayed": observed}
    return differences


def output_path_for_rank(base_path, rank, total):
    if total <= 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_rank{rank:02d}{base_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=Path(sweep_config.RESULTS_CSV))
    parser.add_argument("--mesh-cache", type=Path, help="Defaults to the results directory's meshes folder.")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--rank-by", choices=("distance", "cot", "walk_score"), default="distance")
    parser.add_argument("--target-distance", type=float)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--video-out", type=Path, default=Path("data/videos/target_walk_xy.mp4"))
    parser.add_argument("--params-out", type=Path, help="Defaults to the video path with a .json suffix.")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--camera-azimuth", type=float, default=130.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--verify-only", action="store_true", help="Replay and compare metrics without rendering videos.")
    args = parser.parse_args()
    if args.top_n < 1 or args.video_fps <= 0 or min(args.width, args.height) <= 0 or args.tolerance < 0:
        parser.error("Counts/dimensions must be positive and tolerance nonnegative.")
    selected = select_trials(args.results_csv, args.top_n, args.rank_by, args.target_distance, args.tolerance)
    if not selected:
        raise SystemExit("No saved gait passes match the requested selection.")
    cache_dir = (args.mesh_cache or args.results_csv.parent / "meshes").resolve()
    print(f"Selected {len(selected)} saved gait passes ranked by {args.rank_by}.")
    recordings = []
    for rank, row in enumerate(selected, 1):
        geometry, params = saved_trial(row)
        print(f"Rank {rank}: point {row.get('Point_Index', '?')}, saved distance={row['Distance_Traversed']}")
        with tempfile.TemporaryDirectory(prefix="record_saved_walk_") as directory:
            model_path, model_source = resolve_model(geometry, cache_dir, Path(directory))
            ctx = sweep.load_simulation(model_path)
            result, frames = run_trial(ctx, params, video_fps=None if args.verify_only else args.video_fps,
                                       width=args.width, height=args.height, camera_distance=args.camera_distance,
                                       camera_azimuth=args.camera_azimuth, camera_elevation=args.camera_elevation)
            differences = replay_differences(row, result)
            output = output_path_for_rank(args.video_out, rank, len(selected))
            if not args.verify_only:
                export_video(frames, output, args.video_fps)
            print(f"  {'REPLAY MISMATCH: ' + str(differences) if differences else 'Replay matches saved metrics.'}")
            recordings.append({"rank": rank, "point_index": row.get("Point_Index"),
                               "geometry": geometry, "trial_parameters": params, "saved_row": row,
                               "recorded_result": result, "replay_differences": differences,
                               "model_source": model_source,
                               "video_out": None if args.verify_only else str(output)})
    params_out = args.params_out or args.video_out.with_suffix(".json")
    params_out.parent.mkdir(parents=True, exist_ok=True)
    params_out.write_text(json.dumps({"results_csv": str(args.results_csv.resolve()),
                                     "rank_by": args.rank_by, "recordings": recordings}, indent=2))
    print(f"Saved replay report to {params_out}")
    if any(recording["replay_differences"] for recording in recordings):
        raise SystemExit("Saved parameters were replayed, but metrics differ; see the replay report.")


if __name__ == "__main__":
    main()
