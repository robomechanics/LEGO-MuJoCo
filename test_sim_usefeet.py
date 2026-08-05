#!/usr/bin/env python3
"""
Pipeline:
    1. Generate 3-part feet (front/middle/back) via OpenSCAD from ellipsoid +
       bounding-box parameters.
    2. Preview the 3-part feet standalone in the MuJoCo viewer.
    3. Inject the generated meshes into the robot MJCF via mjSpec, replacing
       the existing right_foot_1/2/3 (+_col) and left_foot_1/2/3 (+_col)
       geoms, reusing each foot's existing pos/quat plus a per-side rotation
       correction and position offset (applied exactly once, here).
    4. Run the sinusoidal hip trajectory simulation directly on the modified
       (created locally) model with hardware replication parameters.
    5. Plot joint telemetry and roll/pitch/yaw orientation.

Requirements:
    - OpenSCAD installed (path set below via OPENSCAD_PATH, make sure version is newest)
    - mujoco python package
    - feet_generator.scad

Usage:
    Set Parameter manually at beginning of the script, then command (no extra arguments needed):
        mjpython test_sim_modfeet.py
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import mujoco
import mujoco.viewer


# ═══════════════════════════════════════════════════════════════════════════
# USER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# ── File paths (foot generation) ────────────────────────────────────────────
REPO_DIR      = Path(__file__).resolve().parent
OPENSCAD_PATH = "openscad-nightly"                      # snap: `sudo snap install openscad-nightly` (needed for .obj export)
SCAD_DIR      = str(REPO_DIR)                           # dir containing feet_generator.scad
OUT_DIR       = "./foot_section_out"                     # where generated .obj files go

# Point ENTRY_XML at whichever file is the correct load target.
# Use scene.xml if robot.xml is a bare <worldbody> fragment meant to be
# pulled in via <include> (i.e. it has no its own <mujoco>/<compiler>/<asset>
# tags) -- scene.xml supplies that context. Use robot.xml directly only if
# it's a fully standalone, compilable MJCF on its own.
ENTRY_XML  = str(REPO_DIR / "bigfoot" / "scene.xml")
OUTPUT_XML = None   # optional: path to also write the modified model XML to disk

# ── Mode flags ────────────────────────────────────────────────────────────
PREVIEW_ONLY    = False   # True: only generate + preview the feet, skip injection + sim
SWAP_FRONT_BACK = False   # True: flip which end is labeled front/back

# ── Foot ellipsoid / footprint geometry ──────────────────────────────────
# Constraint: (BOX_X/X)^2 + (BOX_Y/Y)^2 must be < 1 (footprint must fit
# inside the ellipsoid -- checked automatically, with a clear error if not).
# Constraint: BOX_X must be > MIDDLE_SECTION_LENGTH.
X     = 0.78       #Current Robot: 0.78
Y     = 0.936      # Current Robot: 0.936
Z     = 0.35       # foot thickness scales ~linearly with Z, Use ~.35-.4
BOX_X = 0.667      # total foot length, Current Robot: 0.667
BOX_Y = 0.24       # total foot width, Current Robot: 0.24
FN    = 80         # OpenSCAD sphere facet resolution (higher = smoother, slower)
MIDDLE_SECTION_LENGTH = 0.25  # 250mm (fixed length of the middle foot part)
RIGHT_OFFSET = np.array([0.07, -0.0105, 0.0])
LEFT_OFFSET  = np.array([0.0, 0.0105, 0.07])


# ── Motor / control -- match these directly to motorwave.py for hardware
#    replication ─────────────────────────────────────────────────────────
KP           = 42.0
KD           = 6.7
TORQUE_LIMIT = 25.0       # Nm -- Torque Limit for AK80-8

# ── Trajectory ──────────────────────────────────────────────────────────────
HIP_OMEGA       = 0.57 * 2 * np.pi   # natural freq should be 0.52 Hz
LEG_AMP_DEG     = 43.46
# LEG_AMP_DEG     = 0
T_WAIT          = 3.0
START_FREQ_MULT = 1.17
START_AMP_MULT  = 1.58

# ── Per-foot rotation corrections ────────────────────────────────────────
# Semicolon-separated "axis:degrees" tokens, applied (in order, left to
# right) to the mesh's local frame before the original geom transform.
LEFT_CORRECTION  = "z:90"
RIGHT_CORRECTION = "z:90;x:180"

# ── Per-foot position offsets (applied exactly once, during injection) ────
# OFFSET_FRAME = "body":  dx/dy/dz applied directly in the parent body's axes.
# OFFSET_FRAME = "local": dx/dy/dz applied in the foot's own (rotated) axes.
OFFSET_FRAME = "body"

# ── Gain ramp ───────────────────────────────────────────────────────────────
USE_RAMP  = False
RAMP_TIME = 2.0

# ── CAN latency simulation ───────────────────────────────────────────────────
CMD_DELAY_STEPS = 1

# ── Output files ──────────────────────────────────────────────────────────────
TELEMETRY_PLOT_FILE   = 'results/joint_telemetry_plot.png'
ORIENTATION_PLOT_FILE = 'results/orientation_telemetry_plot.png'

# ── Derived constants ─────────────────────────────────────────────────────
leg_amp_rad = np.deg2rad(LEG_AMP_DEG)

# ═══════════════════════════════════════════════════════════════════════════
# Quaternion / offset utilities
# ═══════════════════════════════════════════════════════════════════════════

def quat_from_axis_angle(axis, angle_deg: float) -> np.ndarray:
    """Unit quaternion (w,x,y,z) for a rotation of angle_deg about axis."""
    angle = np.radians(angle_deg)
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, *xyz])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2, both in (w,x,y,z) order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


_AXIS_MAP = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}


def parse_correction_string(s: str) -> np.ndarray:
    """
    Parses e.g. "z:90" or "z:90;x:180" into a single composed quaternion.
    Tokens are applied in order, left to right, in the mesh's local frame.
    """
    composed = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    if not s.strip():
        return composed
    for token in s.split(";"):
        axis_letter, deg_str = token.split(":")
        axis = _AXIS_MAP[axis_letter.strip().lower()]
        q = quat_from_axis_angle(axis, float(deg_str))
        composed = quat_multiply(composed, q)
    return composed


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w,x,y,z) unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def apply_offset(pos: np.ndarray, quat: np.ndarray, offset: np.ndarray, frame: str) -> np.ndarray:
    """Applies a position offset either in body frame (direct add) or in the
    mesh's local frame (rotated by quat first)."""
    if frame == "body":
        return pos + offset
    elif frame == "local":
        return pos + quat_to_rotmat(quat) @ offset
    else:
        raise ValueError(f"Unknown frame '{frame}', expected 'body' or 'local'.")


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Vectorized (N,4) quat[w,x,y,z] -> (N,3) [roll, pitch, yaw] in radians."""
    q = np.atleast_2d(quat_wxyz)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack([roll, pitch, yaw])


def validate_ellipsoid_box_fit(X: float, Y: float, Z: float, box_x: float, box_y: float) -> None:
    """
    Replicates the z_top math from feet_generator.scad in Python so we can
    raise a clear error *before* calling OpenSCAD, instead of a cryptic
    'nan'/'Current top level object is empty' failure from the subprocess.
    """
    a, b, c = X / 2, Y / 2, Z / 2
    hx, hy = box_x / 2, box_y / 2

    ratio = (hx * hx) / (a * a) + (hy * hy) / (b * b)
    if ratio >= 1:
        raise ValueError(
            f"Invalid parameters: the clipping box's corner (hx={hx:.4f}, "
            f"hy={hy:.4f}) lies outside the ellipsoid (a={a:.4f}, b={b:.4f}).\n"
            f"  (hx/a)^2 + (hy/b)^2 = {ratio:.4f}, but this must be < 1.\n"
            f"Fix: increase X and/or Y (the ellipsoid dimensions) relative to "
            f"box_x/box_y (the footprint), or decrease box_x/box_y."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Section geometry
# ═══════════════════════════════════════════════════════════════════════════

def compute_section_bounds(box_x: float, swap_front_back: bool = False) -> dict:
    """
    Given total foot length (box_x), compute local-X slice bounds for the
    front, back, and middle sections.
    """
    if box_x <= MIDDLE_SECTION_LENGTH:
        raise ValueError(
            f"box_x ({box_x}) must be greater than the fixed middle section "
            f"length ({MIDDLE_SECTION_LENGTH}) to leave room for front/back."
        )

    hx = box_x / 2
    half_mid = MIDDLE_SECTION_LENGTH / 2

    back_bounds = (-hx, -half_mid)
    mid_bounds = (-half_mid, half_mid)
    front_bounds = (half_mid, hx)

    if swap_front_back:
        front_bounds, back_bounds = back_bounds, front_bounds
        front_bounds = (-front_bounds[1], -front_bounds[0])
        back_bounds = (-back_bounds[1], -back_bounds[0])

    return {"front": front_bounds, "back": back_bounds, "middle": mid_bounds}


# ═══════════════════════════════════════════════════════════════════════════
# OpenSCAD generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_foot_section_obj(
    scad_file: Path,
    out_path: Path,
    X: float,
    Y: float,
    Z: float,
    box_x: float,
    box_y: float,
    fn: int,
    left_foot_flag: int,
    slice_x0: float,
    slice_x1: float,
    verbose: bool = True,
) -> None:
    """Call OpenSCAD to render one foot *section* to an .obj file."""
    command = [
        OPENSCAD_PATH,
        "-D", f"X={X}",
        "-D", f"Y={Y}",
        "-D", f"Z={Z}",
        "-D", f"box_x={box_x}",
        "-D", f"box_y={box_y}",
        "-D", f"fn={fn}",
        "-D", f"left_foot={left_foot_flag}",
        "-D", f"slice_x0={slice_x0}",
        "-D", f"slice_x1={slice_x1}",
        "-o", str(out_path),
        str(scad_file),
    ]
    if verbose:
        print(f"Running: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"OpenSCAD failed for {out_path.name}:\n{result.stderr}")
        raise RuntimeError(f"OpenSCAD generation failed: {result.stderr}")
    if not out_path.exists():
        raise RuntimeError(f"Expected output {out_path} was not created.")

    print(f"Generated {out_path}")


def generate_all_sections(
    scad_file: Path,
    out_dir: Path,
    X: float, Y: float, Z: float,
    box_x: float, box_y: float, fn: int,
    swap_front_back: bool = False,
) -> dict:
    """
    Generates front/middle/back sections for both feet.
    Returns dict: {"right": {"front": path, "back": path, "middle": path},
                    "left":  {...}}
    """
    bounds = compute_section_bounds(box_x, swap_front_back)
    validate_ellipsoid_box_fit(X, Y, Z, box_x, box_y)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"right": {}, "left": {}}
    foot_flags = {"right": -1, "left": 1}

    for side, flag in foot_flags.items():
        for section, (x0, x1) in bounds.items():
            out_path = out_dir / f"{side}_foot_{section}.obj"
            generate_foot_section_obj(
                scad_file, out_path, X, Y, Z, box_x, box_y, fn,
                left_foot_flag=flag, slice_x0=x0, slice_x1=x1,
            )
            results[side][section] = out_path

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Standalone preview (3 sections per foot, both feet, no robot XML needed)
# ═══════════════════════════════════════════════════════════════════════════

def build_preview_mjcf(sections: dict, spacing: float = 0.4) -> str:
    meshdir = next(iter(next(iter(sections.values())).values())).parent.resolve()
    section_colors = {
        "front": "0.9 0.3 0.2 1",
        "middle": "0.3 0.9 0.3 1",
        "back": "0.2 0.4 0.9 1",
    }

    assets = []
    bodies = []
    for side_i, side in enumerate(["left", "right"]):
        x_off = (side_i - 0.5) * spacing
        geoms = []
        for section, path in sections[side].items():
            mesh_name = f"{side}_{section}_mesh"
            mat_name = f"{side}_{section}_mat"
            assets.append(f'<mesh name="{mesh_name}" file="{path.name}"/>')
            assets.append(f'<material name="{mat_name}" rgba="{section_colors[section]}"/>')
            geoms.append(
                f'<geom name="{side}_{section}_geom" type="mesh" '
                f'mesh="{mesh_name}" material="{mat_name}"/>'
            )
        bodies.append(f'<body name="{side}_foot" pos="{x_off} 0 {Z}">\n'
                       + "\n".join(geoms) + "\n</body>")

    xml = f"""
<mujoco model="foot_section_preview">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <asset>
    {chr(10).join(assets)}
    <material name="grid" rgba="0.85 0.85 0.85 1"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="2 2 0.1" material="grid"/>
    {chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    return xml


def launch_preview(sections: dict) -> None:
    xml = build_preview_mjcf(sections)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("Launching preview viewer (passive)... red=front, green=middle, blue=back. "
          "Close the window to continue.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Injection into the full robot model via mjSpec
# ═══════════════════════════════════════════════════════════════════════════

# Maps existing geom-name suffixes to the section labels generated.
SUFFIX_TO_SECTION = {"1": "front", "2": "back", "3": "middle"}


def _get_geom_transform(spec: "mujoco.MjSpec", geom_name: str) -> Tuple[list, list]:
    """Read pos/quat off an existing geom in the spec, by name."""
    geom = spec.geom(geom_name)
    if geom is None:
        raise ValueError(f"Could not find geom '{geom_name}' in the model.")
    return list(geom.pos), list(geom.quat)


def inject_feet_into_model(
    robot_xml_path: Path,
    sections: dict,
    output_xml_path: Path,
    left_correction: np.ndarray,
    right_correction: np.ndarray,
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    offset_frame: str,
) -> "mujoco.MjModel":
    """
    Loads robot_xml_path, replaces the existing right_/left_foot_{1,2,3}
    (+_col) geoms with newly generated section meshes, reusing each geom's
    original pos/quat -- with an additional per-side correction rotation
    and position offset applied.

    Visual geoms use the generated mesh directly. Collision is approximated
    with a grid of small spheres sampled from the mesh's bottom (sole)
    surface due to issues with mesh contact with scene floor.
    During Trial, press 3 to see the spheres.
    """
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    corrections = {"right": right_correction, "left": left_correction}
    offsets = {"right": right_offset, "left": left_offset}

    for side in ["right", "left"]:
        for suffix, section in SUFFIX_TO_SECTION.items():
            visual_name = f"{side}_foot_{suffix}"
            collision_name = f"{side}_foot_{suffix}_col"

            pos, quat = _get_geom_transform(spec, visual_name)
            corrected_quat = quat_multiply(np.array(quat), corrections[side])
            corrected_pos = apply_offset(
                np.array(pos), corrected_quat, offsets[side], offset_frame
            )

            visual_geom = spec.geom(visual_name)
            body = visual_geom.parent

            old_mesh_name = visual_geom.meshname

            spec.delete(visual_geom)
            spec.delete(spec.geom(collision_name))

            old_mesh = spec.mesh(old_mesh_name)
            if old_mesh is not None:
                spec.delete(old_mesh)

            new_mesh_path = sections[side][section].resolve()
            new_mesh_name = f"{side}_{section}_mesh"
            spec.add_mesh(name=new_mesh_name, file=str(new_mesh_path))

            body.add_geom(
                name=visual_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=new_mesh_name,
                pos=list(corrected_pos),
                quat=list(corrected_quat),
            )

            # Approximate collision with a grid of spheres sampled from the
            # mesh's bottom surface.
            sphere_radius = 0.005  # 5mm sphere radius

            local_vertices = []
            if new_mesh_path.exists():
                with open(new_mesh_path, "r") as obj_file:
                    for line in obj_file:
                        if line.startswith("v "):
                            parts = line.split()
                            local_vertices.append(
                                [float(parts[1]), float(parts[2]), float(parts[3])]
                            )

            # Bin vertices into an XY grid, keeping only the lowest-Z point
            # per bin (the sole surface), for a roughly even sphere spacing.
            grid_resolution = 0.008  # ~8mm spacing
            binned_points = {}
            for vx, vy, vz in local_vertices:
                bin_key = (round(vx / grid_resolution), round(vy / grid_resolution))
                if bin_key not in binned_points or vz < binned_points[bin_key][2]:
                    binned_points[bin_key] = [vx, vy, vz]

            s_idx = 0
            for local_sphere_pos in binned_points.values():
                global_sphere_pos = corrected_pos + quat_to_rotmat(corrected_quat) @ np.array(local_sphere_pos)

                s_geom = body.add_geom()
                s_geom.name = f"{collision_name}_sphere_{s_idx}"
                s_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                s_geom.size = [sphere_radius, 0.0, 0.0]
                s_geom.pos = list(global_sphere_pos)
                s_geom.group = 3

                s_idx += 1

            print(f"Replaced {visual_name}/{collision_name} -> {new_mesh_name} "
                  f"({s_idx} collision spheres)")

    model = spec.compile()

    if output_xml_path is not None:
        xml_str = spec.to_xml()
        Path(output_xml_path).write_text(xml_str)
        print(f"Wrote modified model XML to {output_xml_path}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# TRAJECTORY -- direct port of motorwave.py calculate_sine_reference
# ═══════════════════════════════════════════════════════════════════════════

def calculate_sine_reference(t):
    w1           = HIP_OMEGA * START_FREQ_MULT
    w2           = HIP_OMEGA
    t0           = T_WAIT
    At           = START_AMP_MULT * leg_amp_rad
    As           = leg_amp_rad
    t_transition = t0 + np.pi / w1

    if t <= t0:
        position, velocity = 0.0, 0.0

    elif t < t_transition:
        phase    = w1 * (t - t0)
        position = At * np.sin(phase)
        velocity = At * w1 * np.cos(phase)

    else:
        phase    = w2 * (t - t_transition)
        position = -As * np.sin(phase)
        velocity = -As * w2 * np.cos(phase)

    return position, velocity


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    scad_file = Path(SCAD_DIR) / "feet_generator.scad"
    if not scad_file.exists():
        sys.exit(f"Could not find {scad_file}")

    out_dir = Path(OUT_DIR).resolve()

    sections = generate_all_sections(
        scad_file, out_dir,
        X, Y, Z, BOX_X, BOX_Y, FN,
        swap_front_back=SWAP_FRONT_BACK,
    )

    if PREVIEW_ONLY or ENTRY_XML is None:
        launch_preview(sections)
        return

    # Preview first anyway, to confirm shape/orientation before
    # committing to the full-robot injection + simulation.
    launch_preview(sections)

    output_xml_path = Path(OUTPUT_XML) if OUTPUT_XML else None
    left_correction = parse_correction_string(LEFT_CORRECTION)
    right_correction = parse_correction_string(RIGHT_CORRECTION)
    model = inject_feet_into_model(
        Path(ENTRY_XML), sections, output_xml_path,
        left_correction=left_correction, right_correction=right_correction,
        left_offset=LEFT_OFFSET, right_offset=RIGHT_OFFSET,
        offset_frame=OFFSET_FRAME,
    )

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION -- runs on the model with the newly generated feet
    # ─────────────────────────────────────────────────────────────────────
    data = mujoco.MjData(model)

    motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "motor")
    print(f"Whole-robot CoM: {data.subtree_com[motor_id].round(4)}")
    print(f"Total mass:      {sum(model.body_mass[i] for i in range(model.nbody)):.3f} kg")

    cmd_buffer = [0.0] * CMD_DELAY_STEPS

    for i in range(model.njnt):
        name = model.joint(i).name
        adr  = model.jnt_qposadr[i]
        dof  = model.jnt_dofadr[i]
        print(f"Joint '{name}': qpos[{adr}], qvel[{dof}]")

    hip_joint = model.joint("hip")
    hip_qpos_adr = hip_joint.qposadr[0]
    hip_qvel_adr = hip_joint.dofadr[0]

    time_history = []
    torqueActual_history = []
    torqueCommand_history = []
    positionActual_history = []
    velocityActual_history = []
    targetPosition_history = []
    targetVelocity_history = []
    quat_history = []

    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)

    print("Launching robot viewer with new feet... close the window to end the run.")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            t = data.time

            target_pos_rad, target_vel_rad = calculate_sine_reference(t)

            current_pos = data.qpos[hip_qpos_adr]
            current_vel = data.qvel[hip_qvel_adr]

            time_history.append(t)
            torqueActual_history.append(data.ctrl[0])
            positionActual_history.append(current_pos)
            velocityActual_history.append(current_vel)
            targetPosition_history.append(target_pos_rad)
            targetVelocity_history.append(target_vel_rad)
            quat_history.append(data.xquat[motor_id].copy())

            ramp      = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
            Kp_ramped = KP * ramp
            Kd_ramped = KD * ramp

            tau = (Kp_ramped * (target_pos_rad - current_pos) +
                   Kd_ramped * (target_vel_rad - current_vel))
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

            cmd_buffer.append(tau)
            data.ctrl[0] = cmd_buffer.pop(0)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    # ─────────────────────────────────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────────────────────────────────
    print("Viewer closed. Generating Matplotlib plots...")

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    time_history_arr = np.array(time_history)
    torqueActual_history_arr = np.array(torqueActual_history)
    positionActual_history_arr = np.array(positionActual_history)
    velocityActual_history_arr = np.array(velocityActual_history)
    targetPosition_history_arr = np.array(targetPosition_history)
    targetVelocity_history_arr = np.array(targetVelocity_history)
    quat_history_arr = np.array(quat_history)
    rpy_history = np.rad2deg(quat_to_rpy(quat_history_arr))
    roll_history, pitch_history, yaw_history = rpy_history[:, 0], rpy_history[:, 1], rpy_history[:, 2]

    Path("results").mkdir(exist_ok=True)

    # ── Joint telemetry plot ────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 7))

    line1 = ax1.plot(time_history_arr, positionActual_history_arr, color='#1f77b4', linewidth=2, label='Actual Joint Position')
    line2 = ax1.plot(time_history_arr, velocityActual_history_arr, color='#1f77b4', linewidth=2, label='Actual Joint Velocity', linestyle='--', alpha=0.7)
    line3 = ax1.plot(time_history_arr, targetPosition_history_arr, color='#2ca02c', linewidth=2, label='Target Joint Position', linestyle='-.', alpha=0.9)
    line4 = ax1.plot(time_history_arr, targetVelocity_history_arr, color='#2ca02c', linewidth=2, label='Target Joint Velocity', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Position / Velocity (rad, rad/s)', fontsize=11, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_title('Joint Control Performance & Telemetry', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    line5 = ax2.plot(time_history_arr, torqueActual_history_arr, color='#ff7f0e', linewidth=1.5, alpha=0.9, label='Actual Torque')

    ax2.set_ylabel('Torque (Nm)', fontsize=11, fontweight='bold', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.grid(False)

    lines = line1 + line2 + line5 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(TELEMETRY_PLOT_FILE, dpi=300)
    print(f"Plot successfully saved as '{TELEMETRY_PLOT_FILE}'")

    # ── Roll / Pitch / Yaw plot (motor body) ────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(11, 7))
    # Roll and Pitch swapped to match true orientation in viewer
    ax3.plot(time_history_arr, roll_history,  color='#d62728', linewidth=2, label='Pitch (motor body)')
    ax3.plot(time_history_arr, pitch_history, color='#9467bd', linewidth=2, label='Roll (motor body)')
    ax3.plot(time_history_arr, yaw_history,   color='#17becf', linewidth=2, label='Yaw (motor body)')

    ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
    ax3.set_title('Motor Body Orientation — Roll / Pitch / Yaw', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(ORIENTATION_PLOT_FILE, dpi=300)
    plt.close(fig2)
    print(f"Plot successfully saved as '{ORIENTATION_PLOT_FILE}'")

    print("Run complete.")


if __name__ == "__main__":
    main()