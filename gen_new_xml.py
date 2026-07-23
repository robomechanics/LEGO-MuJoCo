#!/usr/bin/env python3
"""

Pipeline:
    1. Generate 3-part feet (front/middle/back) via OpenSCAD from ellipsoid +
       bounding-box parameters.
    2. Preview the 3-part feet standalone in the MuJoCo viewer.
    3. Inject the generated meshes into an existing robot MJCF, replacing the
       existing right_foot_1/2/3 (+_col) and left_foot_1/2/3 (+_col) geoms,
       reusing each foot's existing pos/quat (read from the XML itself, not
       hardcoded) plus a per-side rotation correction and position offset.
    4. Optionally save the fully modified robot model to OUTPUT_XML,
        so the saved file can be loaded from a different script/location.

Requirements:
    - OpenSCAD installed (path set below via OPENSCAD_PATH, make sure version is newest)
    - mujoco python package
    - feet_generator.scad
Usage:
    Edit the USER PARAMETERS block below, then just run:
        python3 gen_new_xml.py
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import mujoco
import mujoco.viewer


# ═══════════════════════════════════════════════════════════════════════════
# USER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# ── File paths ────────────────────────────────────────────────────────────
OPENSCAD_PATH = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"  # adjust if needed
SCAD_DIR      = "/Users/benmatthews/Downloads"          # dir containing feet_generator.scad
ENTRY_XML  = "/Users/benmatthews/Desktop/Work/Research/LEGO-MuJoCo/bigfoot/scene.xml"

# ── Foot ellipsoid / footprint geometry ──────────────────────────────────
# Constraint: (BOX_X/X)^2 + (BOX_Y/Y)^2 must be < 1 (footprint must fit
# inside the ellipsoid -- checked automatically, with a clear error if not).
# Constraint: BOX_X must be > 0.25 (the fixed 250mm middle section length).
X     = 0.78       # Current Robot: 0.78
Y     = 0.936      # Current Robot: 0.936
Z     = 0.35       # foot thickness scales ~linearly with Z, use ~.35-.4
BOX_X = 0.667      # total foot length, Current Robot: 0.667
BOX_Y = 0.24       # total foot width, Current Robot: 0.24
FN    = 80         # OpenSCAD sphere facet resolution (higher = smoother, slower)

# Left:  [Down/Up (positive = down), Forward/Backward, Right/Left]
# Right: [Left/Right (positive = left), Backward/Forward, Up/Down]
LEFT_OFFSET  = np.array([0.0, 0.0, 0.113])    # centered reference: [0.0, 0.0, 0.113667]
RIGHT_OFFSET = np.array([0.113, 0.0, 0.0])    # centered reference: [0.113667, 0.0, 0.0]

# ═══════════════════════════════════════════════════════════════════════════
# ADDITIONAL PARAMETERS (Likely do not change)
# ═══════════════════════════════════════════════════════════════════════════
OUT_DIR       = "./foot_section_out"                     # where generated .obj files go
OUTPUT_XML = "/Users/benmatthews/Desktop/Work/Research/LEGO-MuJoCo/modified_model.xml"   # optional: path to also write the modified model XML to disk

# ── Mode flags ────────────────────────────────────────────────────────────
PREVIEW_ONLY    = False   # True: only generate + preview the feet, skip injection
SWAP_FRONT_BACK = False   # True: flip which end is labeled front/back

# CORRECTIONS FOR FEET ORIENTATION
LEFT_CORRECTION  = "z:90"
RIGHT_CORRECTION = "z:90;x:180"
OFFSET_FRAME = "body"


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
    Tokens are applied in order, left to right, in the mesh's local frame
    (i.e. "z:90;x:180" means: first rotate 90deg about local Z, then rotate
    the result 180deg about local X).
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


MIDDLE_SECTION_LENGTH = 0.25  # 250mm, per spec


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

    Middle is centered at local x=0 with fixed length MIDDLE_SECTION_LENGTH.
    Front and back split the remaining length evenly.
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
        # re-flip signs so "front" is still the positive-x-ish label consistently
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
    foot_flags = {"right": -1, "left": 1}  # matches your existing convention

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
        bodies.append(f'<body name="{side}_foot" pos="{x_off} 0 0.3">\n'
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
    print("Launching preview viewer... red=front, green=middle, blue=back. "
          "Close the window to continue.")
    mujoco.viewer.launch(model, data)


# ═══════════════════════════════════════════════════════════════════════════
# Injection into the full robot model via mjSpec
# ═══════════════════════════════════════════════════════════════════════════

# Maps existing geom-name suffixes to the section labels generated.
# Convention Currently: _1=front, _2=back, _3=middle.
SUFFIX_TO_SECTION = {"1": "front", "2": "back", "3": "middle"}


def _get_geom_transform(spec: "mujoco.MjSpec", geom_name: str) -> Tuple[list, list]:
    """Read pos/quat off an existing geom in the spec, by name."""
    geom = spec.geom(geom_name)
    if geom is None:
        raise ValueError(f"Could not find geom '{geom_name}' in the model.")
    return list(geom.pos), list(geom.quat)


def absolutize_all_mesh_paths(spec: "mujoco.MjSpec", xml_dir: Path) -> None:
    """
    Rewrites every mesh asset's file path to an absolute path, resolved
    against the compiler's meshdir setting (itself resolved relative to the
    directory containing the entry XML this spec was loaded from).
    """
    meshdir = spec.compiler.meshdir or ""
    mesh_base_dir = (xml_dir / meshdir) if meshdir else xml_dir

    for mesh in spec.meshes:
        if not mesh.file:
            continue
        mesh_path = Path(mesh.file)
        if not mesh_path.is_absolute():
            mesh_path = (mesh_base_dir / mesh_path).resolve()
        mesh.file = str(mesh_path)
    spec.compiler.meshdir = ""


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
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))
    absolutize_all_mesh_paths(spec, robot_xml_path.parent)

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

            visual_default = spec.find_default("visual")
            collision_default = spec.find_default("collision")

            # Pass visual_default positionally to resolve class inheritance at creation time
            if visual_default is not None:
                new_visual_geom = body.add_geom(visual_default)
            else:
                new_visual_geom = body.add_geom()
                print(f"Warning: no 'visual' default class found; "
                      f"{visual_name} will use the body's childclass instead.")

            new_visual_geom.name = visual_name
            new_visual_geom.type = mujoco.mjtGeom.mjGEOM_MESH
            new_visual_geom.meshname = new_mesh_name
            new_visual_geom.pos = list(corrected_pos)
            new_visual_geom.quat = list(corrected_quat)

            # Pass collision_default positionally to resolve class inheritance at creation time
            if collision_default is not None:
                new_collision_geom = body.add_geom(collision_default)
            else:
                new_collision_geom = body.add_geom()
                print(f"Warning: no 'collision' default class found; "
                      f"{collision_name} will use the body's childclass instead.")

            new_collision_geom.name = collision_name
            new_collision_geom.type = mujoco.mjtGeom.mjGEOM_MESH
            new_collision_geom.meshname = new_mesh_name
            new_collision_geom.pos = list(corrected_pos)
            new_collision_geom.quat = list(corrected_quat)

            print(f"Replaced {visual_name}/{collision_name} -> {new_mesh_name}")

    model = spec.compile()

    if output_xml_path is not None:
        xml_str = spec.to_xml()
        Path(output_xml_path).write_text(xml_str)
        print(f"Wrote modified model XML to {output_xml_path}")

    return model



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

    # Preview first to confirm shape/orientation before
    # committing to the full-robot injection.
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
    data = mujoco.MjData(model)
    print("Launching full robot viewer with new feet... close the window to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()