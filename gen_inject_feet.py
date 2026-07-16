#!/usr/bin/env python3
"""
generate_and_inject_feet.py

Pipeline:
    1. Generate 3-part feet (front/middle/back) via OpenSCAD from ellipsoid +
       bounding-box parameters.
    2. (Optional) Preview the 3-part feet standalone in the MuJoCo viewer.
    3. Inject the generated meshes into an existing robot MJCF, replacing the
       existing right_foot_1/2/3 (+_col) and left_foot_1/2/3 (+_col) geoms,
       reusing each foot's existing pos/quat (read from the XML itself, not
       hardcoded).

Requirements:
    - OpenSCAD installed (path set below via OPENSCAD_PATH)
    - mujoco python package: pip install mujoco
    - feet_generator.scad (patched version with slice_x0/slice_x1 support)

Usage:
    # Preview only, 3-part feet, no robot XML needed:
    python3 gen_inject_feet.py \
        --scad-dir /Users/benmatthews/Downloads \
        --X 0.78 --Y 0.50 --Z 0.705 --box_x 0.667 --box_y 0.24 \
        --preview-only

    # Full pipeline: generate + inject into robot XML + view full robot:
    python3 gen_inject_feet.py \
        --scad-dir /Users/benmatthews/Downloads \
        --robot-xml /Users/benmatthews/Desktop/Work/Research/LEGO-MuJoCo/bigfoot/scene.xml \
        --X 0.78 --Y 0.95 --Z 0.3 --box_x 0.667 --box_y 0.24
# Standard Case: 
    # If front/back sections look swapped in the preview, flip them:
    python3 generate_and_inject_feet.py ... --swap-front-back
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import mujoco
import mujoco.viewer

OPENSCAD_PATH = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"  # <-- adjust if needed

import numpy as np


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


# ---------------------------------------------------------------------------
# Section geometry
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OpenSCAD generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Standalone preview (3 sections per foot, both feet, no robot XML needed)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Injection into the full robot model via mjSpec
# ---------------------------------------------------------------------------

# Maps your existing geom-name suffixes to the section labels we generate.
# Per your description: _1=front, _2=back, _3=middle.
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
    output_xml_path: Path = None,
    left_correction: np.ndarray = None,
    right_correction: np.ndarray = None,
    left_offset: np.ndarray = None,
    right_offset: np.ndarray = None,
    offset_frame: str = "body",
) -> "mujoco.MjModel":
    """
    Loads robot_xml_path, replaces the existing right_/left_foot_{1,2,3}
    (+_col) geoms with newly generated section meshes, reusing each geom's
    original pos/quat -- with an additional per-side correction rotation
    and position offset applied.

    If output_xml_path is given, also writes the modified spec back out to
    XML for inspection (does not overwrite your original file unless you
    pass the same path).
    """
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    zero_offset = np.array([0.0, 0.0, 0.0])
    corrections = {
        "right": right_correction if right_correction is not None else identity_quat,
        "left": left_correction if left_correction is not None else identity_quat,
    }
    # ========================================================
    # Manually Set Offsets:
    # ========================================================
    #Left: [Down/Up (Pos = Down),Forward/Backward, Right/Left (Actual X)]
    #Right: [Left/Right (Pos = Left), Backwards/Forwards (Pos = Backwards), Up/Down (actual Z)]
    left_offset = np.array([0.0, 0.0, 0.113]) #Centered: [0.0, 0.0, 0.113667]
    right_offset = np.array([0.113, 0.0, 0.0]) #Centered: [0.113667, 0.0, 0.0]
    
    
    offsets = {
        "right": right_offset if right_offset is not None else zero_offset,
        "left": left_offset if left_offset is not None else zero_offset,
    }

    for side in ["right", "left"]:
        for suffix, section in SUFFIX_TO_SECTION.items():
            visual_name = f"{side}_foot_{suffix}"
            collision_name = f"{side}_foot_{suffix}_col"

            # Read the existing transform (identical for visual/collision
            # in your XML, so we just use the visual geom's).
            pos, quat = _get_geom_transform(spec, visual_name)
            corrected_quat = quat_multiply(np.array(quat), corrections[side])
            corrected_pos = apply_offset(
                np.array(pos), corrected_quat, offsets[side], offset_frame
            )

            # Find owning body before deleting the geom.
            visual_geom = spec.geom(visual_name)
            body = visual_geom.parent  # owning body of this geom

            old_mesh_name = visual_geom.meshname

            # Delete old visual + collision geoms.
            spec.delete(visual_geom)
            spec.delete(spec.geom(collision_name))

            # Delete old mesh asset (safe to skip if already removed/shared).
            old_mesh = spec.mesh(old_mesh_name)
            if old_mesh is not None:
                spec.delete(old_mesh)

            # Add new mesh asset from the generated .obj.
            # Use an absolute path -- if we pass a relative path here, the
            # compiler's meshdir (inherited from scene.xml/robot.xml) gets
            # prepended to it, which breaks the lookup since our generated
            # files live outside that meshdir.
            new_mesh_path = sections[side][section].resolve()
            new_mesh_name = f"{side}_{section}_mesh"
            spec.add_mesh(name=new_mesh_name, file=str(new_mesh_path))

            # Add new visual + collision geoms with the original position
            # and the corrected orientation.
            body.add_geom(
                name=visual_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=new_mesh_name,
                pos=list(corrected_pos),
                quat=list(corrected_quat),
            )
            body.add_geom(
                name=collision_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=new_mesh_name,
                pos=list(corrected_pos),
                quat=list(corrected_quat),
            )

            print(f"Replaced {visual_name}/{collision_name} -> {new_mesh_name}")

    model = spec.compile()

    if output_xml_path is not None:
        xml_str = spec.to_xml()
        Path(output_xml_path).write_text(xml_str)
        print(f"Wrote modified model XML to {output_xml_path}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-part foot geometry and inject into robot model."
    )
    parser.add_argument("--X", type=float, default=0.24)
    parser.add_argument("--Y", type=float, default=0.24)
    parser.add_argument("--Z", type=float, default=0.24)
    parser.add_argument("--box_x", type=float, default=0.30,
                         help="Total foot length (must be > 0.25 to fit the middle section)")
    parser.add_argument("--box_y", type=float, default=0.0527)
    parser.add_argument("--fn", type=int, default=80)
    parser.add_argument("--scad-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./foot_section_out")
    parser.add_argument("--robot-xml", type=str, default=None,
                         help="Path to the full robot MJCF to inject into. "
                              "Use this only if robot.xml has its own "
                              "<mujoco>/<compiler>/<asset> tags. If robot.xml "
                              "is an include-fragment (starts with <worldbody> "
                              "with no <mujoco> root), use --scene-xml instead.")
    parser.add_argument("--scene-xml", type=str, default=None,
                         help="Path to scene.xml, which presumably wraps "
                              "robot.xml via <include> and supplies the "
                              "compiler/asset/meshdir context, ground plane, "
                              "and lighting. Prefer this over --robot-xml if "
                              "robot.xml is a bare fragment.")
    parser.add_argument("--output-xml", type=str, default=None,
                         help="Optional path to write the modified robot XML to.")
    parser.add_argument("--preview-only", action="store_true",
                         help="Only generate + preview the 3-part feet, skip injection.")
    parser.add_argument("--swap-front-back", action="store_true",
                         help="Flip which end is labeled front/back, if the "
                              "preview shows them reversed.")
    parser.add_argument("--left-correction", type=str, default="z:90",
                         help="Semicolon-separated axis:degrees corrections "
                              "applied to the left foot's local frame before "
                              "the original transform, e.g. 'z:90' or "
                              "'z:90;x:180'. Starting guess based on "
                              "described misalignment -- verify visually.")
    parser.add_argument("--right-correction", type=str, default="z:90;x:180",
                         help="Same as --left-correction, for the right foot.")
    parser.add_argument("--left-offset", type=str, default="0 0 0",
                         help="'dx dy dz' position offset applied to the left foot.")
    parser.add_argument("--right-offset", type=str, default="0 0 0",
                         help="'dx dy dz' position offset applied to the right foot.")
    parser.add_argument("--offset-frame", type=str, default="body",
                         choices=["body", "local"],
                         help="Whether dx/dy/dz above are interpreted in the "
                              "parent body's frame ('body') or rotated into "
                              "the foot's own local frame first ('local').")
    args = parser.parse_args()

    scad_file = Path(args.scad_dir) / "feet_generator.scad"
    if not scad_file.exists():
        sys.exit(f"Could not find {scad_file}")

    out_dir = Path(args.out_dir).resolve()

    sections = generate_all_sections(
        scad_file, out_dir,
        args.X, args.Y, args.Z, args.box_x, args.box_y, args.fn,
        swap_front_back=args.swap_front_back,
    )

    entry_xml = args.scene_xml or args.robot_xml

    if args.preview_only or entry_xml is None:
        launch_preview(sections)
        return

    # Preview first anyway, so you can confirm shape/orientation before
    # committing to the full-robot injection.
    launch_preview(sections)

    output_xml_path = Path(args.output_xml) if args.output_xml else None
    left_correction = parse_correction_string(args.left_correction)
    right_correction = parse_correction_string(args.right_correction)
    left_offset = np.array([float(v) for v in args.left_offset.split()])
    right_offset = np.array([float(v) for v in args.right_offset.split()])
    model = inject_feet_into_model(
        Path(entry_xml), sections, output_xml_path,
        left_correction=left_correction, right_correction=right_correction,
        left_offset=left_offset, right_offset=right_offset,
        offset_frame=args.offset_frame,
    )
    data = mujoco.MjData(model)
    print("Launching full robot viewer with new feet... close the window to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()