#!/usr/bin/env python3
"""
Requirements:
    - OpenSCAD installed and on PATH
      (https://openscad.org/downloads.html)
    - mujoco python package: pip install mujoco
    - feet_generator.scad available somewhere on disk

Terminal Command: (Modify path in line 2 for your specific path, this is saved for mine)
    python3 foot_test.py \
        --scad-dir /Users/benmatthews/Downloads \
        --X 0.35 --Y 0.30 --Z 0.30 \
        --box_x 0.40 --box_y 0.125 --fn 80

    # Just generate the .obj files without opening the viewer:
    python3 foot_test.py --scad-dir ./scad --no-viewer
"""

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

OPENSCAD_PATH = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"  # <-- change this if OpenSCAD is not on your PATH

def generate_foot_obj(
    scad_file: Path,
    out_path: Path,
    X: float,
    Y: float,
    Z: float,
    box_x: float,
    box_y: float,
    fn: int,
    left_foot_flag: int,
    verbose: bool = True,
) -> None:
    """Call OpenSCAD to render one foot to an .obj file."""
    command = [
        OPENSCAD_PATH, # <-- replace with your OpenSCAD executable path if not on PATH
        "-D", f"X={X}",
        "-D", f"Y={Y}",
        "-D", f"Z={Z}",
        "-D", f"box_x={box_x}",
        "-D", f"box_y={box_y}",
        "-D", f"fn={fn}",
        "-D", f"left_foot={left_foot_flag}",
        "-o", str(out_path),
        #"--export-format", "obj",
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


def build_mjcf(left_obj: Path, right_obj: Path, spacing: float = 0.3) -> str:
    """
    Build a minimal MJCF string that loads both foot meshes side by side,
    statically positioned (no joints/gravity) purely for visual inspection.
    """
    meshdir = left_obj.parent.resolve()
    xml = f"""
<mujoco model="foot_preview">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <asset>
    <mesh name="left_foot_mesh" file="{left_obj.name}"/>
    <mesh name="right_foot_mesh" file="{right_obj.name}"/>
    <material name="left_mat" rgba="0.2 0.6 0.9 1"/>
    <material name="right_mat" rgba="0.9 0.4 0.2 1"/>
    <material name="grid" rgba="0.85 0.85 0.85 1"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="2 2 0.1" material="grid"/>
    <body name="left_foot" pos="{spacing / 2} 0 0.3">
      <geom name="left_foot_geom" type="mesh" mesh="left_foot_mesh" material="left_mat"/>
    </body>
    <body name="right_foot" pos="{spacing / 2} 0 0.3">
      <geom name="right_foot_geom" type="mesh" mesh="right_foot_mesh" material="right_mat"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def main():
    parser = argparse.ArgumentParser(
        description="Generate and preview foot geometry in the MuJoCo viewer."
    )
    parser.add_argument("--X", type=float, default=0.24, help="Ellipsoid full width (X)")
    parser.add_argument("--Y", type=float, default=0.24, help="Ellipsoid full depth (Y)")
    parser.add_argument("--Z", type=float, default=0.24, help="Ellipsoid full height (Z)")
    parser.add_argument("--box_x", type=float, default=0.101, help="Clipping box full X dimension")
    parser.add_argument("--box_y", type=float, default=0.0527, help="Clipping box full Y dimension")
    parser.add_argument("--fn", type=int, default=100, help="OpenSCAD sphere facet resolution")
    parser.add_argument(
        "--scad-dir", type=str, required=True,
        help="Directory containing feet_generator.scad",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./foot_preview_out",
        help="Where to write generated .obj files",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Only generate the .obj files, skip launching the viewer",
    )
    args = parser.parse_args()

    scad_file = Path(args.scad_dir) / "feet_generator.scad"
    if not scad_file.exists():
        sys.exit(f"Could not find {scad_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_obj = out_dir / "left_foot_preview.obj"
    right_obj = out_dir / "right_foot_preview.obj"

    generate_foot_obj(
        scad_file, left_obj, args.X, args.Y, args.Z,
        args.box_x, args.box_y, args.fn, left_foot_flag=1,
    )
    generate_foot_obj(
        scad_file, right_obj, args.X, args.Y, args.Z,
        args.box_x, args.box_y, args.fn, left_foot_flag=-1,
    )

    if args.no_viewer:
        return

    xml = build_mjcf(left_obj, right_obj)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print("Launching MuJoCo viewer... close the window to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()