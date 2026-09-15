#!/usr/bin/env python3
"""Generate one-section Bigfoot feet without changing mass or inertia values.

This version starts from Bigfoot/scene.xml, generates one full-foot STL per side,
and swaps only the mesh referenced by the existing foot geoms. Existing geom
placement, orientation, classes, body inertials, actuator definitions, and geom
mass/density settings are left alone.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from gen_new_xml_v2 import parse_correction_string, quat_multiply
from coordinate_frame import public_to_mujoco_vec


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_ENTRY_XML = REPO_ROOT / "Bigfoot" / "scene.xml"
DEFAULT_OUTPUT_XML = REPO_ROOT / "modified_model.xml"
DEFAULT_OUT_DIR = REPO_ROOT / "foot_section_out_v3"
DEFAULT_SCAD_FILE = REPO_ROOT / "foot_generator_v3.scad"
OPENSCAD_PATH = "/usr/bin/openscad"
VERBOSE = True

# Defaults chosen to match the current baseline sweep geometry.
DEFAULT_CURVE_X = 0.78
DEFAULT_CURVE_Y = 0.936
DEFAULT_CURVE_Z = 0.78
DEFAULT_BOX_X = 0.667
DEFAULT_BOX_Y = 0.24
DEFAULT_BOX_Z = 0.1206
DEFAULT_FN = 110
LEFT_CORRECTION = "z:90"
RIGHT_CORRECTION = "z:90;x:180"

FOOT_GEOMS = {
    "right": ("right_foot_1", "right_foot_1_col"),
    "left": ("left_foot_1", "left_foot_1_col"),
}


def vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def _env_flag(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _absolutize_mesh_paths(spec: mujoco.MjSpec, xml_dir: Path) -> None:
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


def _geom(spec: mujoco.MjSpec, name: str):
    geom = spec.geom(name)
    if geom is None:
        raise ValueError(f"Could not find geom '{name}' in the input model.")
    return geom


def _add_public_offset(geom, offset_xyz: np.ndarray) -> None:
    if np.allclose(offset_xyz, 0.0):
        return
    geom.pos = list(np.asarray(geom.pos, dtype=float) + public_to_mujoco_vec(offset_xyz))


def generate_feet(
    scad_file: Path,
    out_dir: Path,
    curve_x: float,
    curve_y: float,
    curve_z: float,
    box_x: float,
    box_y: float,
    box_z: float,
    fn: int,
) -> dict[str, dict[str, Path]]:
    if box_x <= 0 or box_y <= 0 or box_z <= 0:
        raise ValueError("box_x, box_y, and box_z must be positive.")
    if curve_x <= 0 or curve_y <= 0 or curve_z <= 0:
        raise ValueError("curve_x, curve_y, and curve_z must be positive radii.")
    if box_x / 2 >= curve_x:
        raise ValueError(f"box_x/2 ({box_x / 2:.4f}) must be smaller than curve_x ({curve_x:.4f}).")
    if box_y >= curve_y:
        raise ValueError(f"box_y ({box_y:.4f}) must be smaller than curve_y ({curve_y:.4f}).")
    if box_z >= curve_z:
        raise ValueError(f"box_z ({box_z:.4f}) must be smaller than curve_z ({curve_z:.4f}).")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"right": {}, "left": {}}
    foot_flags = {"right": 0, "left": 1}
    for side, flag in foot_flags.items():
        out_path = out_dir / f"{side}_foot_full.stl"
        command = [
            OPENSCAD_PATH,
            "-D",
            f"curve_x={curve_x}",
            "-D",
            f"curve_y={curve_y}",
            "-D",
            f"curve_z={curve_z}",
            "-D",
            f"box_x={box_x}",
            "-D",
            f"box_y={box_y}",
            "-D",
            f"box_z={box_z}",
            "-D",
            f"fn={fn}",
            "-D",
            f"left_foot={flag}",
            "--export-format",
            "binstl",
            "-o",
            str(out_path),
            str(scad_file),
        ]
        vprint(f"Running: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"OpenSCAD generation failed for {side} foot:\n{result.stderr}")
        if not out_path.exists():
            raise RuntimeError(f"Expected output {out_path} was not created.")
        results[side]["full"] = out_path
    return results


def build_preview_mjcf(sections: dict[str, dict[str, Path]], spacing: float = 0.4) -> str:
    meshdir = next(iter(next(iter(sections.values())).values())).parent.resolve()
    assets = []
    bodies = []
    colors = {"left": "0.9 0.3 0.2 1", "right": "0.2 0.4 0.9 1"}

    for side_i, side in enumerate(["left", "right"]):
        x_off = (side_i - 0.5) * spacing
        mesh_name = f"{side}_full_mesh"
        mat_name = f"{side}_full_mat"
        assets.append(f'<mesh name="{mesh_name}" file="{sections[side]["full"].name}"/>')
        assets.append(f'<material name="{mat_name}" rgba="{colors[side]}"/>')
        bodies.append(
            f'<body name="{side}_foot" pos="{x_off} 0 0.3">\n'
            f'  <geom name="{side}_full_geom" type="mesh" mesh="{mesh_name}" material="{mat_name}"/>\n'
            "</body>"
        )

    return f"""
<mujoco model="foot_v3_preview">
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


def launch_preview(sections: dict[str, dict[str, Path]]) -> None:
    model = mujoco.MjModel.from_xml_string(build_preview_mjcf(sections))
    data = mujoco.MjData(model)
    print("Launching foot preview... red=left, blue=right. Close the window to continue.")
    mujoco.viewer.launch(model, data)


def launch_model_preview(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    print("Launching full robot viewer with v3 feet... close the window to exit.")
    mujoco.viewer.launch(model, data)


def inject_feet_preserving_dynamics(
    entry_xml: Path,
    sections: dict[str, dict[str, Path]],
    output_xml: Path,
    left_offset: np.ndarray | None = None,
    right_offset: np.ndarray | None = None,
) -> mujoco.MjModel:
    """Swap foot mesh assets while preserving original dynamics definitions."""
    reference_model = mujoco.MjModel.from_xml_path(str(entry_xml))
    spec = mujoco.MjSpec.from_file(str(entry_xml))
    _absolutize_mesh_paths(spec, entry_xml.parent)

    offsets = {
        "left": np.zeros(3, dtype=float) if left_offset is None else np.asarray(left_offset, dtype=float),
        "right": np.zeros(3, dtype=float) if right_offset is None else np.asarray(right_offset, dtype=float),
    }
    corrections = {
        "left": parse_correction_string(LEFT_CORRECTION),
        "right": parse_correction_string(RIGHT_CORRECTION),
    }
    desired_compiled_pos = {}

    for side, geom_names in FOOT_GEOMS.items():
        mesh_path = sections[side]["full"].resolve()
        mesh_name = f"{side}_full_mesh"
        spec.add_mesh(name=mesh_name, file=str(mesh_path))
        compiled_offset = public_to_mujoco_vec(offsets[side])

        for geom_name in geom_names:
            geom = _geom(spec, geom_name)
            geom.meshname = mesh_name
            geom.quat = list(quat_multiply(np.asarray(geom.quat, dtype=float), corrections[side]))
            _add_public_offset(geom, offsets[side])
            reference_geom_id = mujoco.mj_name2id(
                reference_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if reference_geom_id == -1:
                raise ValueError(f"Could not find reference geom '{geom_name}' in {entry_xml}.")
            desired_compiled_pos[geom_name] = (
                reference_model.geom_pos[reference_geom_id].copy() + compiled_offset
            )

    model = spec.compile()
    for geom_name, target_pos in desired_compiled_pos.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            raise ValueError(f"Could not find generated geom '{geom_name}' after compile.")
        compiled_delta = target_pos - model.geom_pos[geom_id]
        geom = _geom(spec, geom_name)
        geom.pos = list(np.asarray(geom.pos, dtype=float) + compiled_delta)

    model = spec.compile()
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    output_xml.write_text(spec.to_xml())
    return model


def _body_dynamics_snapshot(model: mujoco.MjModel) -> dict[str, tuple[float, tuple[float, ...], tuple[float, ...]]]:
    snapshot = {}
    for body_name in ("motor", "simplified_motor___arm_rod"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            continue
        snapshot[body_name] = (
            float(model.body_mass[body_id]),
            tuple(float(v) for v in model.body_ipos[body_id]),
            tuple(float(v) for v in model.body_inertia[body_id]),
        )
    return snapshot


def print_dynamics_check(entry_xml: Path, output_xml: Path) -> None:
    before = _body_dynamics_snapshot(mujoco.MjModel.from_xml_path(str(entry_xml)))
    after = _body_dynamics_snapshot(mujoco.MjModel.from_xml_path(str(output_xml)))

    print("Dynamics preservation check:")
    for body_name, before_values in before.items():
        after_values = after.get(body_name)
        unchanged = after_values is not None and all(
            np.allclose(before_item, after_item, rtol=0.0, atol=1e-6)
            for before_item, after_item in zip(before_values, after_values)
        )
        print(f"  {body_name}: {'unchanged' if unchanged else 'CHANGED'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-foot shell meshes and inject them into Bigfoot/scene.xml without changing dynamics."
    )
    parser.add_argument("--entry-xml", type=Path, default=DEFAULT_ENTRY_XML)
    parser.add_argument("--output-xml", type=Path, default=DEFAULT_OUTPUT_XML)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scad-file", type=Path, default=DEFAULT_SCAD_FILE)
    parser.add_argument("--curve-x", type=float, default=DEFAULT_CURVE_X)
    parser.add_argument("--curve-y", type=float, default=DEFAULT_CURVE_Y)
    parser.add_argument("--curve-z", type=float, default=DEFAULT_CURVE_Z)
    parser.add_argument("--box-x", type=float, default=DEFAULT_BOX_X)
    parser.add_argument("--box-y", type=float, default=DEFAULT_BOX_Y)
    parser.add_argument("--box-z", type=float, default=DEFAULT_BOX_Z)
    parser.add_argument("--fn", type=int, default=DEFAULT_FN)
    parser.add_argument("--left-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--right-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--preview-only", action="store_true", help="Generate and preview feet without writing XML.")
    parser.add_argument("--no-preview", action="store_true", help="Skip standalone generated-foot preview.")
    parser.add_argument("--no-model-preview", action="store_true", help="Skip full modified robot preview.")
    parser.add_argument("--skip-dynamics-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections = generate_feet(
        scad_file=args.scad_file,
        out_dir=args.out_dir,
        curve_x=args.curve_x,
        curve_y=args.curve_y,
        curve_z=args.curve_z,
        box_x=args.box_x,
        box_y=args.box_y,
        box_z=args.box_z,
        fn=args.fn,
    )
    if not args.no_preview:
        launch_preview(sections)
    if args.preview_only:
        return

    model = inject_feet_preserving_dynamics(
        entry_xml=args.entry_xml,
        sections=sections,
        output_xml=args.output_xml,
        left_offset=np.asarray(args.left_offset, dtype=float),
        right_offset=np.asarray(args.right_offset, dtype=float),
    )
    print(f"Wrote {args.output_xml}")
    if not args.skip_dynamics_check:
        print_dynamics_check(args.entry_xml, args.output_xml)
    if not args.no_model_preview:
        launch_model_preview(model)


if __name__ == "__main__":
    main()
