#!/usr/bin/env python3
"""
Headless driver for the Y-parameter foot sweeps: reuses the generation/
injection functions from gen_new_xml.py directly (no viewer calls), producing
one modified_model_<label>.xml per Y value.
"""
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
import gen_new_xml as g

X = 0.78
Y_BASE = 0.936
Z = 0.35
BOX_X = 0.667
BOX_Y = 0.24
FN = 80

SWEEPS = [("y5pct", 1.05), ("y10pct", 1.10), ("y15pct", 1.15)]

scad_file = REPO_DIR / "feet_generator.scad"
entry_xml = REPO_DIR / "bigfoot" / "scene.xml"

left_correction = g.parse_correction_string(g.LEFT_CORRECTION)
right_correction = g.parse_correction_string(g.RIGHT_CORRECTION)

for label, mult in SWEEPS:
    Y = Y_BASE * mult
    print(f"\n=== {label}: Y={Y:.4f} ({(mult - 1) * 100:.0f}% bigger) ===")

    out_dir = REPO_DIR / f"foot_section_out_{label}"
    sections = g.generate_all_sections(
        scad_file, out_dir, X, Y, Z, BOX_X, BOX_Y, FN,
    )

    output_xml = REPO_DIR / f"modified_model_{label}.xml"
    g.inject_feet_into_model(
        entry_xml, sections, output_xml,
        left_correction=left_correction, right_correction=right_correction,
        left_offset=g.LEFT_OFFSET, right_offset=g.RIGHT_OFFSET,
        offset_frame=g.OFFSET_FRAME,
    )
    print(f"Wrote {output_xml}")

print("\nAll sweeps generated.")
