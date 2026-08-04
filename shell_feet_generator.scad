// =================================================================
// Parameters
// =================================================================
X = 0.78;
Y = 0.936;
Z = 0.35;
box_x = 0.66675;
box_y = 0.24;
box_height = 0.1206; // Step 3: Upper Z cutoff height
left_foot = 0;     // 0 = Right foot, 1 = Left foot, 2 = Render both
fn = 100;

// Shell wall thickness (same units as X/Y/Z/box_x/box_y/box_height).
// The final part is a hollow shell of this thickness rather than a solid
// chunk. See shell module below for exactly which faces get walled.
shell_thickness = 0.03;

// Optional slicing along local X axis (retained for backward compatibility)
// Also doubles as the front/middle/back section-splitting cut used by
// gen_new_xml.py -- see the shell module notes below for why these bounds
// are intentionally NOT inset by shell_thickness.
slice_x0 = -1000;
slice_x1 = 1000;

slice_center_x = (slice_x0 + slice_x1) / 2;
slice_width_x  = abs(slice_x1 - slice_x0);

// =================================================================
// Pipeline Definition
// =================================================================

// Parameterized version of the original Step 1-5 intersection pipeline.
// Both the outer (full-size) and inner (inset-by-shell_thickness) solids
// are built by calling this same module with different arguments, so the
// underlying ellipsoid-cutout LOGIC is unchanged/retained -- only the
// dimensions fed into it differ between the outer shell and the cavity
// that gets subtracted from it.
//
//   ex, ey, ez : ellipsoid bounding-box dimensions (diameter along each axis)
//   ez_off     : extra Z shift applied only to the ellipsoid, so that its
//                bottom pole can be raised by shell_thickness for the
//                inner cavity (otherwise the bottom of an inner ellipsoid
//                that's merely *resized* smaller would still touch Z=0,
//                giving zero wall thickness at the very bottom of the foot)
//   bx, by     : footprint box bounds. bx is the FULL width (cube spans
//                +/-bx/2); by is a HALF-width (cube spans -by..+by) -- this
//                asymmetric convention matches the original file, see the
//                notes in gen_new_xml.py / the writeup for why this is a
//                bit confusing and worth double-checking.
//   bh         : upper Z cutoff (box_height)
//   sx0, sx1   : local-X slice bounds (front/middle/back section split)
//   ylo        : lower Y bound of the medial half-split (0 for the outer
//                shape, shell_thickness for the inner cavity)
module foot_cut(ex, ey, ez, ez_off, bx, by, bh, sx0, sx1, ylo) {
    intersection() {
        // Steps 1 & 2: Create ellipsoid centered at (0,0), shifted up by
        // ez_off + ez/2 (bottommost point sits at Z = ez_off)
        translate([0, 0, ez_off + ez / 2])
            resize([ex, ey, ez])
                sphere(r = 1, $fn = fn);

        // Step 3: Cut out any geometry above Z = bh
        translate([0, 0, bh / 2])
            cube([ex * 4, ey * 4, bh], center = true);

        // Step 4: Top-down rectangular footprint bounds
        // (x in [-bx/2, bx/2], y in [-by, by] -- see note above re: by
        // being a half-width already)
        translate([0, 0, ez / 2])
            cube([bx, 2 * by, ez * 2], center = true);

        // Retained X-axis slice sub-section (front/middle/back split).
        // sx0/sx1 are passed through unchanged for both the outer solid
        // and the inner cavity -- see shell module notes for why.
        translate([slice_center_x, 0, ez / 2])
            cube([abs(sx1 - sx0), ey * 4, ez * 2], center = true);

        // Step 5 (a): Split along y = ylo by keeping the y >= ylo portion
        translate([0, ylo + ey, ez / 2])
            cube([ex * 4, 2 * ey, ez * 2], center = true);
    }
}

// The original solid pipeline result, retained as-is (useful for
// debugging / comparing against the shelled output).
module single_foot_half_solid() {
    foot_cut(X, Y, Z, 0, box_x, box_y, box_height, slice_x0, slice_x1, 0);
}

// =================================================================
// Shell (hollow) version of the foot section
// =================================================================
//
// Built as: outer_solid - inner_solid, where inner_solid is the exact same
// pipeline run with every true exterior dimension reduced by
// shell_thickness (ellipsoid axes, box_height, the footprint box, and the
// medial y-split), producing a uniform-ish wall of shell_thickness on all
// of those faces (this is the standard OpenSCAD "shrink and subtract"
// shell approximation -- it is exact along each axis-aligned direction but
// not a true constant-offset/Minkowski shell, so wall thickness on the
// doubly-curved part of the ellipsoid can vary slightly from the nominal
// shell_thickness, more so the more X/Y/Z differ from one another).
//
// slice_x0/slice_x1 (the front/middle/back cut) are deliberately passed
// through UNCHANGED to both the outer and inner calls. Because the inner
// solid is already smaller everywhere else, its cross-section at the
// slice planes ends up naturally inset from the outer cross-section by
// about shell_thickness anyway -- so each of the three sections still
// comes out as its own closed, watertight shell with a thin "ring cap" at
// the cut faces, rather than needing (or getting) a separate wall
// specifically at the section-splitting cut. If slice_x0/x1 were also
// inset here, that ring cap would end up thicker than shell_thickness.
module single_foot_half_shell() {
    difference() {
        foot_cut(X, Y, Z, 0,
                 box_x, box_y, box_height,
                 slice_x0, slice_x1, 0);
        foot_cut(X - 2 * shell_thickness, Y - 2 * shell_thickness, Z - 2 * shell_thickness, shell_thickness,
                 box_x - 2 * shell_thickness, box_y - shell_thickness, box_height - shell_thickness,
                 slice_x0, slice_x1, shell_thickness);
    }
}

// =================================================================
// Frame Alignment & Output Selection
// =================================================================

// 1. Shift Z axis down by box_height so the foot sits in [-box_height, 0]
//    matching the legacy coordinate origin.
module single_foot_half_aligned() {
    translate([0, 0, -box_height])
        single_foot_half_shell();
}

// 2. Select side and apply handedness mirroring to match legacy orientation:
//    Left Foot  (left_foot = 1) -> Positive Y (y >= 0)
//    Right Foot (left_foot = 0) -> Negative Y (y <= 0)
if (left_foot == 1) {
    // Left foot
    single_foot_half_aligned();
} else if (left_foot == 2) {
    // Render both feet together
    single_foot_half_aligned();
    mirror([0, 1, 0]) single_foot_half_aligned();
} else {
    // Default: Right foot (Mirrored across Y=0)
    mirror([0, 1, 0]) single_foot_half_aligned();
}
