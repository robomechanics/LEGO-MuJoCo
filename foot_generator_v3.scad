// =================================================================
// Bigfoot v3 single-piece foot generator
// =================================================================
//
// Coordinate convention:
//   +X = forward walking direction
//   +Y = robot-left / lateral
//   +Z = local vertical in the generated mesh
//
// curve_x/curve_y/curve_z are ellipsoid radii, not diameters.
// box_x is the centered forward footprint width: x in [-box_x/2, box_x/2].
// box_y is the per-side lateral footprint width:
//   left foot:  y in [0, box_y]
//   right foot: y in [-box_y, 0]
// box_z is the retained vertical slice height above the ellipsoid bottom.
//
// The ellipsoid bottom is placed at z=0, then the generated cap keeps
// z in [0, box_z] and is shifted down so the final top plane is z=0 and
// the curved bottom reaches down toward z=-box_z. For fixed box_z,
// increasing curve_z makes the retained ellipsoid slice flatter.

curve_x = 0.936;
curve_y = 0.78;
curve_z = 0.35;
box_x = 0.667;
box_y = 0.24;
box_z = 0.1206;
left_foot = 0; // 0 = right, 1 = left, 2 = both
fn = 110;

module ellipsoid_with_bottom_at_zero() {
    translate([0, 0, curve_z])
        scale([curve_x, curve_y, curve_z])
            sphere(r = 1, $fn = fn);
}

module half_foot_cap(side) {
    y_center = side == 1 ? box_y / 2 : -box_y / 2;

    translate([0, 0, -box_z])
        intersection() {
            ellipsoid_with_bottom_at_zero();

            // Centered forward bounds and one lateral cut through y=0.
            translate([0, y_center, box_z / 2])
                cube([box_x, box_y, box_z], center = true);
        }
}

if (left_foot == 1) {
    half_foot_cap(1);
} else if (left_foot == 2) {
    half_foot_cap(1);
    half_foot_cap(0);
} else {
    half_foot_cap(0);
}
