// Variables X, Y, Z, box_x, box_y, left_foot, and fn are now passed via command line.
// slice_x0/slice_x1 optionally passed to cut the foot into a sub-section along
// the local X axis (same axis box_x measures). Defaults span far beyond any
// realistic foot, so omitting them reproduces the original full-foot behavior
// exactly (backward compatible with existing callers).
X=0; Y=0; Z=0; box_x=0; box_y=0; left_foot=0; fn=0;
slice_x0=-1000; slice_x1=1000;

// Semi-axes
a = X/2;
b = Y/2;
c = Z/2;
hx = box_x/2;
hy = box_y/2;
// z_height of foot determined by square intersection
z_top = c * sqrt(1 - (hx*hx)/(a*a) - (hy*hy)/(b*b));
// Box runs from z = -c up to z_top
box_z0 = -c;
box_height = c - z_top;

// Generous y/z extents on the slicing cube so it only clips in X.
slice_center_x = (slice_x0 + slice_x1) / 2;
slice_width_x  = abs(slice_x1 - slice_x0);

translate([0,0,c - box_height])
intersection() {
  // Original ellipsoid-cut-by-box foot shape
  resize([X, Y, Z]) sphere(r = 1, $fn = fn);
  translate([ 0, left_foot * hy, box_height/2 - c])
    cube([ box_x, box_y, box_height ], center = true);
  // New: optional X-slice to isolate front/middle/back section
  translate([ slice_center_x, 0, 0 ])
    cube([ slice_width_x, box_y*4, box_height*4 ], center = true);
}
