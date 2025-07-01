include<feet_vars.scad>

// Semi-axes
a = X/2; 
b = Y/2; 
c = Z/2;

hx = box_x/2;
hy = box_y/2;

// z_height of foot determined by sqaure intersection
z_top = c * sqrt(1 - (hx*hx)/(a*a) - (box_y*box_y)/(b*b));
// Box runs from z = -c up to z_top
box_z0 = -c;
box_height = c - z_top;

translate([0,0,c - box_height])
// Intersection
intersection() {
  resize([X, Y, Z]) sphere(r = 1, $fn = 100);
  translate([ 0, left_foot * hy, box_height/2 - c])
    cube([ box_x, box_y, box_height ], center = true);
}