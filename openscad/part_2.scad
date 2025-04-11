w = 0.20;
h = 0.667;

intersection() {
    import("../robots/duplo_hip_feet_centered_mjcf/part_2_backup.stl");
    translate([0, 0, -1])
    linear_extrude(height=2) {
        polygon(points=[[0,-h/2], [w,-h/2], [w,h/2], [0, h/2]]);
    }
}
