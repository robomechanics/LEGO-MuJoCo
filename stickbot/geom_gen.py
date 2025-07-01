import os, subprocess, yaml

def generate_feet_geom(ft_prm: dict, script_dir: str) -> None:
    # make geom dir
    output_dir = script_dir+f"/{ft_prm['file_id']}"
    os.makedirs(output_dir, mode=0o777, exist_ok=True)
    # generate left and right foot geometries
    for left_foot in [1, -1]:
        feet_vars = f"""
            // Ellipsoid diameters
            X = {ft_prm["X"]}; // x span
            Y = {ft_prm["Y"]}; // y span
            Z = {ft_prm["Z"]}; // z span

            // boolean for left/right foot
            left_foot = {left_foot};

            // feet box size
            box_x = {ft_prm["box_x"]};
            box_y = {ft_prm["box_y"]};"""

        with open(script_dir+"/feet_vars.scad", "w") as f:
            f.write(feet_vars)

        prefix = "left_" if left_foot == 1 else "right_"

        # use OpenSCAD to generate the foot stl
        subprocess.run(
            ["openscad", "-o", output_dir+"/"+prefix+"foot_geom.stl", "feet_generator.scad"],
            cwd=script_dir,
            check=True
            )      

        # convert to obj format using meshlabserver
        subprocess.run(
            ["meshlabserver", "-i",  prefix+"foot_geom.stl", "-o",  prefix+"foot_geom.obj"],
            cwd=output_dir,
            )
        
        print(f"Generated {prefix}foot_geom.stl and {prefix}foot_geom.obj in {output_dir}")

    with open(output_dir+"/feet_params.yaml", "w") as f:
        yaml.dump(ft_prm, f, default_flow_style=False)