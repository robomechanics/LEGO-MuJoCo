import yaml
import os

# full absolute path to this script
script_path = os.path.abspath(__file__)
# directory containing this script
script_dir  = os.path.dirname(script_path)

feet_vars_dict = {
    "X": 0.24,  # x span
    "Y": 0.24,  # y span
    "Z": 0.24,  # z span
    "box_x": 0.101,  # feet box size x
    "box_y": 0.0527,  # feet box size y
}

file_id = '_'.join(f"{key}_{value}" for key, value in feet_vars_dict.items())
feet_vars_dict["file_id"] = str(file_id)

# Write the dictionary to a YAML file
with open(script_dir+"/feet_params.yaml", "w") as f:
    yaml.dump(feet_vars_dict, f, default_flow_style=False)