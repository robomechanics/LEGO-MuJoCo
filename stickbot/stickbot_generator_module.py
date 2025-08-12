import os
import xml.etree.ElementTree as ET
from xml_helper import *
from geom_gen import generate_feet_geom

# full absolute path to this script
script_path = os.path.abspath(__file__)
# directory containing this script
script_dir  = os.path.dirname(script_path)

# dict to store design params
params_og = {
    # rod cross-section size
    's'     : 0.005,
    # motor geom dimensions
    'mot_x' : 0.02, 'mot_y' : 0.01, 'mot_z' : 0.02,
    # gap between feets
    'gap_ft': 0.032,
    # width between leg and arm
    'w_arm' : 0.0625,
    # length of arm and leg links
    'l_arm' : 0.104, 'l_leg' : 0.153,
    # hand cube size
    's_hand' : 0.02,
    # hip offset
    'hip_offset' : -0.014,
    # masses 
    'leg_mass' : 0.1, 'feet_mass' : 0.1, 'hand_mass' : 0.1,
    # feet geom params
    'feet_vars_dict' : {
        # Ellipsoid diameters
        "X": 0.24, "Y": 0.24, "Z": 0.24,
        # feet box dimensions
        "box_x": 0.101, "box_y": 0.0527,
    }
}

params = {}

import numpy as np

for i in range(1, 6):

    for key, value in params_og.items():
        if key in ['s', 's_hand']:
            params[key] = value
        if isinstance(value, dict):
            prm_dic = value.copy()
            params[key] = {}
            for prm_key, prm_value in prm_dic.items():
                params[key][prm_key] = np.round(prm_value * (1 + (0.1 * np.random.rand() - 0.05)), 4)
        else:
            params[key] = np.round(value * (1 + (0.1 * np.random.rand() - 0.05)), 4)

    ft_prm = params['feet_vars_dict']
    # generate file_id from feet_vars_dict
    params['file_id'] = '_'.join(f"{key}_{value}" for key, value in ft_prm.items())
    ft_prm['file_id'] = params['file_id']

    left_color = "1 0 0 0.5"
    right_color = "0 0 1 0.5"
    mass_color = "0 1 0 0.5"
    s = params['s']
    s_hand = params['s_hand']

    # generate feet geometries
    generate_feet_geom(ft_prm, script_dir)

    robot = ET.Element('robot', name='stick_bot')
    left_leg = ET.SubElement(robot, 'link', name='left_leg')
    right_leg = ET.SubElement(robot, 'link', name='right_leg')
    # dict to loop through when building urdf
    side_dict = {'left': left_leg, 'right': right_leg}

    # dict to store config of robot from params
    comp_config = {
        'leg_motor': {
            'xyz': [0, params['mot_y']/2, 0],
            'size': [params['mot_x'], params['mot_y'], params['mot_z']]
        },
        'leg_axel': {
            'xyz': [0, params['gap_ft']/4, 0],
            'size': [s, params['gap_ft']/2, s]
        },
        'leg_arm_axel': {
            'xyz': [0, -params['w_arm']/2, 0],
            'size': [s, params['w_arm'], s]
        },
        'leg_arm': {
            'xyz': [0, -params['w_arm'], -params['l_arm']/2],
            'size': [s, s, params['l_arm']]
        },
        'leg_hand_mass': {
            'xyz': [0, -params['w_arm'], -params['l_arm']],
            'size': [s_hand, s_hand, s_hand],
            'mass': params['hand_mass']
        },
        'leg_link': {
            'xyz': [0, params['gap_ft']/2, -params['l_leg']/2],
            'size': [s, s, params['l_leg']]
        },
        'leg_mass': {
            'xyz': [0, params['gap_ft']/2, -params['l_leg']/2],
            'size': [s_hand/2, s_hand/2, s_hand/2],
            'mass': params['leg_mass']
        }
    }

    mass_links_parents = {}
    urdf_prefix = "file:///"

    # loop through left and right sides
    for side, link in side_dict.items():
        # loop through components in dict (all links except the foot)
        for comp_name, comp_params in comp_config.items():
            # Adjust the y-coordinate for left and right sides
            y_val = comp_params['xyz'][1] if side == 'left' else -comp_params['xyz'][1]
            color = left_color if side == 'left' else right_color
            color = mass_color if 'mass' in comp_name else color
            # Add the visual elements (box for links)
            add_box_visual(link, f"{side}_{comp_name}",
                        xyz=f"{comp_params['xyz'][0]} {y_val} {comp_params['xyz'][2]}",
                        size=f"{comp_params['size'][0]} {comp_params['size'][1]} {comp_params['size'][2]}",
                        color=color)
            # Add links for the mass elements (legs and hands)
            if 'mass' in comp_name:
                link_name = f"{side}_{comp_name}"
                mass_link_name = f"{link_name}_mass_link"
                mass_link = ET.SubElement(robot, 'link', name=mass_link_name)
                mass_links_parents[mass_link_name] = link.get('name')
                inertial = ET.SubElement(mass_link, 'inertial', name=f"{link_name}_inertial")
                ET.SubElement(inertial, 'origin', xyz=f"{comp_params['xyz'][0]} {y_val} {comp_params['xyz'][2]}", rpy="0 0 0")
                ET.SubElement(inertial, 'mass', value=f"{comp_params['mass']}")
                ET.SubElement(inertial, 'inertia', ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")

        # Add the foot mesh for both legs
        y_val = params['gap_ft']/2 if side == 'left' else -params['gap_ft']/2
        color = left_color if side == 'left' else right_color
        for mesh_type in ['visual', 'collision']:
            add_mesh(link, mesh_type, f'{side}_leg_foot_{mesh_type}',
                    xyz=f"{-params['hip_offset']} {y_val} {-params['l_leg']}", 
                    filename=f"{urdf_prefix}{params['file_id']}/{side}_foot_geom.obj",
                    color=color)
        # Add link to hold mass    
        link_name = f"{side}_feet"
        mass_link_name = f"{link_name}_mass_link"
        mass_link = ET.SubElement(robot, 'link', name=mass_link_name)
        mass_links_parents[mass_link_name] = link.get('name')        
        inertial = ET.SubElement(mass_link, 'inertial', name=f"{link_name}_inertial")
        ET.SubElement(inertial, 'origin', xyz=f"{-params['hip_offset']} {y_val} {-params['l_leg']}", rpy="0 0 0")
        ET.SubElement(inertial, 'mass', value=f"{params['feet_mass']}")
        ET.SubElement(inertial, 'inertia', ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")

    # add hip joint
    add_rev_joint(robot, 'hip', parent='left_leg', child='right_leg', pos=f"{0} {0} {0}")
    for link, parent in mass_links_parents.items():
        add_fixed_joint(robot, f"fixed_{link}", parent=parent, child=link, pos=f"{0} {0} {0}")

    # export urdf
    tree = ET.ElementTree(robot)
    save_file(tree, robot, f'stick_bot_generated_{i}', script_dir)

    # make modifications compatible with Mujoco
    ET.SubElement(ET.SubElement(robot, 'mujoco'), 'compiler', strippath="false")
    tree_mjc = ET.ElementTree(robot)
    for elem in tree_mjc.iter():
        if elem.tag == 'mesh':
            # Change the filename to absolute path
            elem.set('filename', f"{script_dir}/{params['file_id']}/{elem.get('filename').split('/')[-1]}")
    save_file(tree_mjc, robot, f'stick_bot_generated_mjc_{i}', script_dir)