import os
import xml.etree.ElementTree as ET
# full absolute path to this script
script_path = os.path.abspath(__file__)
# directory containing this script
script_dir  = os.path.dirname(script_path)

def add_box_visual(link: ET.Element, name: str, xyz: str, size: str, color: str) -> None:
    """Add a visual element to a link."""
    visual = ET.SubElement(link, 'visual', name=name)
    ET.SubElement(visual, 'origin', xyz=xyz, rpy="0 0 0")
    ET.SubElement(ET.SubElement(visual, 'geometry'), 'box', size=size)
    ET.SubElement(ET.SubElement(visual, 'material'), 'color', rgba=color)

def add_mesh(link: ET.Element, type: str, name: str, xyz: str, filename: str, color: str) -> None:
    """Add a visual element to a link."""
    mesh_tag = ET.SubElement(link, type, name=name)
    ET.SubElement(mesh_tag, 'origin', xyz=xyz, rpy="0 0 0")
    ET.SubElement(ET.SubElement(mesh_tag, 'geometry'), 'mesh', filename=filename)
    ET.SubElement(ET.SubElement(mesh_tag, 'material'), 'color', rgba=color)

def add_rev_joint(link: ET.Element, name: str, parent: str, child: str, pos: str) -> None:
    """Add a revolute joint to a link."""
    joint = ET.SubElement(link, 'joint', name=name, type='revolute')
    ET.SubElement(joint, 'origin', xyz=pos, rpy="0 0 0")
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=child)
    ET.SubElement(joint, 'axis', xyz="0 1 0")
    ET.SubElement(joint, 'limit', lower=f"{-3.14/4}", upper=f"{3.14/4}")

def add_fixed_joint(link: ET.Element, name: str, parent: str, child: str, pos: str) -> None:
    """Add a fixed joint to a link."""
    joint = ET.SubElement(link, 'joint', name=name, type='fixed')
    ET.SubElement(joint, 'origin', xyz=pos, rpy="0 0 0")
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=child)

# dict to store design params
params = {
    's'     : 0.005,
    'mot_x' : 0.02,
    'mot_y' : 0.01,
    'mot_z' : 0.02,
    'gap_ft': 0.032,
    'w_arm' : 0.0625,
    'l_arm' : 0.104,
    'l_leg' : 0.153,
    's_hand' : 0.02,
    'hip_offset' : -0.014,
    'leg_mass' : 0.1,
    'feet_mass' : 0.1,
    'hand_mass' : 0.1,
    'file_id' : 'X_0.24_Y_0.24_Z_0.24_box_x_0.101_box_y_0.0527'
}

left_color = "1 0 0 0.5"
right_color = "0 0 1 0.5"
mass_color = "0 1 0 0.5"

s = params['s']
s_hand = params['s_hand']

robot = ET.Element('robot', name='stick_bot')
ET.SubElement(ET.SubElement(robot, 'mujoco'), 'compiler', strippath="false")
left_leg = ET.SubElement(robot, 'link', name='left_leg')
right_leg = ET.SubElement(robot, 'link', name='right_leg')

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
                 filename=f"{script_dir}/{params['file_id']}/{side}_foot_geom.obj",
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


# add joints
add_rev_joint(robot, 'hip', parent='left_leg', child='right_leg', pos=f"{0} {0} {0}")
for link, parent in mass_links_parents.items():
    add_fixed_joint(robot, f"fixed_{link}", parent=parent, child=link, pos=f"{0} {0} {0}")

tree = ET.ElementTree(robot)
ET.indent(tree, space="  ", level=0)         # ← add nice indentation :contentReference[oaicite:0]{index=0}
ET.dump(robot)

# Save the URDF file
output_file = f'{script_dir}/stick_bot_generated.urdf'
tree.write(output_file, encoding='utf-8', xml_declaration=True)
print(f"URDF file saved as {output_file}")