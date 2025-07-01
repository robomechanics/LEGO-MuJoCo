import xml.etree.ElementTree as ET

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

def save_file(tree: ET.ElementTree, robot: ET.Element, name: str, script_dir: str) -> None:
    ET.indent(tree, space="  ", level=0)
    ET.dump(robot)
    output_file = f'{script_dir}/{name}.urdf'
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"URDF file saved as {output_file}")