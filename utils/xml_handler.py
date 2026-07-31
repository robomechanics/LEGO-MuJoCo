import os
import trimesh
from typing import Any
import xml.etree.ElementTree as ET
import yaml


def load_mass_config(path: str) -> dict[str,dict[str,float]]:
        """Load the mass config from path"""
        assert path is not None, "mass_config_path not set"
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    
def export_mass_config(mass_dict: dict[str,dict[str,float]], 
                        config_export_path: str) -> None:
    """Save the mass config to a yaml file"""
    with open(config_export_path, "w") as file:
        yaml.dump(mass_dict, file)

def backup_density(path: str):
    mass_config = load_mass_config(path)
    dir = "/".join(path.split("/")[:-1])
    for _, value in mass_config.items():
        value["density"] = value["mass"] / value["volume"]
    export_mass_config(mass_config, f"{dir}/density_backup.yaml")

class MJCFHandler:
    def __init__(self, path: str) -> None:
        """Initialize the XMLHandler class."""
        self.scene_path = path
        self.scene_dir, self.model_path = self.scene_to_robot_path(self.scene_path)
        self.model_tree = ET.parse(self.model_path)
        self.model_root = self.model_tree.getroot()
        
        self.assets = self.model_root.find('asset')
        assert self.assets is not None, "assets tag not found"
        self.worldbody = self.model_root.find('worldbody')
        assert self.worldbody is not None, "worldbody tag not found"
        self.actuator = self.model_root.find('actuator')
        assert self.actuator is not None, "actuator tag not found"
        # find all kinds of stuff
        self.meshes = self.assets.findall('.//mesh')
        self.geoms = self.worldbody.findall('.//geom')
        self.bodies = self.worldbody.findall('.//body')
        self.joints = self.worldbody.findall('.//joint')
        self.motors = self.actuator.findall('.//motor')

        self.mass_dict: dict[str,dict[str,float]] = {}
        self.config_export_path = self.scene_dir + "/mass_config.yaml"
        # if a mass_config file exists, load it
        if os.path.exists(self.config_export_path): 
            self.mass_dict = load_mass_config(self.config_export_path)
        else:
            self.get_mass_of_geoms()
        self.update_volume()
        export_mass_config(self.mass_dict, self.config_export_path)

        if not os.path.exists(self.scene_dir + "/density_backup.yaml"):
            backup_density(self.config_export_path)

    def update_design_params(self, design_params: dict[str, Any]) -> None:
        """Update the design parameters of the model."""
        if design_params == {}: return
        if "body_pos_offset" in design_params.keys():
            body_pos_offset: dict = design_params["body_pos_offset"]
            for body in self.bodies:
                name = body.get("name")
                if name in body_pos_offset.keys():
                    offset = body_pos_offset[name]
                    pos = body.get("pos")
                    pos = (f"{float(pos.split()[0]) + offset[0]} "
                           f"{float(pos.split()[1]) + offset[1]} "
                           f"{float(pos.split()[2]) + offset[2]}")
                    body.set("pos", pos)

        if "geom_pos_offset" in design_params.keys():
            geom_pos_offset: dict = design_params["geom_pos_offset"]
            for geom in self.geoms:
                name = geom.get("name")
                if name in geom_pos_offset.keys():
                    offset = geom_pos_offset[name]
                    pos = geom.get("pos")
                    pos = (f"{float(pos.split()[0]) + offset[0]} "
                           f"{float(pos.split()[1]) + offset[1]} "
                           f"{float(pos.split()[2]) + offset[2]}")
                    geom.set("pos", pos)

        # if "geom_pos_set" in design_params.keys():
        #     geom_pos_set: dict = design_params["geom_pos_set"]
        #     for geom in self.geoms:
        #         name = geom.get("name")
        #         if name in geom_pos_set.keys():
        #             offset = geom_pos_set[name]
        #             pos = geom.get("pos")
        #             pos = (f"{offset[0]} {offset[1]} {offset[2]}")
        #             geom.set("pos", pos)

        if "body_quat" in design_params.keys():
            body_quat: dict = design_params["body_quat"]
            for body in self.bodies:
                name = body.get("name")
                if name in body_quat.keys():
                    quat = body_quat[name]
                    quat = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
                    body.set("quat", quat)

        if "mesh_scale" in design_params.keys():
            mesh_scale: dict = design_params["mesh_scale"]
            for mesh in self.meshes:
                name = mesh.get("name")
                if name in mesh_scale.keys():
                    scale = mesh_scale[name]
                    scale = f"{scale[0]} {scale[1]} {scale[2]}"
                    mesh.set("scale", scale)
        
    def update_volume(self) -> None:
        """Update the volume of the geoms in the model."""
        for mesh_elem in self.meshes:
            mesh_name = mesh_elem.get('name')
            if mesh_name is None: mesh_name = mesh_elem.get('file').split('.')[0]
            mesh_path = f"{self.scene_dir}/{mesh_elem.get('file')}"
            loaded_mesh = trimesh.load_mesh(mesh_path)
            if not loaded_mesh.is_watertight:
                print(f"[WARNING] Mesh '{mesh_name}' is not watertight; keeping existing volume")
                continue
            self.mass_dict[mesh_name]["volume"] = float(loaded_mesh.volume)
            
    def scene_to_robot_path(self, scene_path: str) -> tuple[str, str]:
        """Convert the scene path to robot path."""
        self.scene_tree = ET.parse(scene_path)
        self.scene_root = self.scene_tree.getroot()
        # extract the file value of include tag
        model_subpath = self.scene_root.find("include").get("file")
        # get directory from scene path
        scene_dir = "/".join(scene_path.split("/")[:-1])
        model_path = f"{scene_dir}/{model_subpath}"
        return scene_dir, model_path

    def get_mass_of_geoms(self) -> dict[str, float]:
        """Get the mass of the geoms in the model."""
        for geom in self.geoms:
            mass = geom.get('mass')
            assert mass is not None, "mass attribute not found"
            name = geom.get('mesh')
            assert name is not None, "mesh attribute not found"
            self.mass_dict[name] = {"mass": float(mass), "scale": 1}

    def update_mass(self) -> None:
        """Apply the mass scale to the geoms in the xml."""
        for geom in self.geoms:
            name = geom.get('mesh')
            new_mass = self.mass_dict[name]["mass"] * self.mass_dict[name]["scale"]
            geom.set('mass', str(new_mass))

    def export_xml_scene(self) -> str:
        """Export the modified xml to a file."""
        # write a temp robot file
        new_model_path = self.model_path.replace(".xml", "_temp.xml")
        self.model_tree.write(new_model_path)
        # use robot name to write temp scene file
        new_model_name = new_model_path.split("/")[-1]
        self.scene_root.find("include").set("file", new_model_name)
        new_scene_path = self.scene_path.replace(".xml", "_temp.xml")
        self.scene_tree.write(new_scene_path)
        return new_scene_path
    
    def save_scale_up_robot(self, scale: float, mass_power: float) -> None:
        """Scale up the robot by a factor of scale. Rename all parts"""
        for mesh in self.meshes:
            mesh.set('scale', f"{scale} {scale} {scale}")
            mesh.set('name', f"{mesh.get('name')}_{scale}x")
        
        for element in self.geoms + self.bodies + self.joints:
            element_name = element.get('name')
            if element_name is not None: element.set('name', f"{element_name}_{scale}x")
            element_pos = element.get('pos')
            scaled_element_pos = [float(x) * scale for x in element_pos.split()]
            scaled_element_pos = f"{scaled_element_pos[0]} {scaled_element_pos[1]} {scaled_element_pos[2]}"
            element.set('pos', scaled_element_pos)
            if element.tag == "geom":
                element_mass = element.get('mass')
                element.set('mesh', f"{element.get('mesh')}_{scale}x")
                scaled_element_mass = float(element_mass) * scale ** mass_power
                element.set('mass', str(scaled_element_mass))
        
        for motor in self.motors:
            motor.set('name', f"{motor.get('name')}_{scale}x")
            motor.set('joint', f"{motor.get('joint')}_{scale}x")

        scaled_model_path = self.model_path.replace(".xml", f"_{scale}x.xml")
        self.model_tree.write(scaled_model_path)

        new_model_name = scaled_model_path.split("/")[-1]
        self.scene_root.find("include").set("file", new_model_name)
        new_scene_path = self.scene_path.replace(".xml", f"_{scale}x.xml")
        self.scene_tree.write(new_scene_path)
        return new_scene_path

if __name__ == "__main__":
    pass