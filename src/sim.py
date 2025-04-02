import mujoco
import mujoco.viewer
import numpy as np
import sys
import time
from tqdm import tqdm
from typing import Any
from utils.xml_handler import MJCFHandler

class MjcSim:
    def __init__(self, model_path: str, config: dict) -> None:
        """ Initialize the Mujoco simulation environment."""
        self.simtime = config['sim_time']
        self.gui = config['gui']
        self.v_fps = config['video_fps']
        self.config = config

        # new_scene_path = self.update_xml(model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        total_mass = np.sum(self.model.body_mass)
        print(f"Total mass: {total_mass} kg")

        if self.config['record'] or self.config['gui']: self.setup_gui()
        self.ctrl_joint_names = None # names of the joints to control

    def update_xml(self, scene_path: str, design_params: dict[str, Any]={}) -> str:
        self.mjcf_handler = MJCFHandler(scene_path)
        self.mjcf_handler.update_mass()
        self.mjcf_handler.update_design_params(design_params)
        new_scene_path = self.mjcf_handler.export_xml_scene()
        return new_scene_path
    
    def scale_up_xml(self, scale: float, mass_power: float) -> str:
        """Scale up the model by a factor of scale."""
        scale_scene_path = self.mjcf_handler.save_scale_up_robot(scale, mass_power)
        return scale_scene_path

    def setup_ctrl_joints(self) -> int:
        """Convert actuator joint names to joint ids and dof addresses."""
        self.ctrl_dof_addrs = [self.model.jnt_dofadr[j_id] for j_id in self.ctrl_joint_ids]
        return len(self.ctrl_joint_ids)
    
    @property
    def ctrl_joint_ids(self) -> list[int]:
        ctrl_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name) 
                          for j_name in self.ctrl_joint_names]
        return ctrl_joint_ids

    def setup_gui(self):
        """Setup the GUI for the simulation."""
        self.width, self.height = 1920 // 2, 1080 // 2 
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.camera = mujoco.MjvCamera()

        tracking_obj = self.camera_params['tracking']
        # copy over non-tracking camera parameters
        normal_cam_attrs = {"distance", "lookat", "elevation", "azimuth"}
        for attr in normal_cam_attrs & set(self.camera_params.keys()): 
            setattr(self.camera, attr, self.camera_params[attr])

        if tracking_obj is not None:
            # calculate tracking camera parameters
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, tracking_obj)
            xyaxis = self.camera_params['xyaxis']
            z_axis = np.cross(xyaxis[:3], xyaxis[3:])
            azimuth = np.degrees(np.arctan2(z_axis[1], z_axis[0]))
            elevation = np.degrees(np.arctan2(z_axis[2], np.linalg.norm(z_axis[0:2])))

            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = body_id
            self.camera.azimuth = azimuth
            self.camera.elevation = elevation            

        if self.gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            for attr in dir(self.camera):
                if not attr.startswith("_"): # copy camera public attributes
                    setattr(self.viewer.cam, attr, getattr(self.camera, attr))

    def get_image(self):
        """Get the current camera image from the simulation."""
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()
    
    def mass_center(self):
        """Calculate the center of mass of the robot."""
        mass = np.expand_dims(self.model.body_mass, axis=1)
        xpos = self.data.xipos
        return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:3].copy()

    def step_sim(self):
        """Step the simulation forward by one time step."""
        mujoco.mj_step(self.model, self.data)

        if self.gui: 
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)
            if not self.viewer.is_running(): sys.exit()

    def close(self):
        """Close the simulation environment."""
        if self.config['record']: self.renderer.close()
        if self.gui: self.viewer.close()


class ProgressCallback:
    """Tracks simulation progress using tqdm."""
    def __init__(self, total_time: float) -> None:
        self.started = False
        self.total_time = total_time
        self.next_update_time = 0

    def update(self, sim: MjcSim) -> None:
        """Update the progress bar using the simulation time step."""
        if not self.started: 
            self.pbar = tqdm(total=self.total_time, desc="Simulation Progress", unit="s")
            self.next_update_gap = sim.simtime / 100
            self.started = True
        
        curr_time = round(sim.data.time, 5)        

        if curr_time >= self.next_update_time:
            self.pbar.n = min(curr_time, self.pbar.total)  # Prevents overshooting
            self.pbar.refresh()
            self.next_update_time += self.next_update_gap 

        if curr_time >= self.pbar.total:
            self.pbar.close()
            print(f"{curr_time}s of simulation finished.")