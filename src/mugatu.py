from collections import deque
import numpy as np
import os
from src.sim import MjcSim, ProgressCallback
from utils.sim_args import arg_parser
from utils.recorder import Recorder
from typing import Callable, Any

class Mugatu(MjcSim):
    def __init__(self, config: dict) -> None:
        """Initialize the Mugatu simulation environment."""
        scene_path = f"{config['robot_dir']}/mugatu_mjcf/scene_motor.xml"
        self.camera_params = {
            'tracking': "right_leg",
            'distance': 0.5,
            'xyaxis': [1, 0, 0, 0, 0, 1],
        }
        new_scene_path = self.update_xml(scene_path)
        super().__init__(new_scene_path, config)
        self.get_hip_idx()
        self.init_ctrl_params(config["ctrl_dict"])
        self.step_sim() # Take the first sim step to initialize the data

    def get_hip_idx(self) -> None:
        """Get the joint ID and qpos index for the hip joint."""
        self.ctrl_joint_names = ['hip']
        n_ctrl_joints = self.setup_ctrl_joints()
        print(f"Number of control joints: {n_ctrl_joints}")
        self.hip_qpos_idx = self.model.jnt_qposadr[self.ctrl_joint_ids[0]] # qpos index for the hip joint
        self.hip_dof_idx = self.ctrl_dof_addrs[0] # dof index for the hip joint (for qfrc)
        self.hip_qvel_idx = self.hip_dof_idx
        self.action = None

    def init_ctrl_params(self, ctrl_dict: dict[str, Any]={}) -> None: 
        # Default values
        self.Kp = 0
        self.Kd = 0
        self.leg_amp_deg = 0
        self.hip_omega = None
        # self.pend_len = 0.63 # default from a while back
        for k,v in ctrl_dict.items(): 
            setattr(self, k, v)
            print(f"{k} set to {getattr(self, k)}.")

    @property
    def leg_amp_rad(self) -> float:
        """Calculate the angular frequency of the hip joint."""
        return np.deg2rad(self.leg_amp_deg) 

    def pendulum_length(self) -> tuple[float, float]:
        """Calculate the length and z offset of the pendulum."""
        hip_pos = self.data.joint(self.ctrl_joint_names[0]).xanchor
        com_pos = self.mass_center()
        pedulum_length = np.linalg.norm(hip_pos - com_pos)
        pendulum_z = hip_pos[2] - com_pos[2]
        return pedulum_length, pendulum_z

    def calculate_sine_reference(self, waittime: float=1.0, b:float=1.0) -> None:
        """Calculate the sine wave control signal for the hip joint."""
        # b defines how sharp a sine wave is, higher the sharper
        wave = np.sin(self.hip_omega * (self.data.time-waittime))
        wave_val = np.sqrt((1 + b**2) / (1 + (b**2) * wave**2))*wave
        self.reference = self.leg_amp_rad * wave_val
        if self.data.time < waittime: self.reference = 0

    def apply_ctrl(self) -> None:
        """Apply the calculated control signal to the hip joint."""
        self.data.actuator("hip_joint_act").ctrl = self.action

    def calculate_mujoco_position_ctrl(self) -> None:
        """Calculate the position control signal for the hip joint."""
        self.action = self.reference

    def calculate_pd_ctrl(self, hist_window: int=10) -> None:
        """Calculate the PID control signal for the hip joint."""
        self.calculate_sine_reference()

        if self.action is None: # Initialize the control signal
            # start a queue 
            self.p_hist = deque([self.data.qpos[self.hip_qpos_idx]], maxlen=hist_window)  # Fixed-size queue
            self.p_ref_hist = deque([self.reference], maxlen=hist_window)  # Fixed-size queue
            self.action = 0
            return
            
        # update the queue
        self.p_hist.append(self.data.qpos[self.hip_qpos_idx])
        self.p_ref_hist.append(self.reference)
        p_err_hist = np.array(self.p_ref_hist) - np.array(self.p_hist)

        # average the derivative over the entire queue
        p_err = p_err_hist[-1] # most recent entry is at the end
        p_err_d = np.mean(np.diff(p_err_hist) / self.model.opt.timestep)

        # calculate the control signal
        self.action = self.Kp * p_err + self.Kd * p_err_d

    def data_log(self) -> None:
        """Log the data from the simulation."""
        self.actuator_setpoints = self.reference
        self.actuator_actual_pos = self.data.qpos[self.hip_qpos_idx]
        self.actuator_torque = self.data.qfrc_actuator[self.hip_dof_idx]
        self.applied_torque = self.data.qfrc_applied[self.hip_dof_idx]
        self.actuator_speed = self.data.qvel[self.hip_dof_idx]

    def run_sim(self, callbacks: dict[str, Callable]=None) -> None:
        """Run the simulation for the specified time."""
        if self.hip_omega is None:
            self.pend_len = self.pendulum_length()[0]
            self.hip_omega = np.sqrt(9.81 / self.pend_len)

        print(f"hip freq: {self.hip_omega/(2*np.pi)}")
        loop = range(int(self.simtime // self.model.opt.timestep))

        for _ in loop:
            self.calculate_sine_reference()
            self.calculate_pd_ctrl()    
            self.apply_ctrl()
            self.step_sim()
            self.data_log()

            if callbacks:
                for name, func in callbacks.items():
                    func(self)  # Call function dynamically
        
def main():
    args = arg_parser("Mugatu Sim Args")

    # Define the variables and their properties
    plot_attributes = {
        "actuator_actual_pos"   : {"title": "Joint Angle", "unit": "Rad"},
        "actuator_torque"       : {"title": "Joint Torque", "unit": "Nm"},
        "actuator_setpoints"    : {"title": "Joint Setpoint", "unit": "Rad"},
        "actuator_speed"        : {"title": "Joint Speed", "unit": "Rad/s"},
        "time"                  : {"title": "Time", "unit": "s"},  
    }

    # Define the structure of the plots
    plot_structure = [
        ["time", "actuator_actual_pos", "actuator_setpoints"],  # Subplot 1: X = time, Y = angle & setpoint
        ["time", "actuator_torque"],  # Subplot 2: X = time, Y = torque
        ["actuator_actual_pos", "actuator_torque"],  # Subplot 3: X = angle, Y = torque
        # ["actuator_speed", "actuator_torque"],
    ]

    # dictionary of control parameters
    args['ctrl_dict'] = {
        'Kp': 15,
        'Kd': 0.5,
        'leg_amp_deg': 35,
    }

    robot = Mugatu(args)
    progress_cb = ProgressCallback(args['sim_time'])  # Initialize progress tracker
    callbacks_dict = {
        "progress_bar" : progress_cb.update
        }

    if args["record"]:
        recorder = Recorder(args['video_fps'], plot_attributes, plot_structure)
        callbacks_dict["record_frame"] = recorder.record_frame
        callbacks_dict["record_plot_data"] = recorder.record_plot_data

    robot.run_sim(callbacks=callbacks_dict)

    if args["record"]:
        v_dir = f"{args['video_dir']}/{robot.__class__.__name__}/{args['name']}"
        os.makedirs(v_dir, exist_ok=True)
        recorder.generate_plot_video(output_path=f"{v_dir}/live_plot.mp4")
        recorder.generate_robot_video(output_path=f"{v_dir}/robot_walking.mp4")
        recorder.stack_video_frames(recorder.plot_frames, 
                                    recorder.robot_frames,
                                    output_path=f"{v_dir}/combined.mp4")
        
    robot.close()
        
if __name__ == "__main__":
    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()

    main()

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))