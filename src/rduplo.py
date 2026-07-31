from collections import deque
import copy
import numpy as np
import mujoco
import os
import pickle as pkl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.sim import MjcSim, ProgressCallback
from utils.sim_args import arg_parser
from utils.recorder import Recorder
from typing import Callable, Any
import yaml
from scipy.spatial.transform import Rotation as R


class Duplo(MjcSim):
    def __init__(self, config: dict) -> None:
        """Initialize the Duplo simulation environment."""
        self.configs = config
        self.scene_path = f"{config['robot_dir']}/bigfoot/scene.xml"
        self.camera_params = {
            'tracking': "motor",
            'distance': 5,
            'xyaxis': [1, 0, 0, 0, 0, 1],
        }
        new_scene_path = self.update_xml(self.scene_path, config['design_params'])
        self.start_quat = config['design_params']['body_quat']['motor']
        super().__init__(new_scene_path, config)
        self.get_hip_idx()
        self.init_ctrl_params(config["ctrl_dict"])
        print(f"Start quat: {self.start_quat}")
        self.step_sim() # Take the first sim step to initialize the data
        
         # ─── Electrical + gearbox parameters (AK80-64) ───
        self.V_supply     = 48        # volts  (rated bus)
        self.R_phase      = 0.22         # ohms   (phase-phase)
        self.k_motor      = 0.136        # Nm/A   (bare motor)
        self.gear_ratio   = 64           # :1
        self.eta_gear     = 0.85         # 85 % efficient

        # Effective output-shaft constants
        self.k_eff = self.k_motor * self.gear_ratio * self.eta_gear  # ≈ 7.4 Nm/A
        self.R_eff = self.R_phase                                    # ≈ 0.22 Ω

        self.I_max = 19.0         # A (datasheet peak current)
        
        # Swaying and Fall Parameters
        self.sway_window = config.get('sway_window', 1.0)
        self.sway_threshold_deg = config.get('degree_threshold', 0.1)
        self.sway = False
        self.fall = False
        self.wait_time          = config.get('wait_time', 1.0)   # same t_wait you pass into calculate_sine_reference  




    def get_hip_idx(self) -> None:
        """Get the joint ID and qpos index for the hip joint."""
        self.ctrl_joint_names = ['hip']
        n_ctrl_joints = self.setup_ctrl_joints()
        self.hip_qpos_idx = self.model.jnt_qposadr[self.ctrl_joint_ids[0]] # qpos index for the hip joint
        self.hip_dof_idx = self.ctrl_dof_addrs[0] # dof index for the hip joint (for qfrc)
        self.hip_qvel_idx = self.hip_dof_idx
        self.action = None

    def init_ctrl_params(self, ctrl_dict: dict[str, Any]={}) -> None: 
        # Default values
        self.Kp = 0
        self.Kd = 0
        self.j_damping = 0
        self.leg_amp_deg = 0
        self.hip_omega = None
        # self.pend_len = 0.63 # default from a while back
        for k,v in ctrl_dict.items(): 
            setattr(self, k, v)
            # print(f"{k} set to {getattr(self, k)}.")

    @property
    def leg_amp_rad(self) -> float:
        """Calculate the angular frequency of the hip joint."""
        return np.deg2rad(self.leg_amp_deg) 

    def pendulum_length(self) -> tuple[float, float]:
        """Calculate the length and z offset of the pendulum."""
        hip_pos = self.data.joint(self.ctrl_joint_names[0]).xanchor
        com_pos = self.mass_center()
        pendulum_length = np.linalg.norm(hip_pos - com_pos)
        pendulum_z = hip_pos[2] - com_pos[2]
        # print(f"Center of mass: {com_pos}, Hip position: {hip_pos}")
        return pendulum_length, pendulum_z

    # def calculate_sine_reference(self, waittime: float=1.0, b: float = 1.0) -> None:
    #     t = self.data.time
    #     wave = np.sin(self.hip_omega * (t - waittime))
    #     wave_val = np.sqrt((1 + b**2) / (1 + (b**2) * wave**2)) * wave

    #     # Smooth scaling from 0 to 1
    #     scale = min(1.0, max(0.0, (t / waittime)))  # linear ramp
    #     self.reference = scale * self.leg_amp_rad * wave_val

    def calculate_sine_reference(self, 
                                 t_wait: float=1.0, 
                                 start_freq_mult:float=1.0, 
                                 start_amp_mult:float=1.0) -> None:
        """Calculate the sine wave control signal for the hip joint."""
        # steady state sine wave
        steady_sine = lambda w,t,t0: np.sin(w*(t-t0)) if t > t0 else 0
        # transience sine wave only active in first half period
        trans_sine = lambda w,t,t0: np.sin(w*(t-t0)) if abs(w*(t0-t)+np.pi/2) < np.pi/2 else 0
        # combine them
        composite = lambda A,w1,w2,t,t0: A*trans_sine(w1,t,t0) - steady_sine(w2,t,t0+np.pi/w1)
        self.reference = self.leg_amp_rad * composite(start_amp_mult, 
                                                      self.hip_omega * start_freq_mult, 
                                                      self.hip_omega, 
                                                      self.data.time, 
                                                      t_wait)
        
    def tilt_from_start(self) -> float:
        """Return tilt angle (deg) of body w.r.t starting quaternion q_start."""
        q_now = self.data.qpos[3:7]      
        w_delta = abs(np.dot(self.start_quat, q_now))  # same as conjugate
        phi = 2.0 * np.arccos(np.clip(w_delta, -1.0, 1.0))
        return np.degrees(phi)

    def check_sway(self) -> bool:
        """Checks if robot is swaying (tilted beyond threshold). Returns True if swaying, False otherwise."""
        tilt_deg = self.tilt_from_start()
        if tilt_deg > self.sway_threshold_deg:
            return True
        return False

    def check_fall(self, threshold_deg: float = 85.0) -> bool:
        """Check if the robot has fallen based on the tilt angle."""
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return abs(roll) > threshold_deg or abs(pitch) > threshold_deg



    def apply_ctrl(self) -> None:
        """Apply the calculated control signal to the hip joint."""
        self.data.actuator("hip_joint_act").ctrl = self.action

    def calculate_mujoco_position_ctrl(self) -> None:
        """Calculate the position control signal for the hip joint."""
        self.action = self.reference

    def calculate_pd_ctrl(self, hist_window: int=10) -> None:
        """Calculate the PID control signal for the hip joint."""
        # self.calculate_sine_reference()

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
    
    def record_contact_points(self) -> None:
        for k,v in self.contact_bodies.items():
            body_id = v['body_id']
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]
                if geom1_body == body_id or geom2_body == body_id:
                    # Get world-frame contact position
                    pos_world = contact.pos
                    
                    # Body's current pose
                    body_pos = self.data.body(body_id).xpos
                    body_quat = self.data.body(body_id).xquat

                    mesh_pos = v['pos']
                    mesh_quat = v['quat']
                    mesh_offset = v['mesh_offset']
                    
                    # Convert quaternion to rotation matrix
                    R_body = np.zeros(9)
                    mujoco.mju_quat2Mat(R_body, body_quat)
                    R_body = R_body.reshape(3, 3)

                    R_mesh = np.zeros(9)
                    mujoco.mju_quat2Mat(R_mesh, mesh_quat)
                    R_mesh = R_mesh.reshape(3, 3)
                    
                    # Transform to body frame: p_body = R^T (p_world - body_pos)
                    p_body = R_body.T @ (pos_world - body_pos)
                    p_mesh = R_mesh.T @ (p_body - mesh_pos)

                    timed_p_mesh = np.hstack([self.data.time, p_mesh])

                    if k in self.con_dict.keys():
                        # vstack
                        self.con_dict[k]['t_coords'] = np.vstack([self.con_dict[k]['t_coords'],
                                                                  timed_p_mesh.copy()])
                    else:
                        self.con_dict[k] = {}
                        self.con_dict[k]['t_coords'] = [timed_p_mesh.copy()]
                        self.con_dict[k]['pos'] = mesh_pos
                        self.con_dict[k]['quat'] = mesh_quat

                        for mesh in self.mjcf_handler.meshes:
                            if mesh.get('name') == v['mesh']:
                                file_name = mesh.get('file').split('.')[0]
                                self.con_dict[k]['mesh'] = file_name
                        self.con_dict[k]['mesh_offset'] = mesh_offset

    def voltage_to_torque(self, V_cmd):
        V = np.clip(V_cmd, -self.V_supply, self.V_supply)
        w = self.data.qvel[self.hip_dof_idx]          # rad/s
        I = (V - self.k_eff * w) / self.R_eff         # amps
        I = np.clip(I, -self.I_max, self.I_max)
        tau = self.k_eff * I - self.j_damping * w
        return tau
    

    def run_sim(self, callbacks: dict[str, Callable]=None) -> None:
        """Run the simulation for the specified time."""
        # print(self.config["ctrl_dict"])
        self.mean_quat = self.data.qpos[3:7].copy()
        self._com_log: list[np.ndarray] = [self.mass_center()]
        self._body_log: list[np.ndarray] = [self.data.body('motor').xpos.copy()]
        if self.hip_omega is None:
            self.pend_len = self.pendulum_length()[0]
            self.hip_omega = np.sqrt(9.81 / self.pend_len)

        # print(f"hip freq: {self.hip_omega/(2*np.pi)}")
        loop = range(int(self.simtime // self.model.opt.timestep))
        quats = []
        self.contact_bodies = {
            'leg_v': {
                'pos': np.array([0.14908, -0.9875, -0.0125974]),    # pos of mesh (rel to body)
                'mesh_offset' : np.array([-0.265, 0, 0]),           # pos of body's parent's parent (rel to motor)
                'quat': np.array([0, 0, -0.707107, 0.707107]),      # quat of mesh (rel to body)
                'mesh': 'part_1' 
                },
            'leg_v_2': {
                'pos': np.array([-0.14908, -0.9875, -0.0124026]),
                'mesh_offset' : np.array([0.265, 0, 0]),
                'quat': np.array([0.707107, 0.707107, 0, 0]),
                'mesh': 'part_1'
                }
            }

        for k in self.contact_bodies.keys():
            self.contact_bodies[k]['body_id'] = mujoco.mj_name2id(self.model,
                                                                  mujoco.mjtObj.mjOBJ_BODY, 
                                                                  k)
        self.con_dict: dict[str,dict[str,list|np.ndarray|str]] = {}
        self.con_dict['params'] = {
            'mesh_dir' : '/'.join((self.scene_path.split('/')[:-1])),
            'stretch_factors' : np.array(self.configs['design_params']['mesh_scale']['part_1'])
            }
        for _ in loop:
            self.calculate_sine_reference(start_freq_mult=1.5, 
                                          start_amp_mult=1.2)
            angle_target = self.reference
            G = self.V_supply / 2
            self.cmd_voltage = G * angle_target
            self.cmd_voltage = np.clip(self.cmd_voltage, -self.V_supply, self.V_supply)
            torque = self.voltage_to_torque(self.cmd_voltage)
            self.data.actuator("hip_joint_act").ctrl = torque
            self.calculate_pd_ctrl()    
            self.apply_ctrl()
            self.step_sim()
            t = self.data.time
            if t <= self.wait_time and self.leg_amp_deg != 0:                
                if self.check_sway():
                    self.sway = True
                    # print('[SWAY DETECTED]')
                else:
                    self.sway = False
            if _ % 5 == 0:
                if self.check_fall():   
                    self.fall = True
                    print("[FALL DETECTED]")
                    break
            if self.leg_amp_deg == 0 and self.check_sway():
                self.sway = True
                # print('[SWAY DETECTED]')
            self.data_log()
            self._com_log.append(self.mass_center())
            self._body_log.append(self.data.body('motor').xpos.copy())
            quats.append(self.data.qpos[3:7].copy())
            if callbacks:
                for name, func in callbacks.items():
                    func(self)  # Call function dynamically
        if self.sway: print('[SWAY DETECTED]')

        if len(quats) > 0:
            mean_quat = np.mean(quats, axis=0)
            n = np.linalg.norm(mean_quat)
            if n > 1e-8:
                self.mean_quat = (mean_quat / n).copy()
        # print(quats)
        if self.sway and self.leg_amp_deg == 0:
            print(f"SWAY DETECTED! input new mean quat: {self.mean_quat[0]:.3f}, {self.mean_quat[1]:.3f}, {self.mean_quat[2]:.3f}, {self.mean_quat[3]:.3f}")

        if self.leg_amp_deg > 0:
            self.stats = self._compute_stats()
            self._print_stats()

    def _compute_stats(self) -> dict:
        com_xyz = np.array(self._com_log)
        steps = np.diff(com_xyz[:, :2], axis=0)
        total_distance = float(np.sum(np.linalg.norm(steps, axis=1)))
        avg_speed = total_distance / self.simtime if self.simtime > 0 else 0.0

        com_xyz = np.array(self._com_log)
        disp = com_xyz[-1, :2] - com_xyz[0, :2]  # XY displacement start→end
        if np.linalg.norm(disp) > 1e-6:
            yaw_deg = float(np.degrees(np.arctan2(disp[0], disp[1])))
        else:
            yaw_deg = 0.0

        return {
            "total_distance_m": total_distance,
            "avg_speed_m_s": avg_speed,
            "final_yaw_deg": yaw_deg,
            "fall": self.fall,
        }

    def _print_stats(self) -> None:
        s = self.stats
        print(
            f"\n── Run Stats ──────────────────────────\n"
            f"  Total distance : {s['total_distance_m']:.4f} m\n"
            f"  Average speed  : {s['avg_speed_m_s']:.4f} m/s\n"
            f"  Final yaw      : {s['final_yaw_deg']:.2f} deg\n"
            f"  Fell           : {s['fall']}\n"
            f"───────────────────────────────────────"
        )

    def export_com_plot(self, output_dir: str) -> str | None:
        """Export COM trajectory plots: top-down (XY) and side view (YZ, matching the GUI camera)."""
        if not hasattr(self, "_com_log") or len(self._com_log) < 2:
            print("[COM plot] Not enough COM samples to export a trajectory.")
            return None

        com = np.array(self._com_log)
        body = np.array(self._body_log) if hasattr(self, "_body_log") else None

        fall_status = "FELL" if self.fall else "Stable"
        end_marker = "X" if self.fall else "o"
        end_label = "End (fell)" if self.fall else "End"

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Trajectory ({fall_status})", fontsize=13)

        # ── left: top-down XY ──────────────────────────────────────────
        ax = axes[0]
        ax.plot(com[:, 0], com[:, 1], linewidth=2.0, color="tab:blue", label="COM")
        if body is not None:
            ax.plot(body[:, 0], body[:, 1], linewidth=1.5, color="tab:orange",
                    linestyle="--", label="Motor body")
        ax.scatter(com[0, 0], com[0, 1], s=80, color="tab:green", zorder=3)
        ax.scatter(com[-1, 0], com[-1, 1], s=140 if self.fall else 80,
                   color="tab:red", marker=end_marker, zorder=4)
        ax.annotate("Start", (com[0, 0], com[0, 1]), xytext=(6, 6),
                    textcoords="offset points", color="tab:green", fontsize=9, fontweight="bold")
        ax.annotate(end_label, (com[-1, 0], com[-1, 1]), xytext=(6, -12),
                    textcoords="offset points", color="tab:red", fontsize=9, fontweight="bold")
        ax.set_title("Top-down (X–Y)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y / forward (m)")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best", fontsize=8)

        # ── right: side view YZ — matches GUI camera (looking in +Y) ──
        ax = axes[1]
        ax.plot(com[:, 1], com[:, 2], linewidth=2.0, color="tab:blue", label="COM")
        if body is not None:
            ax.plot(body[:, 1], body[:, 2], linewidth=1.5, color="tab:orange",
                    linestyle="--", label="Motor body")
        ax.scatter(com[0, 1], com[0, 2], s=80, color="tab:green", zorder=3)
        ax.scatter(com[-1, 1], com[-1, 2], s=140 if self.fall else 80,
                   color="tab:red", marker=end_marker, zorder=4)
        ax.annotate("Start", (com[0, 1], com[0, 2]), xytext=(6, 6),
                    textcoords="offset points", color="tab:green", fontsize=9, fontweight="bold")
        ax.annotate(end_label, (com[-1, 1], com[-1, 2]), xytext=(6, -12),
                    textcoords="offset points", color="tab:red", fontsize=9, fontweight="bold")
        ax.set_title("Side view (Y–Z) — matches GUI")
        ax.set_xlabel("Y / forward (m)")
        ax.set_ylabel("Z / height (m)")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best", fontsize=8)

        axes[0].text(0.02, 0.98, self._com_plot_metrics_text(),
                     transform=axes[0].transAxes, ha="left", va="top",
                     fontsize=8, family="monospace",
                     bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9})

        if self.fall:
            for ax in axes:
                ax.text(0.02, 0.02, "X = fall point", transform=ax.transAxes,
                        ha="left", va="bottom", color="tab:red", fontsize=10, fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9})

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self._com_plot_stem()}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"[COM plot] Saved trajectory to {output_path}")
        return output_path

    def _com_plot_stem(self) -> str:
        """Build the COM plot filename stem from this simulation's parameters."""
        x_offset, y_offset = self._com_offsets()
        x_scale, y_scale, z_scale = self._com_scales()
        omega = float(self.hip_omega) if self.hip_omega is not None else 0.0
        leg_amp_deg = float(self.leg_amp_deg)
        return (
            f"x{self._short_num(x_offset)}_y{self._short_num(y_offset)}_"
            f"sx{self._short_num(x_scale)}_sy{self._short_num(y_scale)}_sz{self._short_num(z_scale)}_"
            f"w{self._short_num(omega)}_a{self._short_num(leg_amp_deg)}"
        )

    def _com_plot_metrics_text(self) -> str:
        """Return the metrics summary text shown on the COM plot."""
        x_offset, y_offset = self._com_offsets()
        x_scale, y_scale, z_scale = self._com_scales()
        omega = float(self.hip_omega) if self.hip_omega is not None else 0.0
        return "\n".join([
            f"x_offset    = {self._short_num(x_offset)}",
            f"y_offset    = {self._short_num(y_offset)}",
            f"x_scale     = {self._short_num(x_scale)}",
            f"y_scale     = {self._short_num(y_scale)}",
            f"z_scale     = {self._short_num(z_scale)}",
            f"omega       = {self._short_num(omega)}",
            f"leg_amp_deg = {self._short_num(float(self.leg_amp_deg))}",
        ])

    def _com_offsets(self) -> tuple[float, float]:
        """Pick representative x/y foot offsets from the current design params."""
        geom_offsets = self.configs.get("design_params", {}).get("geom_pos_offset", {})
        for key in ("RightFoot", "LeftFoot", "ballfoot_1"):
            if key in geom_offsets:
                vals = geom_offsets[key]
                return float(vals[0]), float(vals[1])
        for vals in geom_offsets.values():
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                return float(vals[0]), float(vals[1])
        return 0.0, 0.0

    def _com_scales(self) -> tuple[float, float, float]:
        """Pick representative foot scale values from the current design params."""
        mesh_scales = self.configs.get("design_params", {}).get("mesh_scale", {})
        for key in ("RightFoot", "part_1", "footstl_scaled_v4"):
            if key in mesh_scales:
                vals = mesh_scales[key]
                return float(vals[0]), float(vals[1]), float(vals[2])
        for vals in mesh_scales.values():
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                return float(vals[0]), float(vals[1]), float(vals[2])
        return 1.0, 1.0, 1.0

    @staticmethod
    def _short_num(value: float) -> str:
        """Format a value compactly using roughly three significant figures."""
        return format(value, ".3g").replace("+", "")

def main():
    args = arg_parser("Duplo Sim Args")
    

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
        'Kp': 20,
        'Kd': 12,
        # 'leg_amp_deg': 40.6411773,
        'leg_amp_deg': 30,
        # 'leg_amp_deg': 0,        
        'hip_omega': 0.7 * 2 * np.pi,
    }
    
    # Input the x and y offsets for leg positions
    x = -0.8 # changes feet gap (+x means a wider gap)
    y = -13.0 # changes center of feet location (+y means feet are more forward)

    print(f"Using x={x*10e-4}m, y={y*10e-4}m for ballfoot offset")


    args['design_params'] = {
        'body_pos_offset': {
            # 'foot_group_1': [0.04, 0.1, -0.08], 
            # 'foot_group_2': [0.04, 0.1, 0.08]
        },
        'geom_pos_offset': {

            'LeftFoot': [x*1e-3, -y*1e-3, 0],
            'RightFoot': [-x*1e-3, -y*1e-3, 0],

            'hip_rod_1': [0, 0, 0],
            #0 'foot_edge_1_1': [0, 0, 0],
            # 'foot_edge_2_1': [0, 0, 0],
            # 'middle_foot_part_1': [0, 0, 0],
            'leg_rod_1': [0.0, 0, 0],
            'motor_part1_1': [0, 0, 0],
            'motor_part2_1': [0, 0, 0],
            'arm_rod_1': [0, 0, 0],
            'battery_1': [0, 0, 0],
            'hip_rod_2': [0, 0, 0],
            'motor_part3_1': [0, 0, 0],
            # 'foot_edge_1_2': [0, 0, 0],
            # 'foot_edge_2_2': [0, 0, 0],
            # 'middle_foot_part_2': [0, 0, 0],
            'leg_rod_2': [0.0, 0, 0],
            'arm_rod_2': [0, 0, 0],
            'battery_2': [0, 0, 0],
        },
        'mesh_scale': {
            'part_1': [1, 1, 1],
            'hip': [1, 1, 1],
            'leg_rod': [1, 1, 1.00],
        },

        'body_quat': { 
            'motor': [  0.967, -0.013, -0.003, 0.256], 
        }
    }

    robot = Duplo(args)
    progress_cb = ProgressCallback(args['sim_time'])  # Initialize progress tracker
    callbacks_dict = {
        "progress_bar" : progress_cb.update
        }

    if args["record"]:
        recorder = Recorder(args['video_fps'], plot_attributes, plot_structure)
        callbacks_dict["record_frame"] = recorder.record_frame
        callbacks_dict["record_plot_data"] = recorder.record_plot_data

    robot.run_sim(callbacks=callbacks_dict)

    if args.get("com", False):
        com_plot_dir = os.path.join("data", "com_plots")
        robot.export_com_plot(com_plot_dir)

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
