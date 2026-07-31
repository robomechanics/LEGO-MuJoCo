import numpy as np
import can_control as cc
import can
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import csv


class MotorController:
    def __init__(self,
                 motor_id: int,
                 motor_type: str,
                 hip_omega: float,
                 leg_amp_deg: float,
                 Kp: float,
                 Kd: float,
                 t_wait: float = 1.0):
        # store parameters
        self.motor_id    = motor_id
        self.motor_type  = motor_type
        self.hip_omega   = hip_omega
        self.leg_amp_rad = np.deg2rad(leg_amp_deg)
        self.Kp          = Kp
        self.Kd          = Kd
        self.t_wait      = t_wait
        self.kt = 0.199 # torque constant (Nm/A) for AK80-8 motor
        self.gr = 8 # gear ratio for AK80-8 motor

        # --- set up CAN + listener ---
        self.helper   = cc.CanController()
        self.listener = self.StateListener(self)
        self.notifier = can.Notifier(self.helper.bus, [self.listener])
        self.helper.power_on(self.motor_id)

        # wait for first status
        print("[INFO] Motor powering on… waiting for initial state")
        timeout = time.time() + 2.0
        while self.listener.state is None and time.time() < timeout:
            time.sleep(0.01)

        # record start position & time
        if self.listener.state is None:
            print("[WARN] No status received; defaulting start position = 0.0 rad")
            self.start_position = 0.0
        else:
            self.start_position = self.listener.state.position
            print(f"[INFO] Start pos = {self.start_position:.3f} rad")

        self.start_time = time.time()

    class StateListener(can.Listener):
        def __init__(self, controller):
            self.controller = controller
            self.state      = None
        def on_message_received(self, msg):
            st = self.controller.helper.parse_MIT_message(bytes(msg.data),
                                                          self.controller.motor_type)
            self.state = st

    def calculate_sine_reference(self, t: float, start_freq_mult: float = 1.3, start_amp_mult: float = 1.2):
        """Compute self.reference (rad) for elapsed time t."""
        steady_sine = lambda w,t,t0: np.sin(w*(t-t0)) if t > t0 else 0
        trans_sine  = lambda w,t,t0: np.sin(w*(t-t0)) if abs(w*(t0-t)+np.pi/2) < np.pi/2 and t > t0 else 0
        composite = lambda A,w1,w2,t,t0: A*trans_sine(w1,t,t0) - steady_sine(w2,t,t0 + np.pi/w1)

        A   = start_amp_mult
        w1  = self.hip_omega * start_freq_mult
        w2  = self.hip_omega
        t0  = self.t_wait

        self.reference = self.leg_amp_rad * composite(A, w1, w2, t, t0)
        
        # Add in piecewise velocity profile
        At = start_amp_mult * self.leg_amp_rad
        As = self.leg_amp_rad
        if t <= t0:
            velocity = 0.0
        elif t < t0 + np.pi/(2*w1):
           velocity = ((At * w1)) * np.sin(2 * w1 * (t - t0))
        elif t < t0 + np.pi/w1:
            velocity = (As * w2) * np.cos(w1 * (t - t0 ))
        else:
            velocity = -As * w2 *np.cos(w2 * (t - t0 - np.pi/w1))
        
        self.velocity = velocity
    

    def plot_position_results(self, x, y, z, error):
        """Plot position vs time and save to file."""
        plt.plot(x, y, color='g', label='ideal')
        plt.plot(x, z, color='b', label='actual')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (rad)')
        plt.title('Position vs Time')
        plt.grid(True)
        plt.legend()
        out_path = os.path.join('data', 'position_graph.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to {out_path}")
        print("[INFO] Mean error = ", np.mean(error))
    
    def plot_torque_results(self, times, positions, torques):
        """Plot torque vs time and save to file."""
        plt.figure()
        plt.plot(times, torques, color='r', label='Torque')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title('Torque vs Time')
        plt.grid(True)
        plt.legend()
        out_path = os.path.join('data', 'torque_graph.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to {out_path}")
    
    def plot_velocity_results(self, x, velocity):
        """Plot velocity vs time and save to file."""
        plt.figure()
        plt.plot(x, velocity, color='m', label='Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (rad/s)')
        plt.title('Velocity vs Time')
        plt.grid(True)
        plt.legend()
        out_path = os.path.join('data', 'velocity_graph.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to {out_path}")

    def run(self, duration: float, freq: float = 1000.0,
        csv_path: str = None, log_dt: float = 0.01):
        """Main loop: for `duration` seconds at ~`freq` Hz, update the motor.
        Also logs position/velocity/torque/motor_speed to CSV every log_dt seconds."""
        dt = 1.0 / freq
        if csv_path is None:
            csv_path = os.path.join('data', 'timeseries.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        x, y, z = [], [], []
        times, positions, torques = [], [], []
        error = []
        velocity = []

        next_log_time = 0.0

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time_s', 'position_rad', 'velocity_rad_s',
                'torque_Nm', 'motor_speed_rad_s'
            ])

            while True:
                now = time.time()
                elapsed = now - self.start_time
                if elapsed >= duration:
                    break

                self.calculate_sine_reference(elapsed)
                target = self.reference + self.start_position
                print("Reference = ", target)
                print("Error = ", target - self.listener.state.position)

                self.helper.MIT_controller(
                    self.motor_id, self.motor_type,
                    position=target, velocity=self.velocity,
                    Kp=self.Kp, Kd=self.Kd, I=0.0
                )

                x.append(elapsed)
                y.append(self.listener.state.position)
                z.append(target)
                error.append(self.listener.state.position - target)

                time.sleep(max(0, dt - (time.time() - now)))

                fb = self.listener.state
                if fb is None:
                    continue

                tau_motor   = self.kt * fb.current
                tau_out     = tau_motor * self.gr
                motor_speed = fb.velocity * self.gr

                times.append(elapsed)
                positions.append(fb.position)
                torques.append(tau_out)
                velocity.append(self.velocity)

                # --- log at a fixed 0.01s grid (zero-order hold if control loop is slower) ---
                while elapsed >= next_log_time:
                    writer.writerow([
                        f"{next_log_time:.3f}",
                        fb.position, fb.velocity,
                        tau_out, motor_speed
                    ])
                    next_log_time += log_dt

        print(f"[INFO] Time series (position, velocity, torque, motor speed) saved to {csv_path}")
        print("[INFO] Done running sine trajectory.")
        self.plot_position_results(x, y, z, error)
        self.plot_torque_results(times, positions, torques)
        self.plot_velocity_results(x, velocity)
        print("[INFO] Start position = ", self.start_position)

    def return_to_zero(self, 
                   return_kp: float = 10.0,
                   return_kd: float = 2.0,
                   tolerance_rad: float = 0.02,
                   timeout: float = 5.0,
                   freq: float = 200.0):
        """
        Smoothly returns motor to its original start position, then re-homes
        the encoder so the next test begins from a consistent reference.

        Args:
        return_kp:      position gain for the return move (lower = gentler)
        return_kd:      velocity gain for the return move
        tolerance_rad:  how close to zero before declaring success (rad)
        timeout:        max seconds to attempt return before giving up
        freq:           control loop frequency during return (Hz)
        """
        dt = 1.0 / freq

        if self.listener.state is None:
            print("[WARN] No motor state available — cannot return to zero safely.")
            return

        print(f"[INFO] Returning to zero from {self.listener.state.position:.4f} rad...")

        start      = time.time()
        target_pos = 0.0   # true encoder zero

        while True:
            elapsed = time.time() - start

            if elapsed > timeout:
                print(f"[WARN] Return to zero timed out after {timeout}s. "
                  f"Final position: {self.listener.state.position:.4f} rad")
                break

            current_pos = self.listener.state.position
            current_vel = self.listener.state.velocity
            error       = target_pos - current_pos

            if abs(error) < tolerance_rad:
                print(f"[INFO] Reached zero in {elapsed:.2f}s "
                  f"(final error: {np.rad2deg(error):.2f} deg)")
                break

        # low-gain PD to move gently back — no feedforward
            self.helper.MIT_controller(
                self.motor_id,
                self.motor_type,
                position = target_pos,
                velocity = 0.0,
                Kp       = return_kp,
                Kd       = return_kd,
                I        = 0.0
            )

            time.sleep(dt)

        # hold at zero briefly to let it settle
        time.sleep(0.3)

        # re-home encoder only AFTER physically reaching zero
        # zero() shuts off comms for ~1s so wait for it
        print("[INFO] Re-homing encoder...")
        self.helper.zero(self.motor_id)
        time.sleep(1.5)   # wait for zero command to complete

        # reset start_position for the next test
        self.start_position = 0.0
        print("[INFO] Motor zeroed and ready for next test.")


# === usage ===
if __name__ == "__main__":
    ctrl = MotorController(
        motor_id    = 1,
        motor_type  = 'AK80-8',
        hip_omega   = 0.55 * 2 * np.pi,
        leg_amp_deg = 37.5,
        Kp          = 35.5,
        Kd          = 6.5,
        t_wait      = 5.0
    )
    try:
        ctrl.run(duration=20.0, freq=100.0)
    finally:
        ctrl.return_to_zero(
            return_kp  = 8.0,   
            return_kd  = 2.0,
            tolerance_rad = 0.02, 
            timeout    = 5.0
        )
        ctrl.helper.power_off(motor_id=1)
        ctrl.notifier.stop()
        ctrl.helper.can_shutdown()
