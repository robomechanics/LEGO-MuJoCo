"""
Shared trajectory + PD control law, used by both the headless actuation
sweep (gen_sweep_actuation.py) and the GUI viewer (test_sim_closedloop.py).
Keeping this in one place means a gain/trajectory change only has to happen
once, instead of the two callers silently drifting apart.
"""
import numpy as np


def calculate_sine_reference(t, hip_omega, leg_amp_rad, start_amp_mult, start_freq_mult, t_wait=3.0):
    """Direct port of motorwave.py's calculate_sine_reference."""
    w1           = hip_omega * start_freq_mult
    w2           = hip_omega
    t0           = t_wait
    At           = start_amp_mult * leg_amp_rad
    As           = leg_amp_rad
    t_transition = t0 + np.pi / w1

    if t <= t0:
        position, velocity = 0.0, 0.0
    elif t < t_transition:
        phase    = w1 * (t - t0)
        position = At * np.sin(phase)
        velocity = At * w1 * np.cos(phase)
    else:
        phase    = w2 * (t - t_transition)
        position = -As * np.sin(phase)
        velocity = -As * w2 * np.cos(phase)

    return position, velocity


def pd_torque(target_pos, target_vel, current_pos, current_vel, Kp, Kd, torque_limit, ramp=1.0):
    """PD law in radians, clamped to the motor torque limit."""
    tau = Kp * ramp * (target_pos - current_pos) + Kd * ramp * (target_vel - current_vel)
    return float(np.clip(tau, -torque_limit, torque_limit))
