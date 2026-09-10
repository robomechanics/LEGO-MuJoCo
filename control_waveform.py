#!/usr/bin/env python3
"""Shared startup waveform helpers for simulation and hardware scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_HIP_FREQ_HZ = 0.57
DEFAULT_HIP_OMEGA = DEFAULT_HIP_FREQ_HZ * 2.0 * math.pi
DEFAULT_LEG_AMP_DEG = 35.0
DEFAULT_T_WAIT = 9.0
DEFAULT_START_FREQ_MULT = 0.9
DEFAULT_START_AMP_MULT = 1.2
DEFAULT_STARTUP_RAMP_TIME = 2.0


@dataclass(frozen=True)
class WaveformDefaults:
    hip_freq_hz: float = DEFAULT_HIP_FREQ_HZ
    leg_amp_deg: float = DEFAULT_LEG_AMP_DEG
    t_wait: float = DEFAULT_T_WAIT
    start_freq_mult: float = DEFAULT_START_FREQ_MULT
    start_amp_mult: float = DEFAULT_START_AMP_MULT
    startup_ramp_time: float = DEFAULT_STARTUP_RAMP_TIME

    @property
    def hip_omega(self) -> float:
        return self.hip_freq_hz * 2.0 * math.pi

    @property
    def leg_amp_rad(self) -> float:
        return math.radians(self.leg_amp_deg)


DEFAULT_WAVEFORM = WaveformDefaults()


def startup_sine_reference(
    t: float,
    hip_omega: float,
    leg_amp_rad: float,
    t_wait: float,
    start_amp_mult: float,
    start_freq_mult: float,
    ramp_time: float = 0.0,
) -> tuple[float, float]:
    """Return position and velocity targets for the startup gait waveform.

    `ramp_time <= 0` preserves the legacy abrupt handoff after the first
    startup half-cycle. Positive `ramp_time` blends both amplitude and
    frequency down to steady-state while keeping phase and velocity continuous.
    """

    if t <= t_wait:
        return 0.0, 0.0

    w_start = hip_omega * start_freq_mult
    w_steady = hip_omega
    if abs(w_start) < 1e-12 or abs(w_steady) < 1e-12:
        return 0.0, 0.0

    amp_start = start_amp_mult * leg_amp_rad
    amp_steady = leg_amp_rad
    startup_end = t_wait + math.pi / w_start

    if t < startup_end:
        phase = w_start * (t - t_wait)
        return amp_start * math.sin(phase), amp_start * w_start * math.cos(phase)

    transition_dt = t - startup_end
    if ramp_time <= 0.0:
        phase = math.pi + w_steady * transition_dt
        return amp_steady * math.sin(phase), amp_steady * w_steady * math.cos(phase)

    if transition_dt < ramp_time:
        amp_slope = (amp_steady - amp_start) / ramp_time
        omega_slope = (w_steady - w_start) / ramp_time
        amplitude = amp_start + amp_slope * transition_dt
        omega = w_start + omega_slope * transition_dt
        phase = math.pi + w_start * transition_dt + 0.5 * omega_slope * transition_dt * transition_dt
        position = amplitude * math.sin(phase)
        velocity = amp_slope * math.sin(phase) + amplitude * omega * math.cos(phase)
        return position, velocity

    omega_slope = (w_steady - w_start) / ramp_time
    phase_at_ramp_end = math.pi + w_start * ramp_time + 0.5 * omega_slope * ramp_time * ramp_time
    steady_dt = transition_dt - ramp_time
    phase = phase_at_ramp_end + w_steady * steady_dt
    return amp_steady * math.sin(phase), amp_steady * w_steady * math.cos(phase)
