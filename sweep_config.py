"""Configuration for geometry and actuation sweep runs.

Coordinate convention:
    +x = forward walking direction
    +y = robot-left / lateral
    +z = down

Choose any number of sweep axes from:
    - "curve_x"
    - "curve_y"
    - "box_x"
    - "box_y"
    - "amp_deg"
    - "start_freq_mult"

The active axes are defined by `SWEEP_AXES`. Any parameter not being swept stays
fixed at its base/default value unless it is listed in `NORMAL_TRIAL_DISTRIBUTIONS`.
"""

from __future__ import annotations

from control_waveform import (
    DEFAULT_HIP_FREQ_HZ,
    DEFAULT_LEG_AMP_DEG,
    DEFAULT_START_AMP_MULT,
    DEFAULT_START_FREQ_MULT,
    DEFAULT_STARTUP_RAMP_TIME,
)

GEOMETRY_SWEEP_AXES = ("curve_x", "curve_y", "box_x", "box_y")
ACTUATION_SWEEP_AXES = ("amp_deg", "start_freq_mult")
ALLOWED_SWEEP_AXES = GEOMETRY_SWEEP_AXES + ACTUATION_SWEEP_AXES

GEOMETRY_BASE = {
    "curve_x": 0.78,
    "curve_y": 0.936,
    "box_x": 0.667,
    "box_y": 0.24,
}

ACTUATION_BASE = {
    "amp_deg": DEFAULT_LEG_AMP_DEG,
    "start_freq_mult": DEFAULT_START_FREQ_MULT,
}

SWEEP_BASE = {
    **GEOMETRY_BASE,
    **ACTUATION_BASE,
}

PERCENT_DELTAS = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]


def scaled_values(base_value: float, percent_deltas: list[float]) -> list[float]:
    return [round(base_value * (1.0 + pct / 100.0), 6) for pct in percent_deltas]


SWEEP_VALUES = {
    axis_name: scaled_values(base_value, PERCENT_DELTAS)
    for axis_name, base_value in SWEEP_BASE.items()
}

# Pick the active geometry sweep axes here. Any non-empty subset is valid.
SWEEP_AXES = ("curve_x", "box_x")

RUNS_PER_POINT = 1
RUNS_PER_PAIR = RUNS_PER_POINT  # Backward-compatible alias.
DEFAULT_TRIALS_PER_POINT = 30
DEFAULT_TRIALS_PER_PAIR = DEFAULT_TRIALS_PER_POINT  # Backward-compatible alias.
TRIALS_PER_AXIS_SET = {
    ("curve_x",): 30,
    ("curve_y",): 30,
    ("box_x",): 30,
    ("box_y",): 30,
    ("amp_deg",): 30,
    ("start_freq_mult",): 30,
    ("curve_x", "curve_y"): 30,
    ("curve_x", "box_x"): 3,
    ("curve_x", "box_y"): 30,
    ("curve_x", "amp_deg"): 30,
    ("curve_x", "start_freq_mult"): 30,
    ("curve_y", "box_x"): 30,
    ("curve_y", "box_y"): 30,
    ("curve_y", "amp_deg"): 30,
    ("curve_y", "start_freq_mult"): 30,
    ("box_x", "box_y"): 30,
    ("box_x", "amp_deg"): 30,
    ("box_x", "start_freq_mult"): 30,
    ("box_y", "amp_deg"): 30,
    ("box_y", "start_freq_mult"): 30,
    ("amp_deg", "start_freq_mult"): 30,
    ("curve_x", "curve_y", "box_x"): 30,
    ("curve_x", "curve_y", "box_y"): 30,
    ("curve_x", "box_x", "box_y"): 30,
    ("curve_y", "box_x", "box_y"): 30,
    ("curve_x", "curve_y", "box_x", "box_y"): 30,
}
TRIALS_PER_PAIRING = TRIALS_PER_AXIS_SET  # Backward-compatible alias.

RUNNER_VERBOSE = True
MAX_WORKERS = None
TRIAL_WORKERS_PER_PAIR = None

OUTPUT_XML = "modified_model.xml"
RESULTS_CSV = "sweep_results.csv"
OVERWRITE_RESULTS_CSV = True

# Fixed trial parameters for every simulation in the geometry sweep.
FIXED_TRIAL_PARAMS = {
    "foot_x": 0.0,
    "foot_y": 0.0,
    "torque_limit": 25.0,
    "amp_deg": DEFAULT_LEG_AMP_DEG,
    "freq_hz": DEFAULT_HIP_FREQ_HZ,
    "ramp_time": DEFAULT_STARTUP_RAMP_TIME,
}

# Whether startup ramp-down is enabled at all. If False, `ramp_time` stays
# fixed at `FIXED_TRIAL_PARAMS["ramp_time"]`.
USE_RAMPED_START = True

# Gaussian trial sampling. Each parameter is sampled independently from a
# clipped normal distribution.
NORMAL_TRIAL_DISTRIBUTIONS = {
    "Kp": {"mean": 90.0, "std": 4.0, "min": 80.0, "max": 100.0},
    "Kd": {"mean": 7.0, "std": 2.0, "min": 2.0, "max": 15.0},
    "start_amp_mult": {"mean": DEFAULT_START_AMP_MULT, "std": 0.2, "min": 0.8, "max": 1.8},
    "start_freq_mult": {"mean": DEFAULT_START_FREQ_MULT, "std": 0.15, "min": 0.7, "max": 1.4},
}

if USE_RAMPED_START:
    NORMAL_TRIAL_DISTRIBUTIONS["ramp_time"] = {
        "mean": 1.0,
        "std": 0.35,
        "min": 0.1,
        "max": 3.0,
    }


def canonicalize_axes(axis_names: tuple[str, ...]) -> tuple[str, ...]:
    if not axis_names:
        raise ValueError("Expected at least 1 sweep axis.")
    if len(set(axis_names)) != len(axis_names):
        raise ValueError(f"Sweep axes must be distinct, got {axis_names}.")
    for axis_name in axis_names:
        if axis_name not in ALLOWED_SWEEP_AXES:
            raise ValueError(f"Unsupported sweep axis '{axis_name}'.")
    return axis_names


def geometry_axis_names(axis_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(axis_name for axis_name in canonicalize_axes(axis_names) if axis_name in GEOMETRY_SWEEP_AXES)


def trial_axis_names(axis_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(axis_name for axis_name in canonicalize_axes(axis_names) if axis_name in ACTUATION_SWEEP_AXES)


def canonicalize_axis_pair(axis_pair: tuple[str, str]) -> tuple[str, str]:
    canonical_axes = canonicalize_axes(axis_pair)
    if len(canonical_axes) != 2:
        raise ValueError(f"Expected exactly 2 sweep axes, got {axis_pair}.")
    return canonical_axes  # type: ignore[return-value]


def trials_for_axes(axis_names: tuple[str, ...]) -> int:
    canonical_axis_names = canonicalize_axes(axis_names)
    value = TRIALS_PER_AXIS_SET.get(canonical_axis_names)
    if value is None:
        value = TRIALS_PER_AXIS_SET.get(tuple(reversed(canonical_axis_names)))
    if value is None:
        value = TRIALS_PER_AXIS_SET.get(tuple(sorted(canonical_axis_names)))
    if value is not None:
        return int(value)
    return int(DEFAULT_TRIALS_PER_POINT)


def trials_for_pairing(axis_pair: tuple[str, str]) -> int:
    return trials_for_axes(axis_pair)


# Backward-compatible aliases used by some plotting/recording scripts.
BASE_X = GEOMETRY_BASE["curve_x"]
BASE_Y = GEOMETRY_BASE["curve_y"]
X_VALUES = SWEEP_VALUES["curve_x"]
Y_VALUES = SWEEP_VALUES["curve_y"]
