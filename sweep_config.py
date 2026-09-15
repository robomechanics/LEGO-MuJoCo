"""Configuration for geometry and actuation sweep runs.

Coordinate convention:
    +x = forward walking direction
    +y = robot-left / lateral
    +z = down

Choose any number of sweep axes from:
    - "curve_x"
    - "curve_y"
    - "curve_z"
    - "box_x"
    - "box_y"
    - "box_z"
    - "foot_x" (metres, positive inward; matches closed-loop FOOT_X)
    - "foot_y" (metres, positive forward; matches closed-loop FOOT_Y)
    - "amp_deg"
    - "start_freq_mult"

The active axes are defined by `SWEEP_AXES`. Any parameter not being swept stays
fixed at its base/default value unless it is listed in `NORMAL_TRIAL_DISTRIBUTIONS`.
"""

from __future__ import annotations

from control_waveform import (
    DEFAULT_KP,
    DEFAULT_KD,
    DEFAULT_TORQUE_LIMIT,
    DEFAULT_GAIN_DISTRIBUTIONS,
    DEFAULT_HIP_FREQ_HZ,
    DEFAULT_LEG_AMP_DEG,
    DEFAULT_START_AMP_MULT,
    DEFAULT_START_FREQ_MULT,
    DEFAULT_STARTUP_RAMP_TIME,
)

FOOT_OFFSET_SWEEP_AXES = ("foot_x", "foot_y")
GEOMETRY_SHAPE_SWEEP_AXES = ("curve_x", "curve_y", "curve_z", "box_x", "box_y", "box_z")
GEOMETRY_SWEEP_AXES = GEOMETRY_SHAPE_SWEEP_AXES + FOOT_OFFSET_SWEEP_AXES
ACTUATION_SWEEP_AXES = ("amp_deg", "start_freq_mult")
ALLOWED_SWEEP_AXES = GEOMETRY_SWEEP_AXES + ACTUATION_SWEEP_AXES

GEOMETRY_BASE = {
    "curve_x": 0.78,
    "curve_y": 0.936,
    "curve_z": 0.78,
    "box_x": 0.667,
    "box_y": 0.24,
    "box_z": 0.1206,
    "foot_x": 0.004,
    "foot_y": -0.023,
}

ACTUATION_BASE = {
    "amp_deg": DEFAULT_LEG_AMP_DEG,
    "start_freq_mult": DEFAULT_START_FREQ_MULT,
}

SWEEP_BASE = {
    **GEOMETRY_BASE,
    **ACTUATION_BASE,
}

def linear_values(low: float, high: float, count: int = 11) -> list[float]:
    return [round(low + (high - low) * i / (count - 1), 9) for i in range(count)]


def scaled_values(base_value: float, percent_deltas: list[float]) -> list[float]:
    return [round(base_value * (1.0 + pct / 100.0), 9) for pct in percent_deltas]


SWEEP_RANGE_SPECS = {
    "curve_x": {"mode": "scaled", "default": GEOMETRY_BASE["curve_x"], "low": -15.0, "high": 10.0, "count": 11},
    "curve_y": {"mode": "scaled", "default": GEOMETRY_BASE["curve_y"], "low": -10.0, "high": 25.0, "count": 11},
    "curve_z": {"mode": "scaled", "default": GEOMETRY_BASE["curve_z"], "low": -30.0, "high": 30.0, "count": 11},
    "box_x": {"mode": "scaled", "default": GEOMETRY_BASE["box_x"], "low": -30.0, "high": 80.0, "count": 11},
    "box_y": {"mode": "scaled", "default": GEOMETRY_BASE["box_y"], "low": -30.0, "high": 80.0, "count": 11},
    "box_z": {"mode": "scaled", "default": GEOMETRY_BASE["box_z"], "low": -30.0, "high": 30.0, "count": 11},
    "foot_x": {"mode": "linear", "low": -0.05, "high": 0.04, "count": 11},
    "foot_y": {"mode": "linear", "low": -0.05, "high": 0.02, "count": 11},
    "amp_deg": {"mode": "linear", "low": 35.0, "high": 50.0, "count": 11},
    "start_freq_mult": {"mode": "linear", "low": 0.8, "high": 2.2, "count": 11},
}


def sweep_values_from_spec(spec: dict) -> list[float]:
    mode = spec["mode"]
    count = int(spec.get("count", 11))
    if mode == "linear":
        return linear_values(float(spec["low"]), float(spec["high"]), count)
    if mode == "scaled":
        percent_deltas = linear_values(float(spec["low"]), float(spec["high"]), count)
        return scaled_values(float(spec["default"]), percent_deltas)
    raise ValueError(f"Unsupported sweep range mode: {mode}")


SWEEP_VALUES = {
    axis_name: sweep_values_from_spec(spec)
    for axis_name, spec in SWEEP_RANGE_SPECS.items()
}

# Pick the active geometry sweep axes here. Any non-empty subset is valid.
SWEEP_AXES = ("curve_x", "box_x")

RUNS_PER_POINT = 1
RUNS_PER_PAIR = RUNS_PER_POINT  # Backward-compatible alias.
DEFAULT_TRIALS_PER_POINT = 75
DEFAULT_TRIALS_PER_PAIR = DEFAULT_TRIALS_PER_POINT  # Backward-compatible alias.
TRIALS_PER_AXIS_SET = {}
TRIALS_PER_PAIRING = TRIALS_PER_AXIS_SET  # Backward-compatible alias.

RUNNER_VERBOSE = True
MAX_WORKERS = 12
TRIAL_WORKERS_PER_PAIR = None

OUTPUT_XML = "modified_model.xml"
FOLDER_NAME = "cxbx0"
SWEEP_DIR = f"data/sweeps/{FOLDER_NAME}"
RESULTS_CSV = f"{SWEEP_DIR}/sweep_results.csv"
GEOMETRY_CACHE_DIR = f"{SWEEP_DIR}/meshes"
SAVE_MESH_CACHE = False
WRITE_FINAL_XML_SNAPSHOT = False
PLOT_RESULTS_AT_END = True
PLOT_RESULTS_PATH = f"{SWEEP_DIR}/sweep_results_plot.png"
PLOT_COLORBAR = "pitch"  # Options: "distance", "velocity", "roll", "pitch".
PREVIEW_FIRST_TRIAL = False
OVERWRITE_RESULTS_CSV = True

# Fixed trial parameters for every simulation in the geometry sweep.
FIXED_TRIAL_PARAMS = {
    "Kp": DEFAULT_KP,
    "Kd": DEFAULT_KD,
    "foot_x": GEOMETRY_BASE["foot_x"],
    "foot_y": GEOMETRY_BASE["foot_y"],
    "torque_limit": DEFAULT_TORQUE_LIMIT,
    "amp_deg": DEFAULT_LEG_AMP_DEG,
    "freq_hz": DEFAULT_HIP_FREQ_HZ,
    "ramp_time": DEFAULT_STARTUP_RAMP_TIME,
}

# Whether startup ramp-down is enabled at all. If False, `ramp_time` stays
# fixed at `FIXED_TRIAL_PARAMS["ramp_time"]`.
USE_RAMPED_START = False

# Gaussian trial sampling. Each parameter is sampled independently from a
# clipped normal distribution.
NORMAL_TRIAL_DISTRIBUTIONS = {
    "Kp": {**DEFAULT_GAIN_DISTRIBUTIONS["Kp"], "use": False},
    "Kd": {**DEFAULT_GAIN_DISTRIBUTIONS["Kd"], "use": False},
    "curve_x": {"use": True,
                "mean": GEOMETRY_BASE["curve_x"],
                "std": 0.02,
                "min": 0.5*GEOMETRY_BASE["curve_x"],
                "max": 1.5*GEOMETRY_BASE["curve_x"]},
    "curve_y": {"use": False,
                "mean": GEOMETRY_BASE["curve_y"],
                "std": 0.025,
                "min": 0.5*GEOMETRY_BASE["curve_y"],
                "max": 1.5*GEOMETRY_BASE["curve_y"]},
    "box_x": {"use": True,
              "mean": GEOMETRY_BASE["box_x"],
              "std": 0.02,
              "min": 0.5*GEOMETRY_BASE["box_x"],
              "max": 1.5*GEOMETRY_BASE["box_x"]},
    "box_y": {"use": False,
              "mean": GEOMETRY_BASE["box_y"],
              "std": 0.01,
              "min": 0.5*GEOMETRY_BASE["box_y"],
              "max": 1.5*GEOMETRY_BASE["box_y"]},
    "start_amp_mult": {"use": False,
                       "mean": DEFAULT_START_AMP_MULT, 
                       "std": 0.15, 
                       "min": 0.5*DEFAULT_START_AMP_MULT, 
                       "max": 1.5*DEFAULT_START_AMP_MULT},
    "start_freq_mult": {"use": False,
                        "mean": DEFAULT_START_FREQ_MULT, 
                        "std": 0.15, 
                        "min": 0.5*DEFAULT_START_FREQ_MULT, 
                        "max": 1.5*DEFAULT_START_FREQ_MULT},
}

# NORMAL_TRIAL_DISTRIBUTIONS = {}

if USE_RAMPED_START:
    NORMAL_TRIAL_DISTRIBUTIONS["ramp_time"] = {
        "use": False,
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
    # Placement is geometry, but is applied at trial setup without remeshing.
    return tuple(
        axis_name for axis_name in canonicalize_axes(axis_names)
        if axis_name in ACTUATION_SWEEP_AXES + FOOT_OFFSET_SWEEP_AXES
    )


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
