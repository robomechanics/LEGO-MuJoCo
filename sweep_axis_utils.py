"""Shared helpers for sweep-axis metadata and labeling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SweepAxisSpec:
    axis_names: tuple[str, ...]
    value_columns: tuple[str, ...]

    @property
    def axis_x_name(self) -> str:
        return self.axis_names[0]

    @property
    def axis_y_name(self) -> str:
        if len(self.axis_names) < 2:
            raise ValueError("This sweep has fewer than 2 axes.")
        return self.axis_names[1]

    @property
    def x_value_column(self) -> str:
        return self.value_columns[0]

    @property
    def y_value_column(self) -> str:
        if len(self.value_columns) < 2:
            raise ValueError("This sweep has fewer than 2 axes.")
        return self.value_columns[1]


AXIS_LABELS = {
    "curve_x": "Curve X (forward)",
    "curve_y": "Curve Y (left/lateral)",
    "box_x": "Box X (forward length)",
    "box_y": "Box Y (left/lateral width)",
    "amp_deg": "Regular Amplitude",
    "start_freq_mult": "Startup Frequency Multiplier",
}

AXIS_SHORT_LABELS = {
    "curve_x": "Curve X",
    "curve_y": "Curve Y",
    "box_x": "Box X",
    "box_y": "Box Y",
    "amp_deg": "Amp",
    "start_freq_mult": "Start Freq",
}


def percentage_difference(value: float, base: float) -> float:
    return 100.0 * (value / base - 1.0)


def _load_sweep_config():
    try:
        import sweep_config as config
    except ImportError:
        try:
            import xy_sweep_config as config
        except ImportError:
            return None
    return config


def axis_label(axis_name: str) -> str:
    return AXIS_LABELS.get(axis_name, axis_name.replace("_", " ").title())


def axis_short_label(axis_name: str) -> str:
    return AXIS_SHORT_LABELS.get(axis_name, axis_label(axis_name))


def axis_percent_label(axis_name: str) -> str:
    return f"{axis_short_label(axis_name)} Difference (%)"


def infer_base(values: list[float], axis_name: str) -> float:
    config = _load_sweep_config()
    if config is not None:
        try:
            return float(config.SWEEP_BASE[axis_name])
        except (AttributeError, KeyError, TypeError, ValueError):
            try:
                return float(config.GEOMETRY_BASE[axis_name])
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
    return float(np.median(values))


def configured_percent_centers(values: list[float], axis_name: str) -> list[float]:
    config = _load_sweep_config()
    if config is not None:
        try:
            base = float(config.SWEEP_BASE[axis_name])
            configured_values = config.SWEEP_VALUES[axis_name]
            configured_centers = {
                round(percentage_difference(float(value), base), 6)
                for value in configured_values
            }
            observed_centers = {round(float(value), 6) for value in values}
            return sorted(configured_centers | observed_centers)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return values


def _single_nonempty_value(rows: list[dict], column: str) -> str | None:
    values = {
        str(row.get(column, "")).strip()
        for row in rows
        if str(row.get(column, "")).strip()
    }
    if not values:
        return None
    if len(values) != 1:
        joined = ", ".join(sorted(values))
        raise ValueError(f"Expected a single value in column {column}, found: {joined}")
    return next(iter(values))


def validate_sweep_columns(fieldnames: list[str] | None) -> None:
    field_set = set(fieldnames or [])
    required = {"Gait_Quality_Pass", "Distance_Traversed"}
    missing = required - field_set
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_list}")

    has_n_axis = "Sweep_Axes" in field_set
    has_generic = {"Sweep_X_Param", "Sweep_Y_Param", "Sweep_X_Value", "Sweep_Y_Value"} <= field_set
    has_legacy = {"Mesh_X", "Mesh_Y"} <= field_set
    if not has_n_axis and not has_generic and not has_legacy:
        raise ValueError(
            "Missing sweep-axis columns. Expected either "
            "`Sweep_Axes` with per-axis value columns, "
            "`Sweep_X_Param`/`Sweep_Y_Param` with `Sweep_X_Value`/`Sweep_Y_Value`, "
            "or legacy `Mesh_X`/`Mesh_Y` columns."
        )


def resolve_sweep_axes(rows: list[dict], fieldnames: list[str] | None) -> SweepAxisSpec:
    field_set = set(fieldnames or [])
    if "Sweep_Axes" in field_set:
        axes_value = _single_nonempty_value(rows, "Sweep_Axes")
        if axes_value:
            axis_names = tuple(axis_name.strip() for axis_name in axes_value.split(",") if axis_name.strip())
            value_columns = tuple(f"Sweep_{axis_name}_Value" for axis_name in axis_names)
            missing = [column for column in value_columns if column not in field_set]
            if missing:
                raise ValueError(f"Missing per-axis value column(s): {', '.join(missing)}")
            return SweepAxisSpec(axis_names=axis_names, value_columns=value_columns)

    if {"Sweep_X_Param", "Sweep_Y_Param", "Sweep_X_Value", "Sweep_Y_Value"} <= field_set:
        axis_x_name = _single_nonempty_value(rows, "Sweep_X_Param")
        axis_y_name = _single_nonempty_value(rows, "Sweep_Y_Param")
        if axis_x_name and axis_y_name:
            return SweepAxisSpec(
                axis_names=(axis_x_name, axis_y_name),
                value_columns=("Sweep_X_Value", "Sweep_Y_Value"),
            )

    return SweepAxisSpec(
        axis_names=("curve_x", "curve_y"),
        value_columns=("Mesh_X", "Mesh_Y"),
    )


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))
