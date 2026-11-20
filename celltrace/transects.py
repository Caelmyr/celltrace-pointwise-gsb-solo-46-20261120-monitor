from dataclasses import dataclass
from math import atan2, degrees, hypot

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell, millimeter_scale
from .images import distribution
from .model import Issue, Plane, integer, real, xy_list


MAX_TRANSECT_SAMPLES = 2048


@dataclass(frozen=True)
class Transect:
    points: tuple[tuple[float, float], ...]
    width: int
    interval: float

    @classmethod
    def from_dict(cls, payload: dict):
        points = xy_list(payload.get("points"), "Transect points", 2, 2, (0, 1), (0, 1))
        width = integer(payload.get("width", 5), "Strip width", 1, 51)
        interval = real(payload.get("interval", 1), "Sampling interval", 0.5, 24)
        if width % 2 == 0:
            raise Issue("Strip width must be an odd number of pixels.")
        return cls(points, width, interval)


@dataclass(frozen=True)
class Strip:
    x: np.ndarray
    y: np.ndarray
    centers: np.ndarray
    distances: np.ndarray
    direction: tuple[float, float]
    endpoints: np.ndarray
    actual_interval: float


def prepare_strip(transect: Transect, plane: Plane) -> Strip:
    endpoints = np.asarray(transect.points) * (plane.width, plane.height)
    dx, dy = endpoints[1] - endpoints[0]
    length = hypot(dx, dy)
    if length < 4:
        raise Issue("A transect must be at least four plane pixels long.")
    samples = int(np.ceil(length / transect.interval)) + 1
    if samples > MAX_TRANSECT_SAMPLES:
        raise Issue(f"A transect can contain at most {MAX_TRANSECT_SAMPLES} positions. Increase the sampling interval.")
    distance = np.linspace(0, length, samples)
    units = dx / length, dy / length
    centers = endpoints[0] + distance[:, None] * units
    offsets = np.arange(transect.width, dtype=np.float64) - transect.width // 2
    xpos = centers[:, 0, None] - units[1] * offsets
    ypos = centers[:, 1, None] + units[0] * offsets
    return Strip(xpos, ypos, centers, distance, units, endpoints, float(distance[1]))


def sample_strip(values: np.ndarray, support: np.ndarray, strip: Strip) -> tuple[np.ndarray, np.ndarray]:
    corrected = np.where(support, values, 0)
    offsets = strip.y - 0.5, strip.x - 0.5
    weights = ndimage.map_coordinates(
        support.astype(np.float32), offsets, order=1, mode="constant", cval=0, prefilter=False,
    )
    accepted = weights > 0.9999
    signal = ndimage.map_coordinates(
        corrected, offsets, order=1, mode="constant", cval=0, prefilter=False,
    )
    signal[~accepted] = np.nan
    return signal, accepted


def strip_cell_indices(cells: list[Cell], centers: np.ndarray, plane: Plane) -> list[str | None]:
    result = [None] * centers.shape[0]
    for cell in cells:
        x0, y0, x1, y1 = cell.bounds
        visible = (
            (centers[:, 0] >= x0) & (centers[:, 0] < x1)
            & (centers[:, 1] >= y0) & (centers[:, 1] < y1)
        )
        for index in np.flatnonzero(visible):
            result[index] = cell.id
    return result


def analyze_transect(panel: PanelImage, plane: Plane, cells: list[Cell], transect: Transect) -> dict:
    strip = prepare_strip(transect, plane)
    values, support = sample_strip(panel.signal, panel.valid, strip)
    counts = support.sum(axis=1)
    dominant = counts >= max(1, (transect.width + 1) // 2)
    across = strip_cell_indices(cells, strip.centers, plane)
    horizontal, vertical = strip.direction
    scale = millimeter_scale(plane)
    millimeters_per_pixel = hypot(horizontal * scale[0], vertical * scale[1]) if scale else None
    measurements = []
    for index, (point, cell) in enumerate(zip(strip.centers, across)):
        chosen = values[index, support[index]]
        value = float(np.median(chosen)) if dominant[index] else None
        measurements.append({
            "index": index + 1,
            "x": round(float(point[0]), 3), "y": round(float(point[1]), 3),
            "distance": round(float(strip.distances[index]), 3),
            "distance_mm": round(strip.distances[index] * millimeters_per_pixel, 3)
                           if millimeters_per_pixel is not None else None,
            "cell": cell,
            "usable": int(counts[index]),
            "support": round(float(counts[index] / transect.width), 4),
            "signal": round(value, 6) if value is not None else None,
            "spread": round(float(np.ptp(chosen)), 6) if value is not None else None,
        })
    trace_values = np.array([row["signal"] if row["signal"] is not None else np.nan for row in measurements])
    span = {
        "cells": list(dict.fromkeys(cell for cell in across if cell is not None)),
        "data_positions": int(dominant.sum()),
        "positions": int(values.shape[0]),
        "width": transect.width,
        "angle": round(degrees(atan2(vertical, horizontal)), 3),
        "pixels_traveled": round(float(strip.distances[-1]), 3),
        "mm_traveled": measurements[-1]["distance_mm"],
        "interval": round(strip.actual_interval, 3),
        "signal": distribution(trace_values),
        "missing_intervals": unavailable_intervals(measurements),
        "deepest_valleys": notable_valleys(measurements, trace_values),
    }
    return {"summary": span, "rows": measurements}


def unavailable_intervals(measurements: list[dict]) -> list[dict]:
    first = None
    intervals = []
    for index, row in enumerate(measurements):
        if row["signal"] is None:
            if first is None:
                first = index
            continue
        if first is not None:
            intervals.append(interval(first, index - 1, measurements))
            first = None
    if first is not None:
        intervals.append(interval(first, len(measurements) - 1, measurements))
    return intervals


def interval(start: int, end: int, data: list[dict]) -> dict:
    return {"start": data[start]["distance"], "end": data[end]["distance"], "positions": end - start + 1}


def notable_valleys(measurements: list[dict], signal: np.ndarray) -> list[dict]:
    if signal.size < 7:
        return []
    valleys = []
    for index in range(3, signal.size - 3):
        values = signal[index - 3:index + 4]
        if not np.isfinite(values).all():
            continue
        current = values[3]
        if current > values[2] or current >= values[4]:
            continue
        reference = min(np.max(values[:3]), np.max(values[4:]))
        drop = reference - current
        if drop >= 0.02:
            valleys.append({
                "distance": measurements[index]["distance"],
                "cell": measurements[index]["cell"],
                "signal": round(float(current), 5),
                "contrast": round(float(drop), 5),
            })
    return sorted(valleys, key=lambda valley: -valley["contrast"])[:12]
