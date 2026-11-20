from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .geometry import Cell
from .intensity import IntensityMap
from .model import Plane


@dataclass(frozen=True)
class StringMap:
    records: list[dict]
    heat: np.ndarray
    support: np.ndarray
    report: dict


def physical_runs(cells: list[Cell], plane: Plane) -> dict[int, list[Cell]]:
    groups = defaultdict(list)
    for cell in cells:
        groups[cell.string].append(cell)
    if plane.string_axis == "columns":
        key = lambda cell: (cell.col, cell.row if cell.col % 2 == 0 else -cell.row)
    else:
        key = lambda cell: (cell.row, cell.col if cell.row % 2 == 0 else -cell.col)
    return {number: sorted(groups[number], key=key) for number in sorted(groups)}


def readings_along_run(run: list[Cell], map_data: IntensityMap) -> list[dict]:
    readings = []
    for index, cell in enumerate(run):
        median = map_data.medians[cell.id]
        relative = None
        if median is not None and map_data.panel_median is not None and map_data.panel_median > 1e-6:
            relative = median / map_data.panel_median
        readings.append({
            "cell": cell.id,
            "index": index + 1,
            "signal": median,
            "relative": round(relative, 6) if relative is not None else None,
        })
    return readings


def largest_step(readings: list[dict]) -> dict | None:
    largest = None
    for first, second in zip(readings, readings[1:]):
        a, b = first["relative"], second["relative"]
        if a is None or b is None:
            continue
        change = b - a
        if largest is None or abs(change) > abs(largest["delta"]):
            largest = {"from": first["cell"], "to": second["cell"], "delta": round(change, 6)}
    return largest


def low_runs(readings: list[dict], floor: float) -> list[list[str]]:
    clusters = []
    active = []
    for row in readings:
        value = row["relative"]
        if value is not None and value < floor:
            active.append(row["cell"])
        elif active:
            clusters.append(active)
            active = []
    if active:
        clusters.append(active)
    return clusters


def inspect_strings(cells: list[Cell], plane: Plane, map_data: IntensityMap) -> StringMap:
    heat = np.full((plane.height, plane.width), np.nan, dtype=np.float32)
    support = map_data.comparable.copy()
    groups = physical_runs(cells, plane)
    results = []
    for number, run in groups.items():
        readings = readings_along_run(run, map_data)
        available = [row["relative"] for row in readings if row["relative"] is not None]
        fraction = len(available) / len(run)
        robust = float(np.median(available)) if available else None
        variation = float(np.percentile(available, 90) - np.percentile(available, 10)) if available else None
        low = low_runs(readings, 0.72)
        step = largest_step(readings)
        review = []
        if fraction < 0.7:
            review.append("incomplete run")
        if robust is not None and fraction >= 0.7 and robust < 0.8:
            review.append("dim string")
        if any(len(cluster) >= 2 for cluster in low):
            review.append("adjacent dim cells")
        if step and abs(step["delta"]) > 0.35:
            review.append("large neighboring step")
        results.append({
            "string": number, "cells": [cell.id for cell in run], "readings": readings,
            "coverage": round(fraction, 5),
            "median_relative": round(robust, 5) if robust is not None and fraction >= 0.7 else None,
            "p90_p10_span": round(variation, 5) if variation is not None else None,
            "largest_step": step,
            "low_runs": low,
            "flags": review,
        })
        for cell, row in zip(run, readings):
            if row["relative"] is not None:
                heat[cell.view][support[cell.view]] = row["relative"]
    return StringMap(results, heat, np.isfinite(heat), {
        "groups": len(results), "groups_with_flags": sum(bool(row["flags"]) for row in results),
        "axis": plane.string_axis, "string_width": plane.string_size,
        "weighted": "One median per supported cell in the declared serpentine order",
    })
