from collections import defaultdict
from dataclasses import dataclass
from math import atan2, degrees, hypot

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell, millimeter_scale
from .model import AnalysisSettings, Plane
from .quality import supported_blur


@dataclass(frozen=True)
class CrackMap:
    evidence: np.ndarray
    skeleton: np.ndarray
    exclusions: np.ndarray
    busbars: np.ndarray
    records: list[dict]
    report: dict


CONNECT = np.ones((3, 3), dtype=bool)
NEIGHBORS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)
EDGES = ((0, 1), (1, 0), (1, 1), (-1, 1))


def crack_evidence(panel: PanelImage, plane: Plane, cells: list[Cell], settings: AnalysisSettings,
                   dark_areas: np.ndarray) -> CrackMap:
    all_candidates = np.zeros(panel.signal.shape, dtype=bool)
    skeleton = np.zeros(panel.signal.shape, dtype=bool)
    exclusions = np.zeros(panel.signal.shape, dtype=bool)
    busbars = np.zeros(panel.signal.shape, dtype=bool)
    records = []
    scale = millimeter_scale(plane)
    for cell in cells:
        view = cell.view
        owned = cell.interior_mask(view)
        image = panel.signal[view]
        valid = panel.valid[view] & owned
        exclusion = cell_exclusions(valid, settings.ignore_margin, dark_areas[view] & owned)
        bar = busbar_bands(cell, settings) & owned
        busbars[view] |= bar
        exclusion |= bar
        exclusions[view] |= exclusion
        response = valley_response(image, valid)
        threshold = (response >= settings.crack_contrast) & ~exclusion
        components, count = ndimage.label(threshold, structure=CONNECT)
        areas = np.bincount(components.ravel())
        for component_id, region in enumerate(ndimage.find_objects(components, count), 1):
            if region is None or areas[component_id] < settings.minimum_length:
                continue
            piece = components[region] == component_id
            if not slender_enough(piece, settings.minimum_length):
                continue
            trace = thin(piece)
            shape = trace_metrics(trace, scale)
            if shape["length_px"] < settings.minimum_length:
                continue
            metrics = piece_metrics(piece, trace, response[region], shape)
            if metrics["mean_width_px"] > 5.5:
                continue
            all_candidates[view][region] |= piece
            skeleton[view][region] |= trace
            records.append(make_record(cell, component_id, region, trace, metrics))
    records.sort(key=lambda row: (-row["length_px"], row["id"]))
    return CrackMap(all_candidates, skeleton, exclusions, busbars, records, {
        "count": len(records),
        "cells_with_traces": len({row["cell"] for row in records}),
        "trace_length_px": round(sum(row["length_px"] for row in records), 3),
        "trace_length_mm": round(sum(row["length_mm"] for row in records), 3) if scale else None,
        "crack_contrast": settings.crack_contrast,
        "minimum_length": settings.minimum_length,
        "excluded_pixels": int(exclusions.sum()),
        "busbar_pixels": int(busbars.sum()),
        "busbar_positions": settings.busbars,
        "busbar_half_width": settings.busbar_radius,
        "measurement": "Connected valley centerlines within individual cell interiors",
    })


def busbar_bands(cell: Cell, settings: AnalysisSettings) -> np.ndarray:
    bound_left, _, bound_right, _ = cell.bounds
    left, top, right, bottom = cell.interior
    result = np.zeros((bottom - top, right - left), dtype=bool)
    for fraction in settings.busbars:
        image_x = bound_left + fraction * (bound_right - bound_left)
        center = int(np.floor(image_x)) - left
        x0 = max(0, center - settings.busbar_radius)
        x1 = min(right - left, center + settings.busbar_radius + 1)
        if x0 < x1:
            result[:, x0:x1] = True
    return result


def cell_exclusions(valid: np.ndarray, clearance: int, dark: np.ndarray) -> np.ndarray:
    exclusion = ~valid | dark
    if clearance:
        exclusion = ndimage.binary_dilation(exclusion, iterations=clearance, border_value=1)
    return exclusion


def valley_response(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lowpass, weights = supported_blur(image, valid, 2.1)
    contrast = np.zeros(image.shape, dtype=np.float32)
    supported = valid & (weights > 0.87) & (lowpass > 0.035)
    np.divide(lowpass - np.where(valid, image, 0), lowpass, where=supported, out=contrast)
    return contrast


def slender_enough(region: np.ndarray, length: int) -> bool:
    height, width = region.shape
    if max(width, height) < length * 0.56:
        return False
    density = region.sum() / (height * width)
    return density < 0.83 or min(width, height) <= 5


def thin(foreground: np.ndarray) -> np.ndarray:
    image = np.pad(foreground, 1, constant_values=False)
    height, width = foreground.shape
    for _ in range(max(height, width)):
        changed = False
        for parity in (False, True):
            remove = suppressible_pixels(image, parity)
            if remove.any():
                image[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1]


def suppressible_pixels(padded: np.ndarray, parity: bool) -> np.ndarray:
    height, width = padded.shape
    neighbors = [
        padded[1 + row:height - 1 + row, 1 + col:width - 1 + col]
        for row, col in NEIGHBORS
    ]
    counts = np.zeros((height - 2, width - 2), dtype=np.uint8)
    transitions = np.zeros_like(counts)
    for index, neighbor in enumerate(neighbors):
        counts += neighbor
        transitions += ~neighbor & neighbors[(index + 1) % len(neighbors)]
    north, east, south, west = neighbors[0], neighbors[2], neighbors[4], neighbors[6]
    if parity:
        border = ~(north & east & west) & ~(north & south & west)
    else:
        border = ~(north & east & south) & ~(east & south & west)
    return padded[1:-1, 1:-1] & (counts >= 2) & (counts <= 6) & (transitions == 1) & border


def trace_metrics(trace: np.ndarray, scale: tuple[float, float] | None) -> dict:
    graph = trace_graph(trace)
    endpoints = [point for point, adjacent in graph.items() if len(adjacent) == 1]
    branches = [point for point, adjacent in graph.items() if len(adjacent) > 2]
    length_px = 0.0
    length_mm = 0.0
    for (ay, ax), adjacent in graph.items():
        for by, bx in adjacent:
            if (by, bx) <= (ay, ax):
                continue
            dx, dy = bx - ax, by - ay
            length_px += hypot(dx, dy)
            if scale is not None:
                length_mm += hypot(dx * scale[0], dy * scale[1])
    return {
        "length_px": round(length_px, 3), "length_mm": round(length_mm, 3) if scale else None,
        "endpoints": sorted(endpoints), "junctions": sorted(branches),
        "paths": compact_paths(graph), "axis_degrees": principal_axis(trace),
    }


def trace_graph(trace: np.ndarray) -> dict:
    points = {tuple(int(index) for index in point) for point in np.argwhere(trace)}
    graph = defaultdict(set)
    for ay, ax in sorted(points):
        graph[ay, ax]
        for dy, dx in EDGES:
            by, bx = ay + dy, ax + dx
            if (by, bx) not in points:
                continue
            if dx and dy and ((ay, bx) in points or (by, ax) in points):
                continue
            graph[ay, ax].add((by, bx))
            graph[by, bx].add((ay, ax))
    return graph


def edge_key(a, b):
    return min(a, b), max(a, b)


def compact_paths(graph: dict) -> list[list[tuple[int, int]]]:
    seen = set()
    paths = []
    roots = sorted(graph, key=lambda point: (len(graph[point]) == 2, point))
    for root in roots:
        for adjacent in sorted(graph[root]):
            if edge_key(root, adjacent) in seen:
                continue
            path = [root, adjacent]
            seen.add(edge_key(root, adjacent))
            while len(graph[path[-1]]) == 2:
                tail = path[-1]
                following = next(point for point in graph[tail] if point != path[-2])
                key = edge_key(tail, following)
                if key in seen:
                    break
                seen.add(key)
                path.append(following)
            paths.append(remove_collinear(path))
    return paths


def remove_collinear(path: list) -> list:
    if len(path) < 3:
        return path
    straight = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            straight.append(b)
    straight.append(path[-1])
    return straight


def principal_axis(trace: np.ndarray) -> float | None:
    coords = np.argwhere(trace)
    if coords.shape[0] < 3:
        return None
    centered = coords - coords.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    if values[-1] < max(0.1, 1.16 * values[0]):
        return None
    dy, dx = vectors[:, -1]
    return round(degrees(atan2(dy, dx)) % 180, 2)


def piece_metrics(piece: np.ndarray, trace: np.ndarray, response: np.ndarray, metrics: dict) -> dict:
    lengths = max(metrics["length_px"], 1)
    return {
        **metrics, "peak_contrast": round(float(response[piece].max()), 5),
        "median_contrast": round(float(np.median(response[trace])), 5) if trace.any() else None,
        "mean_width_px": round(float(piece.sum() / lengths), 3),
    }


def make_record(cell, number, region, trace, metrics) -> dict:
    y_slice, x_slice = region
    panel_x = cell.interior[0] + x_slice.start
    panel_y = cell.interior[1] + y_slice.start
    centerline_y, centerline_x = np.argwhere(trace).mean(axis=0)

    def global_xy(point):
        return [int(panel_x + point[1]), int(panel_y + point[0])]

    return {
        "id": f"{cell.id}-C{number:03}", "cell": cell.id,
        "bounds": [panel_x, panel_y, panel_x + trace.shape[1], panel_y + trace.shape[0]],
        "center": [round(panel_x + centerline_x + 0.5, 2), round(panel_y + centerline_y + 0.5, 2)],
        **metrics,
        "endpoints": [global_xy(point) for point in metrics["endpoints"]],
        "junctions": [global_xy(point) for point in metrics["junctions"]],
        "paths": [[global_xy(point) for point in path] for path in metrics["paths"]],
    }
