"""Review support for piecewise seam calibration.

Two services live here:

* ``seam_review`` summarizes the declared separators against the calibrated
  image so an operator can pick calibration points on the original dark line.
* ``finding_seam_conflicts`` checks previously saved annotations against a
  freshly corrected seam. Findings keep their original polygon position and
  recorded cell identifier; the caller surfaces a reminder to re-check the
  ownership of any annotation the new seam passes through.
"""
import numpy as np

from .findings import finding_status
from .geometry import plane_polygon
from .grid import (
    SeamCalibration, adjusted_seam_keys, calibration_objects, pixel_ownership,
    round_edge, seam_positions,
)
from .model import Inspection, Issue, cell_id, mapping
from .placement_check import local_separator


def seam_review(inspection: Inspection, engine, role: str) -> dict:
    context = engine.panel_context(inspection, role)
    panel, plane = context.panel, inspection.plane
    x_spans = _bands(plane.width, plane.cols)
    y_spans = _bands(plane.height, plane.rows)
    radius_x = max(4, min(22, int(0.14 * plane.width / plane.cols)))
    radius_y = max(4, min(22, int(0.14 * plane.height / plane.rows)))
    vertical = [
        _seam_row(panel, "vertical", index, plane, y_spans, radius_x, inspection.grid_adjustments)
        for index in range(1, plane.cols)
    ]
    horizontal = [
        _seam_row(panel, "horizontal", index, plane, x_spans, radius_y, inspection.grid_adjustments)
        for index in range(1, plane.rows)
    ]
    return {
        "revision": inspection.revision,
        "capture": context.capture.id,
        "role": role,
        "plane": [plane.width, plane.height],
        "grid": [plane.rows, plane.cols],
        "vertical": vertical,
        "horizontal": horizontal,
        "calibrated": sorted(list(adjusted_seam_keys(inspection.grid_adjustments))),
        "cells": [[cell.id, cell.bounds] for cell in context.cells],
    }


def _bands(length: int, divisions: int) -> list[slice]:
    bounds = np.rint(np.linspace(0, length, divisions + 1)).astype(int)
    return [slice(int(start), int(end)) for start, end in zip(bounds, bounds[1:])]


def _seam_row(panel, axis: str, index: int, plane, spans, radius, adjustments) -> dict:
    calibrated_seams = {seam.index: seam for seam in calibration_objects(adjustments).get(axis, [])}
    samples = []
    for span in spans:
        middle = (span.start + span.stop) // 2
        if axis == "vertical":
            seam = calibrated_seams.get(index)
            expected = int(round_edge(seam_positions(seam, np.array([middle]))[0])) if seam else \
                int(round_edge(plane.width * index / plane.cols))
            row = local_separator(panel.signal.T, panel.valid.T, expected, radius, span, "vertical")
        else:
            seam = calibrated_seams.get(index)
            expected = int(round_edge(seam_positions(seam, np.array([middle]))[0])) if seam else \
                int(round_edge(plane.height * index / plane.rows))
            row = local_separator(panel.signal, panel.valid, expected, radius, span, "horizontal")
        samples.append({
            "along": middle, "span": [span.start, span.stop],
            "expected": expected, "detected": row["detected"],
            "offset": row["offset"], "status": row["status"], "contrast": row["contrast"],
        })
    calibrated = calibrated_seams.get(index)
    if axis == "vertical":
        label = f"Vertical seam {index} · columns C{index:02d} / C{index + 1:02d}"
    else:
        label = f"Horizontal seam {index} · rows R{index:02d} / R{index + 1:02d}"
    return {
        "axis": axis, "index": index,
        "label": label,
        "calibrated": calibrated.to_dict() if calibrated else None,
        "samples": samples,
    }


def apply_seam_adjustment(inspection: Inspection, raw: dict) -> tuple[Inspection, list[dict]]:
    """Validate and store one seam calibration, returning ownership reminders."""
    raw = mapping(raw, "Seam calibration request")
    seam = SeamCalibration.from_dict(raw, inspection.plane)
    kept = [
        existing for existing in _calibrations(inspection)
        if not (existing.axis == seam.axis and existing.index == seam.index)
    ]
    kept.append(seam)
    document = {
        "vertical": [item.to_dict() for item in kept if item.axis == "vertical"],
        "horizontal": [item.to_dict() for item in kept if item.axis == "horizontal"],
    }
    # Re-parse the whole document so cross-seam ordering is enforced together.
    from .grid import parse_grid_adjustments
    inspection.grid_adjustments = parse_grid_adjustments(document, inspection.plane)
    conflicts = finding_seam_conflicts(inspection, seam)
    return inspection, conflicts


def clear_seam_adjustment(inspection: Inspection, raw: dict) -> Inspection:
    raw = mapping(raw, "Seam reset request")
    axis = raw.get("axis")
    if axis not in ("vertical", "horizontal"):
        raise Issue("A seam axis must be vertical or horizontal.")
    index = raw.get("index")
    kept = [
        seam for seam in _calibrations(inspection)
        if not (seam.axis == axis and seam.index == index)
    ]
    inspection.grid_adjustments = {
        "vertical": [seam.to_dict() for seam in kept if seam.axis == "vertical"],
        "horizontal": [seam.to_dict() for seam in kept if seam.axis == "horizontal"],
    }
    return inspection


def _calibrations(inspection: Inspection) -> list[SeamCalibration]:
    objects = calibration_objects(inspection.grid_adjustments)
    return [*objects["vertical"], *objects["horizontal"]]


def seam_crosses_polygon(seam: SeamCalibration, polygon: np.ndarray, plane) -> bool:
    """Intersection test between the calibrated seam polyline and a polygon."""
    if seam.axis == "vertical":
        along = np.arange(plane.height + 1, dtype=np.float64)
        line = np.column_stack([seam_positions(seam, along), along])
    else:
        along = np.arange(plane.width + 1, dtype=np.float64)
        line = np.column_stack([along, seam_positions(seam, along)])
    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    if seam.axis == "vertical":
        if line[:, 0].max() < min_x or line[:, 0].min() > max_x:
            return False
    else:
        if line[:, 1].max() < min_y or line[:, 1].min() > max_y:
            return False
    if _polygon_contains_any(line, polygon):
        return True
    boundary = np.vstack([polygon, polygon[0]])
    for start, end in zip(line, line[1:]):
        for edge_start, edge_end in zip(boundary, boundary[1:]):
            if _segments_cross(start, end, edge_start, edge_end):
                return True
    return False


def _polygon_contains_any(points: np.ndarray, polygon: np.ndarray) -> bool:
    # Ray casting on plane pixel coordinates.
    inside = np.zeros(len(points), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[1] == b[1]:
            continue
        hit = ((a[1] > points[:, 1]) != (b[1] > points[:, 1]))
        cross_x = a[0] + (points[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        inside ^= hit & (points[:, 0] < cross_x)
    return bool(inside.any())


def _segments_cross(a, b, c, d) -> bool:
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    epsilon = 1e-7
    c1, c2 = cross(a, b, c), cross(a, b, d)
    c3, c4 = cross(c, d, a), cross(c, d, b)
    return ((c1 * c2 < -epsilon and c3 * c4 < -epsilon)
            or (abs(c1) <= epsilon and _on_segment(a, b, c))
            or (abs(c2) <= epsilon and _on_segment(a, b, d))
            or (abs(c3) <= epsilon and _on_segment(c, d, a))
            or (abs(c4) <= epsilon and _on_segment(c, d, b)))


def _on_segment(a, b, p) -> bool:
    return (min(a[0], b[0]) - 1e-7 <= p[0] <= max(a[0], b[0]) + 1e-7
            and min(a[1], b[1]) - 1e-7 <= p[1] <= max(a[1], b[1]) + 1e-7)


def finding_seam_conflicts(inspection: Inspection, seam: SeamCalibration) -> list[dict]:
    """Annotations whose region the corrected seam now passes through.

    The finding's polygon and recorded cell identifier are never rewritten: the
    response is a review reminder with the cell each polygon point now lands in.
    """
    plane = inspection.plane
    if seam.axis == "vertical":
        left_col, right_col = seam.index - 1, seam.index
    else:
        left_col = right_col = None
    # Resolve ownership once; every polygon point then costs one map lookup.
    row_owner, col_owner = pixel_ownership(plane, inspection.grid_adjustments)
    reminders = []
    for finding in inspection.findings:
        polygon = plane_polygon(finding.points, plane)
        if not seam_crosses_polygon(seam, np.asarray(polygon, dtype=np.float64), plane):
            continue
        current_cells = sorted({
            cell_id(int(row_owner[int(y), int(x)]), int(col_owner[int(y), int(x)]))
            for x, y in (np.floor(polygon).astype(int))
        })
        reminder = {
            "finding": finding.id,
            "title": finding.title,
            "recorded_cell": finding.cell,
            "current_cells": current_cells,
            "source_role": finding.source_role,
            "source_status": finding_status(inspection, finding),
        }
        if seam.axis == "vertical":
            reminder["seam_cells"] = [cell_id(row, left_col) for row in range(plane.rows)] + \
                [cell_id(row, right_col) for row in range(plane.rows)]
        else:
            reminder["seam_cells"] = [cell_id(seam.index - 1, col) for col in range(plane.cols)] + \
                [cell_id(seam.index, col) for col in range(plane.cols)]
        reminders.append(reminder)
    return reminders
