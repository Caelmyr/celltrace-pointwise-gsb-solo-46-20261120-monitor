from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell
from .model import Plane


@dataclass(frozen=True)
class PlacementCheck:
    vertical: list[dict]
    horizontal: list[dict]
    corners: list[dict]
    assessment: dict


def median_slices(values: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occupancies = np.sum(support, axis=1)
    profile = np.full(values.shape[0], np.nan, dtype=np.float32)
    enough = occupancies >= values.shape[1] * 0.46
    for index in np.flatnonzero(enough):
        profile[index] = np.median(values[index][support[index]])
    return profile, occupancies / values.shape[1]


def local_separator(values: np.ndarray, support: np.ndarray, position: int,
                    radius: int, span: slice, cross: str) -> dict:
    lo = max(0, position - radius)
    hi = min(values.shape[0], position + radius + 1)
    slab = values[lo:hi, span]
    valid = support[lo:hi, span]
    profile, coverage = median_slices(slab, valid)
    center = position - lo
    weight = np.isfinite(profile)
    output = {
        "axis": cross, "expected": position, "start": span.start, "end": span.stop,
        "search": [lo, hi], "data": int(weight.sum()), "coverage": round(float(np.mean(coverage)), 4),
        "detected": None, "offset": None, "contrast": None, "status": "unsupported",
    }
    if weight.sum() < 4:
        return output
    slanted = np.interp(np.arange(len(profile)), np.flatnonzero(weight), profile[weight])
    smooth = ndimage.gaussian_filter1d(slanted, 0.85, mode="nearest")
    side = max(1, radius // 4)
    before = profile[:side]
    after = profile[-side:]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    if before.size == 0 or after.size == 0:
        return output
    shoulder = min(float(np.median(before)), float(np.median(after)))
    if shoulder < 0.04:
        return {**output, "status": "low shoulders"}
    candidates = local_minima(smooth, weight, center)
    if not candidates:
        return {**output, "status": "ambiguous valley"}
    first = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    contrast = float((shoulder - smooth[first]) / shoulder)
    if contrast < 0.25:
        return {**output, "contrast": round(contrast, 5), "status": "weak separator"}
    if second is not None and abs(first - second) > 2 and abs(smooth[first] - smooth[second]) < 0.06 * shoulder:
        return {**output, "status": "multiple comparable valleys", "contrast": round(contrast, 5)}
    detected = lo + refined_minimum(smooth, first)
    difference = detected - position
    return {
        **output,
        "status": "offset" if abs(difference) >= 3 else "aligned",
        "detected": round(detected, 3), "offset": round(difference, 3), "contrast": round(contrast, 5),
    }


def refined_minimum(values: np.ndarray, index: int) -> float:
    if index == 0 or index == len(values) - 1:
        return float(index)
    previous, current, following = (float(value) for value in values[index - 1:index + 2])
    curvature = previous - 2 * current + following
    if curvature < 1e-7:
        return float(index)
    offset = np.clip(0.5 * (previous - following) / curvature, -0.45, 0.45)
    return float(index + offset)


def local_minima(values: np.ndarray, support: np.ndarray, center: int) -> list[int]:
    candidates = [
        index for index in range(1, len(values) - 1)
        if support[index - 1:index + 2].all()
        and values[index] < values[index - 1] and values[index] <= values[index + 1]
    ]
    return sorted(candidates, key=lambda index: (values[index], abs(index - center), index))


def intervals(length: int, count: int) -> list[slice]:
    bounds = np.rint(np.linspace(0, length, count + 1)).astype(int)
    spans = []
    for start, end in zip(bounds, bounds[1:]):
        margin = max(2, int(0.23 * (end - start)))
        spans.append(slice(int(start + margin), int(end - margin)))
    return spans


def panel_placement_check(panel: PanelImage, plane: Plane, cells: list[Cell]) -> PlacementCheck:
    radius_x = max(4, min(22, int(0.14 * plane.width / plane.cols)))
    radius_y = max(4, min(22, int(0.14 * plane.height / plane.rows)))
    x_spans = intervals(plane.width, plane.cols)
    y_spans = intervals(plane.height, plane.rows)
    nominal_x = {cell.bounds[0] for cell in cells if cell.col > 0}
    nominal_y = {cell.bounds[1] for cell in cells if cell.row > 0}
    vertical = [
        local_separator(panel.signal.T, panel.valid.T, x, radius_x, span, "vertical")
        for x in sorted(nominal_x) for span in y_spans
    ]
    horizontal = [
        local_separator(panel.signal, panel.valid, y, radius_y, span, "horizontal")
        for y in sorted(nominal_y) for span in x_spans
    ]
    offsets = [item["offset"] for item in vertical + horizontal if item["offset"] is not None]
    clipping = corner_samples(panel, plane)
    report = {
        "testable": len(offsets),
        "positions": len(vertical) + len(horizontal),
        "aligned": sum(row["status"] == "aligned" for row in vertical + horizontal),
        "offset": sum(row["status"] == "offset" for row in vertical + horizontal),
        "median_abs_offset": round(float(np.median(np.abs(offsets))), 4) if offsets else None,
        "maximum_abs_offset": round(float(np.max(np.abs(offsets))), 4) if offsets else None,
        "plane_outside_fraction": round(float((~panel.any_support).mean()), 5),
        "scope": "Observed dark separator samples compared to the declared straight grid",
    }
    return PlacementCheck(vertical, horizontal, clipping, report)


def corner_samples(panel: PanelImage, plane: Plane) -> list[dict]:
    radius = min(40, max(8, int(min(plane.height, plane.width) * 0.06)))
    y0 = slice(0, radius)
    y1 = slice(plane.height - radius, plane.height)
    x0 = slice(0, radius)
    x1 = slice(plane.width - radius, plane.width)
    positions = [("top left", y0, x0), ("top right", y0, x1), ("bottom right", y1, x1), ("bottom left", y1, x0)]
    result = []
    for name, ys, xs in positions:
        coverage = panel.any_support[ys, xs]
        usable = panel.valid[ys, xs]
        result.append({
            "corner": name, "bounds": [xs.start, ys.start, xs.stop, ys.stop],
            "coverage": round(float(coverage.mean()), 5),
            "usable": round(float(usable.mean()), 5),
            "clipped": round(float(panel.clipped[ys, xs].mean()), 5),
        })
    return result


def placement_rows(result: PlacementCheck) -> list[dict]:
    rows = []
    for row in result.horizontal + result.vertical:
        marker = f"{row['axis']}-{row['expected']}-{row['start']}"
        rows.append({"marker": marker, **row})
    return rows


def paint_placement_diagnostics(panel: PanelImage, display, result: PlacementCheck):
    from PIL import ImageDraw
    from .render import draw_tag, font, panel_preview

    image = panel_preview(panel, display)
    draw = ImageDraw.Draw(image, "RGBA")
    for line in result.vertical + result.horizontal:
        paint_separator(draw, line, font(10))
    for corner in result.corners:
        color = (116, 225, 188) if corner["coverage"] >= 0.95 else (244, 148, 130)
        x0, y0, x1, y1 = corner["bounds"]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*color, 224), width=1)
        if x1 - x0 >= 25:
            draw_tag(draw, (x0 + 3, y0 + 4), f"{corner['coverage']:.0%}", color, font(9))
    return image


def paint_separator(draw, row, face):
    expected = row["expected"]
    measured = row["detected"]
    start, end = row["start"], row["end"]
    middle = (start + end) / 2
    vertical = row["axis"] == "vertical"
    color = (245, 188, 94) if row["status"] == "offset" else (139, 231, 201)
    expected_points = (expected, start, expected, end) if vertical else (start, expected, end, expected)
    draw.line(expected_points, fill=(179, 205, 207, 126), width=1)
    if measured is None:
        if vertical:
            draw.line((expected - 3, middle - 3, expected + 3, middle + 3), fill=(233, 138, 163, 216))
            draw.line((expected - 3, middle + 3, expected + 3, middle - 3), fill=(233, 138, 163, 216))
        else:
            draw.line((middle - 3, expected - 3, middle + 3, expected + 3), fill=(233, 138, 163, 216))
            draw.line((middle - 3, expected + 3, middle + 3, expected - 3), fill=(233, 138, 163, 216))
        return
    observed_points = (measured, start, measured, end) if vertical else (start, measured, end, measured)
    draw.line(observed_points, fill=(*color, 205), width=2)
    if abs(measured - expected) >= 3:
        label = f"{measured - expected:+.1f}"
        position = (measured + 5, middle - 5) if vertical else (middle + 6, measured + 2)
        draw_tag(draw, position, label, color, face)


def rectification_edges(panel: PanelImage, plane: Plane) -> list[dict]:
    zones = {
        "left": np.s_[:, :max(1, plane.width // 20)],
        "right": np.s_[:, plane.width - max(1, plane.width // 20):],
        "top": np.s_[:max(1, plane.height // 20), :],
        "bottom": np.s_[plane.height - max(1, plane.height // 20):, :],
    }
    output = []
    for key, zone in zones.items():
        sampled = panel.any_support[zone]
        usable = panel.valid[zone]
        gap = panel.calibration_gap[zone]
        clipping = panel.clipped[zone]
        output.append({
            "edge": key,
            "plane_area": int(sampled.size),
            "out_of_frame": int((~sampled).sum()),
            "usable": int(usable.sum()),
            "calibration_gap": int(gap.sum()),
            "clipped": int(clipping.sum()),
            "usable_fraction": round(float(usable.mean()), 5),
        })
    return output


def placement_report(ctx, plane, revision):
    result = panel_placement_check(ctx.panel, plane, ctx.cells)
    render_ready = {
        "revision": revision,
        "capture": ctx.capture.id,
        "plane": [plane.width, plane.height],
        "vertical": result.vertical,
        "horizontal": result.horizontal,
        "corners": result.corners,
        "summary": result.assessment,
        "edges": rectification_edges(ctx.panel, plane),
        "transform": ctx.panel.matrix.tolist(),
    }
    return result, render_ready
