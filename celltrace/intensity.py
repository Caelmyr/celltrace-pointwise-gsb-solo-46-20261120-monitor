from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell, millimeter_scale
from .images import distribution
from .model import AnalysisSettings, Plane
from .quality import supported_blur


@dataclass(frozen=True)
class IntensityMap:
    records: list[dict]
    medians: dict[str, float | None]
    heat: np.ndarray
    comparable: np.ndarray
    panel_median: float | None
    dark_reference: dict[str, float | None]


@dataclass(frozen=True)
class DarkMap:
    mask: np.ndarray
    reference: np.ndarray
    records: list[dict]
    report: dict


def cell_intensity(panel: PanelImage, cells: list[Cell], settings: AnalysisSettings) -> IntensityMap:
    records = []
    reference = {}
    medians = {}
    heat = np.full(panel.signal.shape, np.nan, dtype=np.float32)
    comparable = np.zeros(panel.signal.shape, dtype=bool)
    for cell in cells:
        view = cell.view
        usable = panel.valid[view]
        fraction = float(usable.sum() / cell.pixels)
        stats = distribution(panel.signal[view][usable])
        trust = fraction >= settings.usable_cell
        median = stats["median"] if trust else None
        high_band = stats["p90"] if trust else None
        if high_band is not None and high_band < 0.025:
            high_band = None
        medians[cell.id] = median
        reference[cell.id] = high_band
        row = {
            "cell": cell.id, "string": cell.string, "interior": cell.interior,
            "support": round(fraction, 6), "reliable": bool(trust),
            "median": median, "p10": stats["p10"] if trust else None,
            "p90": stats["p90"] if trust else None,
            "mad": stats["mad"] if trust else None,
            "active_pixels": stats["count"],
        }
        if trust and median is not None:
            heat[view][usable] = median
            comparable[view] = usable
        records.append(row)
    qualified = [median for median in medians.values() if median is not None]
    median = float(np.median(qualified)) if qualified else None
    for row in records:
        local = row["median"]
        row["relative_panel"] = (
            round(local / median, 5) if median is not None and median > 0.0001 and local is not None else None
        )
    return IntensityMap(records, medians, heat, comparable, median, reference)


def dark_area_map(panel: PanelImage, plane: Plane, cells: list[Cell], settings: AnalysisSettings,
                  inventory: IntensityMap) -> DarkMap:
    output = np.zeros(panel.signal.shape, dtype=bool)
    reference = np.full(panel.signal.shape, np.nan, dtype=np.float32)
    records = []
    scale = millimeter_scale(plane)
    for cell in cells:
        anchor = inventory.dark_reference[cell.id]
        if anchor is None:
            continue
        view = cell.view
        valid = panel.valid[view]
        orig = panel.signal[view]
        filtered, weights = supported_blur(orig, valid, 0.75)
        testable = valid & (weights >= 0.73)
        reference[view][testable] = float(anchor)
        ratio = filtered / anchor
        flagged = testable & (ratio < settings.dark_fraction)
        labels, count = ndimage.label(flagged, structure=np.ones((3, 3), dtype=bool))
        counts = np.bincount(labels.ravel())
        x0, y0, _, _ = cell.interior
        for marker, slices in enumerate(ndimage.find_objects(labels, count), start=1):
            if slices is None or counts[marker] < settings.minimum_area:
                continue
            target = labels[slices] == marker
            output[view][slices] |= target
            ys, xs = slices
            intensity = orig[slices][target]
            border_reasons = dark_border_reasons(target, slices, testable)
            yy, xx = np.where(target)
            row = {
                "id": f"{cell.id}-D{marker:03}",
                "cell": cell.id,
                "bounds": [x0 + xs.start, y0 + ys.start, x0 + xs.stop, y0 + ys.stop],
                "center": [round(x0 + xs.start + float(xx.mean()) + 0.5, 2),
                           round(y0 + ys.start + float(yy.mean()) + 0.5, 2)],
                "pixels": int(counts[marker]),
                "cell_fraction": round(float(counts[marker] / valid.sum()), 6),
                "area_mm2": round(float(counts[marker] * scale[0] * scale[1]), 3) if scale else None,
                "mean_signal": round(float(np.mean(intensity)), 5),
                "depth": round(1 - float(np.median(intensity)) / anchor, 4),
                "reference": round(anchor, 5),
                "open_boundary": border_reasons,
            }
            records.append(row)
    records.sort(key=lambda row: (-row["pixels"], row["id"]))
    eligible = np.isfinite(reference)
    return DarkMap(output, reference, records, {
        "count": len(records),
        "covered_cells": len({row["cell"] for row in records}),
        "analysis_pixels": int(eligible.sum()),
        "dark_pixels": int(output.sum()),
        "dark_fraction": round(float(output.sum() / eligible.sum()), 6) if eligible.any() else None,
        "cutoff": settings.dark_fraction,
        "minimum_area": settings.minimum_area,
    })


def dark_border_reasons(component: np.ndarray, slices, valid: np.ndarray) -> list[str]:
    ys, xs = slices
    reasons = []
    if any((
        ys.start == 0 and component[0].any(),
        xs.start == 0 and component[:, 0].any(),
        ys.stop == valid.shape[0] and component[-1].any(),
        xs.stop == valid.shape[1] and component[:, -1].any(),
    )):
        reasons.append("inset boundary")
    crop = (
        slice(max(0, ys.start - 1), min(valid.shape[0], ys.stop + 1)),
        slice(max(0, xs.start - 1), min(valid.shape[1], xs.stop + 1)),
    )
    padded = np.zeros(valid[crop].shape, dtype=bool)
    padded[
        ys.start - crop[0].start:ys.stop - crop[0].start,
        xs.start - crop[1].start:xs.stop - crop[1].start,
    ] = component
    border = ndimage.binary_dilation(padded, structure=np.ones((3, 3), dtype=bool)) & ~padded
    if (border & ~valid[crop]).any():
        reasons.append("unsupported boundary")
    return reasons
