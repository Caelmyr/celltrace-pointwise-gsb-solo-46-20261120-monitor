from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell
from .images import distribution, histogram_rows
from .model import AnalysisSettings, Capture


@dataclass(frozen=True)
class QualityMap:
    layers: dict[str, np.ndarray]
    records: list[dict]
    report: dict


def supported_blur(values: np.ndarray, mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    weights = ndimage.gaussian_filter(mask.astype(np.float32), sigma, mode="constant")
    signal = ndimage.gaussian_filter(np.where(mask, values, 0).astype(np.float32), sigma, mode="constant")
    result = np.zeros(mask.shape, dtype=np.float32)
    np.divide(signal, weights, where=weights > 0.002, out=result)
    return result, weights


def local_contrast(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lowpass, weights = supported_blur(values, mask, 1.35)
    contrast = np.zeros(mask.shape, dtype=np.float32)
    applicable = mask & (weights >= 0.78)
    contrast[applicable] = abs(values[applicable] - lowpass[applicable])
    return contrast, applicable


def capture_quality(panel: PanelImage, capture: Capture, cells: list[Cell],
                    settings: AnalysisSettings) -> QualityMap:
    contrasts, applicable = local_contrast(panel.signal, panel.valid)
    deep_floor = panel.valid & (panel.signal < 0.025)
    records = []
    for cell in cells:
        view = cell.view
        support = panel.valid[view]
        raster = panel.signal[view]
        counts = {
            "expected_pixels": cell.pixels,
            "usable_pixels": int(support.sum()),
            "clipped_pixels": int(panel.clipped[view].sum()),
            "reference_gap_pixels": int(panel.calibration_gap[view].sum()),
            "outside_pixels": int((~panel.any_support[view]).sum()),
            "near_floor_pixels": int(deep_floor[view].sum()),
        }
        fraction = support.sum() / cell.pixels
        grad = contrasts[view][applicable[view]]
        flags = []
        if fraction < settings.usable_cell:
            flags.append("insufficient support")
        if counts["clipped_pixels"] > cell.pixels * 0.04:
            flags.append("clipped highlights")
        if counts["near_floor_pixels"] > cell.pixels * 0.25:
            flags.append("low signal")
        if grad.size > 30 and np.percentile(grad, 90) < 0.007:
            flags.append("little local texture")
        records.append({
            "cell": cell.id,
            "string": cell.string,
            **counts,
            "usable_fraction": round(float(fraction), 5),
            "local_contrast_p90": round(float(np.percentile(grad, 90)), 5) if grad.size else None,
            "median": distribution(raster[support])["median"],
            "flags": flags,
        })
    overall = int(panel.valid.sum())
    report = {
        "capture_id": capture.id,
        "bit_depth": capture.bit_depth,
        "white_level": capture.white_level,
        "plane_size": list(panel.signal.shape[::-1]),
        "usable_pixels": overall,
        "total_pixels": int(panel.signal.size),
        "clipped_pixels": int(panel.clipped.sum()),
        "calibration_gap_pixels": int(panel.calibration_gap.sum()),
        "outside_pixels": int((~panel.any_support).sum()),
        "near_floor_pixels": int(deep_floor.sum()),
        "cell_review_count": sum(bool(row["flags"]) for row in records),
        "histogram": histogram_rows(panel.signal[panel.valid], (0, 1.2), 28),
        "signal": distribution(panel.signal[panel.valid]),
    }
    return QualityMap({
        "clipped": panel.clipped, "invalid calibration": panel.calibration_gap,
        "outside source": ~panel.any_support, "near floor": deep_floor,
    }, records, report)


def source_quality(values: np.ndarray, white_level: int) -> dict:
    normalized = values.astype(np.float32) / white_level
    clipped = values >= white_level
    zero = values == 0
    marginal = np.s_[::max(1, values.shape[0] // 500), ::max(1, values.shape[1] // 500)]
    reduced = normalized[marginal]
    rows = np.median(reduced, axis=1)
    columns = np.median(reduced, axis=0)
    finite = np.ones(reduced.shape, dtype=bool)
    contrast, good = local_contrast(reduced, finite)
    return {
        "size": [int(values.shape[1]), int(values.shape[0])],
        "signal": distribution(normalized),
        "histogram": histogram_rows(normalized, (0, 1), 24),
        "clipped_pixels": int(clipped.sum()),
        "zero_pixels": int(zero.sum()),
        "rows": profile_summary(rows),
        "columns": profile_summary(columns),
        "local_contrast": distribution(contrast[good]),
    }


def profile_summary(values: np.ndarray) -> list[float]:
    if values.size <= 40:
        return [round(float(value), 5) for value in values]
    return [round(float(np.median(block)), 5) for block in np.array_split(values, 40)]
