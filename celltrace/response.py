from dataclasses import dataclass

import numpy as np

from .calibration import PanelImage
from .geometry import Cell
from .images import distribution, histogram_rows
from .model import AnalysisSettings, Capture, Issue


@dataclass(frozen=True)
class CurrentResponse:
    ratio: np.ndarray
    relative: np.ndarray
    eligible: np.ndarray
    common: np.ndarray
    weak: np.ndarray
    records: list[dict]
    report: dict


def explain_ratio_requirements(high: Capture, low: Capture) -> list[str]:
    warnings = []
    if high.current_a is None or low.current_a is None:
        warnings.append("Drive currents are missing. Ratio is labeled by selected role only.")
    elif high.current_a <= low.current_a:
        warnings.append("The selected high-current image must have the larger drive current.")
    if not 0.02 <= low.exposure_ms / high.exposure_ms <= 50:
        warnings.append("Exposure times differ by more than 50×. Verify the metadata.")
    if high.bit_depth != low.bit_depth:
        warnings.append("Capture code depths differ. Ratios use the declared code ranges.")
    return warnings


def response_map(high: PanelImage, low: PanelImage, high_capture: Capture, low_capture: Capture,
                 cells: list[Cell], settings: AnalysisSettings) -> CurrentResponse:
    if high.signal.shape != low.signal.shape:
        raise Issue("Both acquisitions need to use the same rectified panel plane.", 409)
    messages = explain_ratio_requirements(high_capture, low_capture)
    if (
        high_capture.current_a is not None
        and low_capture.current_a is not None
        and high_capture.current_a <= low_capture.current_a
    ):
        raise Issue(messages[0], 409)
    common = high.valid & low.valid
    stable = common & (high.signal > 0.02) & (low.signal > 0.002)
    exposure_scale = high_capture.exposure_ms / low_capture.exposure_ms
    current_ratio = np.full(high.signal.shape, np.nan, dtype=np.float32)
    np.divide(low.signal, high.signal, where=stable, out=current_ratio)
    current_ratio[stable] *= exposure_scale
    eligible = np.zeros(high.signal.shape, dtype=bool)
    rows = []
    references = []
    for cell in cells:
        view = cell.view
        owned = cell.interior_mask(view)
        count_common = int((common[view] & owned).sum())
        count_stable = int((stable[view] & owned).sum())
        qualified = count_stable / cell.pixels >= settings.usable_cell
        stats = distribution(current_ratio[view][stable[view] & owned])
        median_value = stats["median"] if qualified else None
        if median_value is not None:
            references.append(median_value)
            eligible[view][stable[view] & owned] = True
        rows.append({
            "cell": cell.id,
            "string": cell.string,
            "expected_pixels": cell.pixels,
            "common_pixels": count_common,
            "stable_pixels": count_stable,
            "support": round(count_stable / cell.pixels, 5),
            "ratio": median_value,
            "ratio_p10": stats["p10"] if qualified else None,
            "ratio_p90": stats["p90"] if qualified else None,
        })
    reference = float(np.median(references)) if references else None
    relative = np.full(current_ratio.shape, np.nan, dtype=np.float32)
    weak = np.zeros(current_ratio.shape, dtype=bool)
    if reference is not None and reference > 1e-8:
        relative[eligible] = current_ratio[eligible] / reference
        weak = eligible & (relative < settings.current_ratio)
    qualified_cells = 0
    weak_cells = 0
    for row, cell in zip(rows, cells):
        if row["ratio"] is None or reference is None or reference <= 1e-8:
            row.update(relative=None, weak_fraction=None, assessment="insufficient intersection")
            continue
        qualified_cells += 1
        relative_value = float(row["ratio"] / reference)
        count_weak = int((weak[cell.view] & cell.interior_mask(cell.view)).sum())
        is_weak = relative_value < settings.current_ratio
        if is_weak:
            weak_cells += 1
        row.update(
            relative=round(relative_value, 5),
            weak_fraction=round(count_weak / row["stable_pixels"], 6),
            assessment="low relative response" if is_weak else "in range",
        )
    if qualified_cells == 0:
        messages.append("No cell reaches the usable overlap threshold.")
    return CurrentResponse(current_ratio, relative, eligible, common, weak, rows, {
        "low_capture": low_capture.id, "high_capture": high_capture.id,
        "high_current_a": high_capture.current_a, "low_current_a": low_capture.current_a,
        "exposure_scale": round(exposure_scale, 5),
        "reference_ratio": round(reference, 5) if reference is not None else None,
        "usable_cells": qualified_cells,
        "weak_cells": weak_cells,
        "shared_pixels": int(common.sum()),
        "stable_pixels": int(stable.sum()),
        "weak_pixels": int(weak.sum()),
        "relative_histogram": histogram_rows(relative[eligible], (0, 1.6), 20),
        "messages": messages,
    })


def probe_response(analysis: CurrentResponse, x: int, y: int) -> dict:
    relative = analysis.relative[y, x]
    ratio = analysis.ratio[y, x]
    return {
        "reported": bool(analysis.eligible[y, x]),
        "common": bool(analysis.common[y, x]),
        "ratio": round(float(ratio), 6) if np.isfinite(ratio) else None,
        "relative": round(float(relative), 6) if np.isfinite(relative) else None,
        "weak": bool(analysis.weak[y, x]),
    }
