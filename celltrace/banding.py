from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage, signal

from .images import distribution, png_bytes
from .model import Capture, Issue
from .render import draw_tag, font, source_preview


@dataclass(frozen=True)
class ReferenceBanding:
    averages: np.ndarray
    detailed_axis: str
    profile: np.ndarray
    trend: np.ndarray
    residual: np.ndarray
    runs: list
    spectral_peaks: list
    invalid_fraction: np.ndarray
    ready_rows: np.ndarray


def capture_reference_profile(capture: Capture, pixels: np.ndarray, axis: str) -> ReferenceBanding:
    if capture.role not in ("dark", "flat"):
        raise Issue("Sensor banding is evaluated on a selected dark or flat reference.")
    oriented = pixels if axis == "rows" else pixels.T
    ready = oriented < capture.white_level
    # Use a regular cross-section across the long dimension to bound the statistics.
    step = max(1, (oriented.shape[1] + 699) // 700)
    small = oriented[:, ::step].astype(np.float32) / capture.white_level
    use = ready[:, ::step]
    totals = np.full(oriented.shape[0], np.nan, dtype=np.float32)
    clipped = np.mean(~use, axis=1)
    keep = clipped < 0.7
    for index in np.flatnonzero(keep):
        totals[index] = float(np.median(small[index, use[index]]))
    count = int(keep.sum())
    if count < max(12, 0.6 * len(keep)):
        return ReferenceBanding(totals, axis, totals, totals, np.zeros(totals.shape), [], [], clipped, keep)
    xi = np.arange(totals.size)
    interpolated = np.interp(xi, xi[keep], totals[keep])
    sigma = max(3.0, totals.size / 42.0)
    smooth = ndimage.gaussian_filter1d(interpolated, sigma, mode="reflect")
    residual = interpolated - smooth
    residual[~keep] = np.nan
    average = float(np.median(totals[keep]))
    threshold = max(0.001, average * 0.02)
    runs = outlying_runs(residual, keep, threshold)
    peaks = periodic_noise(residual, keep, threshold)
    return ReferenceBanding(totals, axis, interpolated, smooth, residual, runs, peaks, clipped, keep)


def outlying_runs(residual: np.ndarray, accepted: np.ndarray, threshold: float) -> list:
    anomalies = []
    center = np.median(residual[accepted])
    distance = abs(residual - center)
    median = np.median(distance[accepted])
    high = max(threshold, 4.4 * 1.4826 * median)
    markers = (distance >= high) & accepted
    labels, count = ndimage.label(markers)
    for group, region in enumerate(ndimage.find_objects(labels, count), 1):
        if region is None:
            continue
        span = region[0]
        selection = labels[span] == group
        values = residual[span][selection]
        peak = int(np.argmax(abs(values)))
        anomalies.append({
            "from": span.start, "through": span.stop - 1, "lines": int(selection.sum()),
            "peak": round(float(values[peak]), 6),
            "mean": round(float(np.mean(values)), 6),
        })
    return sorted(anomalies, key=lambda item: (-abs(item["peak"]), item["from"]))[:24]


def periodic_noise(residual: np.ndarray, accepted: np.ndarray, scale: float) -> list:
    if len(residual) < 44:
        return []
    clean = np.where(accepted, residual - np.median(residual[accepted]), 0)
    segment = min(512, len(clean))
    frequency, energy = signal.welch(clean, nperseg=segment, scaling="spectrum", detrend="constant")
    peaks, _ = signal.find_peaks(energy)
    selected = []
    floor = max(float(np.median(energy[1:])) * 11, scale**2 * 0.006)
    for index in sorted(peaks, key=lambda index: -energy[index]):
        if index <= 0 or energy[index] < floor:
            continue
        period = 1 / frequency[index]
        if not 3 <= period <= min(150, len(clean) / 4):
            continue
        selected.append({
            "period_pixels": round(float(period), 3),
            "frequency": round(float(frequency[index]), 6),
            "rms": round(float(np.sqrt(energy[index])), 6),
        })
        if len(selected) == 8:
            break
    return selected


def profile_summary(computed: ReferenceBanding) -> dict:
    rows = [
        {
            "index": index,
            "level": round(float(computed.averages[index]), 6) if present else None,
            "trend": round(float(computed.trend[index]), 6) if present else None,
            "residual": round(float(computed.residual[index]), 6) if present else None,
            "clipped_fraction": round(float(computed.invalid_fraction[index]), 5),
        }
        for index, present in enumerate(computed.ready_rows)
    ]
    return {
        "axis": computed.detailed_axis,
        "rows": rows,
        "summary": {
            "usable_lines": int(computed.ready_rows.sum()),
            "all_lines": int(computed.ready_rows.size),
            "line_levels": distribution(computed.averages[computed.ready_rows]),
            "residual_levels": distribution(computed.residual[computed.ready_rows]),
            "outlying_runs": computed.runs,
            "periodic_peaks": computed.spectral_peaks,
        },
    }


def reference_band_image(engine, capture, computed: ReferenceBanding):
    image = source_preview(engine, capture, long_edge=1000, highlight=True)
    painter = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    is_row = computed.detailed_axis == "rows"
    scale = (height if is_row else width) / len(computed.averages)
    face = font(11)
    for row in computed.runs[:10]:
        low = row["from"] * scale
        high = (row["through"] + 1) * scale
        color = (255, 112, 156) if row["peak"] > 0 else (115, 206, 255)
        rectangle = (0, low, width, high) if is_row else (low, 0, high, height)
        painter.rectangle(rectangle, fill=(*color, 64), outline=(*color, 199), width=1)
        if is_row:
            draw_tag(painter, (8, min(height - 18, low)), str(row["from"]), color, face)
        else:
            draw_tag(painter, (min(width - 35, low + 3), 4), str(row["from"]), color, face)
    return png_bytes(image)
