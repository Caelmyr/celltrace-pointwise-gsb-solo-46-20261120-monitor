from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .engine import Engine
from .images import (
    compact_thumbnail, continuous_palette, distribution, grayscale_preview, histogram_rows, png_bytes,
    set_invalid_pattern,
)
from .model import Capture, Inspection, Issue, integer
from .render import DIVERGING, draw_tag, font


@dataclass(frozen=True)
class ComparisonRegion:
    row: int
    column: int
    bounds: tuple[int, int, int, int]

    @property
    def view(self):
        return slice(self.bounds[1], self.bounds[3]), slice(self.bounds[0], self.bounds[2])

    @property
    def label(self):
        return f"C{self.column + 1:02}R{self.row + 1:02}"


@dataclass(frozen=True)
class ReferenceComparison:
    old: np.ndarray
    new: np.ndarray
    old_usable: np.ndarray
    new_usable: np.ndarray
    delta: np.ndarray
    support: np.ndarray
    change: np.ndarray
    summary: dict
    tiles: list[dict]
    events: list[dict]


def compare_references(engine: Engine, inspection: Inspection, prior: str, target: str, bins: int):
    first = inspection.capture(prior)
    second = inspection.capture(target)
    same_sensor(first, second)
    key = (
        "reference-review", first.digest, second.digest, first.white_level, second.white_level, first.role, bins,
    )
    values = engine.measurements.get(
        key,
        lambda: compute_comparison(
            first, second, engine.read_source(first).pixels, engine.read_source(second).pixels, bins
        ),
    )
    sources = {
        "before": capture_details(first), "after": capture_details(second),
        "revision": inspection.revision, "inspection": inspection.id,
    }
    return values, sources


def same_sensor(first: Capture, second: Capture):
    if first.id == second.id:
        raise Issue("Choose two separately captured references.")
    if first.role not in ("dark", "flat") or first.role != second.role:
        raise Issue("Use two dark references or two flat references.")
    if first.width != second.width or first.height != second.height:
        raise Issue("Reference comparison requires matching native sensor dimensions.")
    if first.bit_depth != second.bit_depth or first.white_level != second.white_level:
        raise Issue("Reference comparison requires the same native sensor code range.")
    if abs(first.exposure_ms - second.exposure_ms) > max(0.025, 0.005 * first.exposure_ms):
        raise Issue("Reference comparison requires matching exposure times.")


def capture_details(capture: Capture) -> dict:
    return {
        "id": capture.id, "name": capture.name, "exposure_ms": capture.exposure_ms,
        "digest": capture.digest, "white_level": capture.white_level,
        "size": [capture.width, capture.height], "role": capture.role,
    }


def compute_comparison(first, second, before, after, bins):
    old = before.astype(np.float32) / first.white_level
    new = after.astype(np.float32) / second.white_level
    old_valid = usable_reference(old, first.role)
    new_valid = usable_reference(new, second.role)
    shared = old_valid & new_valid
    if shared.sum() < 50:
        raise Issue("Fewer than 50 sensor pixels support a reference comparison.", 409)
    delta = np.full(old.shape, np.nan, dtype=np.float32)
    np.subtract(new, old, where=shared, out=delta)
    common_shift = float(np.median(delta[shared]))
    residual = np.abs(delta[shared] - common_shift)
    cutoff = max(0.007 if first.role == "dark" else 0.02, 5.5 * 1.4826 * float(np.median(residual)))
    lost = old_valid & ~new_valid
    gained = new_valid & ~old_valid
    changes = np.zeros(old.shape, dtype=np.uint8)
    changes[lost] = 1
    changes[gained] = 2
    changes[shared & (delta - common_shift >= cutoff)] = 3
    changes[shared & (common_shift - delta >= cutoff)] = 4
    height, width = old.shape
    regions = sensor_grid(width, height, bins)
    records = region_stats(regions, old, new, delta, shared, changes)
    overall = {
        "scope": "Native sensor pixels shared by two matched reference exposures",
        "size": [width, height], "plane_pixels": int(old.size),
        "supported_before": int(old_valid.sum()), "supported_after": int(new_valid.sum()),
        "common_pixels": int(shared.sum()), "lost_pixels": int(lost.sum()), "gained_pixels": int(gained.sum()),
        "median_drift": round(common_shift, 7),
        "visible_drift_cutoff": round(cutoff, 7),
        "old": distribution(old[old_valid]), "new": distribution(new[new_valid]),
        "delta": distribution(delta[shared]),
        "delta_histogram": histogram_rows(delta[shared], (-0.08, 0.08), 24),
        "counts": {label: int((changes == count).sum()) for label, count in CHANGE_CODES.items()},
    }
    events = label_changes(changes, old_valid, new_valid)
    return ReferenceComparison(old, new, old_valid, new_valid, delta, shared, changes, overall, records, events)


CHANGE_CODES = {
    "lost": 1, "gained": 2, "increase": 3, "decrease": 4,
}
CHANGE_COLOR = {
    "lost": (241, 117, 155), "gained": (112, 232, 190),
    "increase": (242, 192, 102), "decrease": (124, 180, 236),
}


def usable_reference(signal: np.ndarray, role: str):
    valid = signal < 1
    if role == "flat" and valid.any():
        sampled = signal[::max(1, signal.shape[0] // 400), ::max(1, signal.shape[1] // 400)]
        middle = np.median(sampled[sampled < 1])
        valid &= signal > max(0.01, float(middle) * 0.20)
    return valid


def sensor_grid(width: int, height: int, bins: int) -> list[ComparisonRegion]:
    xcuts = np.rint(np.linspace(0, width, bins + 1)).astype(int).tolist()
    ycuts = np.rint(np.linspace(0, height, bins + 1)).astype(int).tolist()
    return [
        ComparisonRegion(
            row, col, (xcuts[col], ycuts[row], xcuts[col + 1], ycuts[row + 1])
        ) for row in range(bins) for col in range(bins)
    ]


def region_stats(regions, original, current, diff, set_mask, changes):
    records = []
    for region in regions:
        view = region.view
        shared = set_mask[view]
        old = original[view][shared]
        new = current[view][shared]
        values = diff[view][shared]
        flags = changes[view]
        counts = {name: int((flags == code).sum()) for name, code in CHANGE_CODES.items()}
        records.append({
            "region": region.label,
            "bounds": region.bounds,
            "pixels": int(shared.size),
            "common": int(shared.sum()),
            "lost": counts["lost"],
            "gained": counts["gained"],
            "increased": counts["increase"],
            "decreased": counts["decrease"],
            "old_median": round(float(np.median(old)), 6) if old.size else None,
            "new_median": round(float(np.median(new)), 6) if new.size else None,
            "median_delta": round(float(np.median(values)), 6) if values.size else None,
            "peak_delta": round(float(np.max(np.abs(values))), 6) if values.size else None,
        })
    return records


def label_changes(changes: np.ndarray, old_valid: np.ndarray, new_valid: np.ndarray):
    events = []
    for name, code in CHANGE_CODES.items():
        labels, count = ndimage.label(changes == code, structure=np.ones((3, 3), dtype=bool))
        counts = np.bincount(labels.reshape(-1))
        usable = min(36, max(8, changes.size // 25000))
        selections = [
            (index, counts[index]) for index in np.flatnonzero(counts >= usable) if index != 0
        ]
        if not selections:
            continue
        biggest = sorted(selections, key=lambda item: -item[1])[:16]
        boxes = ndimage.find_objects(labels, max_label=count)
        for index, amount in biggest:
            ys, xs = boxes[index - 1]
            region = ys, xs
            marker = labels[region] == index
            center_x, center_y = coordinate_average(marker, xs.start, ys.start)
            old_count = int((marker & old_valid[region]).sum())
            new_count = int((marker & new_valid[region]).sum())
            events.append({
                "type": name, "pixels": int(amount), "previous_support": old_count, "current_support": new_count,
                "bounds": [xs.start, ys.start, xs.stop, ys.stop],
                "center": [round(center_x, 2), round(center_y, 2)],
            })
    return sorted(events, key=lambda event: (-event["pixels"], event["type"]))[:40]


def coordinate_average(mask, x0, y0):
    y, x = np.where(mask)
    return x0 + float(x.mean()) + 0.5, y0 + float(y.mean()) + 0.5


def compare_heat(image: ReferenceComparison):
    limit = max(0.02, min(0.15, 4 * image.summary["visible_drift_cutoff"]))
    values = np.nan_to_num(image.delta)
    raster = continuous_palette(values, (-limit, limit), reversed_palette())
    set_invalid_pattern(raster, image.support)
    for label in ("lost", "gained"):
        code = CHANGE_CODES[label]
        raster[image.change == code] = CHANGE_COLOR[label]
    return Image.fromarray(raster, mode="RGB")


def reversed_palette():
    return (DIVERGING[0], DIVERGING[1], (26, 39, 48), DIVERGING[3], DIVERGING[4])


def compare_canvas(image: ReferenceComparison, sources: dict) -> bytes:
    high = max(0.03, image.summary["old"]["p90"], image.summary["new"]["p90"])
    high *= 1.12
    base = [
        grayscale_preview(image.old, image.old_usable, 0, high, 0.75),
        grayscale_preview(image.new, image.new_usable, 0, high, 0.75),
        compare_heat(image),
    ]
    captions = ["Reference before", "Reference after", "Native drift and support changes"]
    panels = [compact_thumbnail(view, 390) for view in base]
    tile_h = max(panel.height for panel in panels)
    tube = Image.new("RGB", (1232, tile_h + 102), (23, 36, 43))
    painter = ImageDraw.Draw(tube, "RGBA")
    painter.text((18, 13), f"{sources['before']['role'].capitalize()} reference comparison",
                 fill=(222, 238, 230), font=font(23))
    for index, (panel, title) in enumerate(zip(panels, captions)):
        x = 18 + index * 404
        y = 48
        tube.paste(panel, (x, y))
        draw_tag(painter, (x + 8, y + 6), title, (220, 240, 222), font(11))
        if index == 2:
            scale_x = panel.width / image.old.shape[1]
            scale_y = panel.height / image.old.shape[0]
            annotate_events(painter, image.events, scale_x, scale_y, x, y)
    for index, (label, color) in enumerate(CHANGE_COLOR.items()):
        x = 22 + index * 190
        bottom = tube.height - 33
        painter.rounded_rectangle((x, bottom, x + 16, bottom + 16), radius=2, fill=(*color, 255))
        painter.text((x + 22, bottom + 1), label.capitalize(), fill=(221, 231, 227), font=font(13))
    return png_bytes(tube)


def annotate_events(painter, events, scale_x, scale_y, x, y):
    for event in events[:18]:
        x0, y0, x1, y1 = event["bounds"]
        inner = (x + x0 * scale_x, y + y0 * scale_y, x + x1 * scale_x, y + y1 * scale_y)
        color = CHANGE_COLOR[event["type"]]
        painter.rectangle(inner, outline=(*color, 230), width=1)


def reference_result(comparison, sources, bins):
    return {
        "sources": sources, "summary": comparison.summary, "tiles": comparison.tiles,
        "connected_changes": comparison.events, "tiling": bins,
    }
