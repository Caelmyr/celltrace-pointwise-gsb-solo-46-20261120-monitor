from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell, millimeter_scale
from .intensity import IntensityMap
from .model import EdgeSettings, Plane
from .quality import supported_blur


CONNECT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class EdgeMap:
    setback: np.ndarray
    missing: np.ndarray
    deactivation: np.ndarray
    records: list[dict]
    report: dict
    axis: str
    boundary: int
    faces: tuple[str, ...]


def edge_boundary(plane: Plane, settings: EdgeSettings) -> int:
    if settings.axis == "vertical":
        limit = plane.width
    else:
        limit = plane.height
    boundary = int(round(settings.boundary * limit))
    return min(max(1, boundary), limit - 1)


def edge_participants(cells: list[Cell], plane: Plane, settings: EdgeSettings):
    """Yield (cell, face) pairs whose physical cell edge lies on the cut boundary."""
    boundary = edge_boundary(plane, settings)
    for cell in cells:
        x0, y0, x1, y1 = cell.bounds
        if settings.axis == "vertical":
            if settings.right_face and x1 == boundary:
                yield cell, "right"
            if settings.left_face and x0 == boundary:
                yield cell, "left"
        else:
            if settings.right_face and y1 == boundary:
                yield cell, "bottom"
            if settings.left_face and y0 == boundary:
                yield cell, "top"


def face_orientation(face: str) -> str:
    return "vertical" if face in ("left", "right") else "horizontal"


def oriented_view(panel: PanelImage, cell: Cell, face: str):
    """Return full-bounds cell arrays with the inspected physical edge on the local left."""
    x0, y0, x1, y1 = cell.bounds
    signal = panel.signal[y0:y1, x0:x1]
    valid = panel.valid[y0:y1, x0:x1]
    if face in ("left", "right"):
        if face == "left":
            return signal, valid, x0, y0
        return signal[:, ::-1], valid[:, ::-1], x1 - 1, y0
    transposed = signal.T
    supported = valid.T
    if face == "top":
        return transposed, supported, y0, x0
    return transposed[::-1, :], supported[::-1, :], y1 - 1, x0


def edge_anchor(inventory: IntensityMap, cell_id: str):
    return inventory.dark_reference.get(cell_id)


def active_face(signal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mask of the genuinely emissive cell face, excluding structural dark margins."""
    if not valid.any():
        return np.zeros(signal.shape, dtype=bool)
    level = np.nanmedian(signal[valid])
    floor = max(0.02, level * 0.35)
    bright = valid & (signal >= floor)
    filled = ndimage.binary_closing(bright, structure=np.ones((5, 5), dtype=bool))
    labels, count = ndimage.label(filled, structure=CONNECT)
    if count == 0:
        return bright
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    face_label = int(np.argmax(sizes))
    face = labels == face_label
    return ndimage.binary_fill_holes(face)


def active_origin_from_face(face: np.ndarray, max_bezel: int = 12) -> int:
    """First column where the active face occupies most rows."""
    height = face.shape[0]
    for column in range(min(max_bezel, face.shape[1])):
        if face[:, column].sum() >= height * 0.6:
            return column
    return 0


def active_dark_labels(signal: np.ndarray, valid: np.ndarray, anchor: float,
                       dark_fraction: float, edge_column: int, face: np.ndarray):
    """Threshold and label dark regions inside the active face, excluding the bezel."""
    filtered, weights = supported_blur(signal, valid, 0.75)
    testable = valid & (weights >= 0.73) & face
    dark = testable & (filtered < anchor * dark_fraction)
    dark[:, :edge_column] = False
    labels, count = ndimage.label(dark, structure=CONNECT)
    return labels, count


def edge_components(labels: np.ndarray, count: int, edge_column: int, footprint: int,
                    max_depth: int) -> dict[int, dict]:
    """Keep only dark components that touch the active cut boundary; reject detached spots."""
    if count == 0:
        return {}
    edge_markers = np.unique(labels[:, edge_column])
    edge_markers = edge_markers[edge_markers != 0]
    kept = {}
    for marker in edge_markers:
        ys, xs = np.where(labels == marker)
        edge_run = int(np.sum(labels[:, edge_column] == marker))
        depth = int(xs.max() + 1) - edge_column
        if edge_run < footprint or depth > max_depth or depth <= 0:
            continue
        kept[int(marker)] = {
            "rows": ys, "cols": xs, "edge_run": edge_run, "depth": depth,
        }
    return kept


def build_setback(labels: np.ndarray, kept: dict[int, dict], length: int, depth: int,
                  edge_column: int) -> np.ndarray:
    """Profile of recession per row plus a raster marking the inner active boundary."""
    profile = np.zeros(length, dtype=np.uint16)
    raster = np.zeros((length, depth), dtype=np.uint16)
    if kept:
        band = np.zeros((length, depth), dtype=bool)
        for component in kept.values():
            band[component["rows"], component["cols"]] = True
        for row in range(length):
            if not band[row, edge_column]:
                continue
            dark_cols = np.flatnonzero(band[row])
            run_end = int(dark_cols.max()) + 1
            recession = max(0, run_end - edge_column)
            profile[row] = recession
            boundary = min(run_end, depth - 1)
            raster[row, boundary] = recession
    raster[:, 0] = profile
    return raster


def merge_runs(flags: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, flagged in enumerate(flags):
        if flagged and start is None:
            start = index
        elif not flagged and start is not None:
            runs.append([start, index - 1])
            start = None
    if start is not None:
        runs.append([start, len(flags) - 1])
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(start, end) for start, end in merged]


def interval_rows(runs: list[tuple[int, int]], origin: float, scale: float,
                  spacing: float) -> list[dict]:
    rows = []
    for start, end in runs:
        length_px = end - start + 1
        rows.append({
            "start_px": round((start + origin) * spacing, 3),
            "end_px": round((end + origin + 1) * spacing, 3),
            "start_mm": round((start + origin) * spacing * scale, 3) if scale else None,
            "end_mm": round((end + origin + 1) * spacing * scale, 3) if scale else None,
            "length_px": length_px,
            "length_mm": round(length_px * spacing * scale, 3) if scale else None,
        })
    return rows


def deactivation_mask(labels: np.ndarray, kept: dict[int, dict],
                      shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for component in kept.values():
        mask[component["rows"], component["cols"]] = True
    return mask


def analyze_edge(panel: PanelImage, cell: Cell, face: str, settings: EdgeSettings,
                 inventory: IntensityMap, dark_fraction: float,
                 scales: tuple[float | None, float | None]) -> tuple[dict, dict]:
    """Measure one physical cut edge. Returns the record and local rasters."""
    vertical = face_orientation(face) == "vertical"
    signal, valid, edge_u, edge_v = oriented_view(panel, cell, face)
    length, depth = signal.shape
    anchor = edge_anchor(inventory, cell.id)
    scale_along = scales[1] if vertical else scales[0]
    scale_across = scales[0] if vertical else scales[1]
    base = {
        "cell": cell.id, "string": cell.string, "axis": "vertical" if vertical else "horizontal",
        "face": face, "edge_length_px": length,
        "edge_length_mm": round(length * scale_along, 3) if scale_along else None,
        "edge_origin_px": [edge_u, edge_v],
    }
    local = {"setback": np.zeros(signal.shape, dtype=np.uint16), "band": np.zeros(signal.shape, bool),
             "missing": np.zeros(signal.shape, bool), "vertical": vertical,
             "reliable": anchor is not None}
    if anchor is None:
        local["missing"][:] = True
        record = {
            **base, "status": "unreliable cell", "bezel_px": None,
            "mean_setback_px": None, "max_setback_px": None,
            "mean_setback_mm": None, "max_setback_mm": None,
            "deactivated_pixels": 0, "deactivated_area_mm2": None,
            "measured_positions": 0, "missing_positions": length,
            "missing_fraction": 1.0, "missing_segments": [], "edge_components": 0,
            "setback_px": [None] * length, "setback_mm": None,
        }
        return record, local
    face = active_face(signal, valid)
    edge_column = min(active_origin_from_face(face) + settings.seam_gap_px, depth - 1)
    labels, count = active_dark_labels(signal, valid, anchor, dark_fraction, edge_column, face)
    kept = edge_components(
        labels, count, edge_column, settings.edge_footprint, settings.max_setback_px
    )
    setback_raster = build_setback(labels, kept, length, depth, edge_column)
    setback = setback_raster[:, 0]
    band = deactivation_mask(labels, kept, signal.shape)
    # A boundary row is unreadable when any supported column from the cut edge through
    # the active origin is missing (film edge covered or frame cropped).
    support_span = valid[:, :edge_column + 1]
    edge_support = support_span.all(axis=1)
    missing_runs = merge_runs(~edge_support, settings.merge_gap_px)
    missing_segments = interval_rows(missing_runs, edge_v, scale_along, 1.0)
    missing_positions = int((~edge_support).sum())
    # Setback is only meaningful where the active face reaches the edge column.
    setback[~face[:, edge_column]] = 0
    setback_raster[:, 0] = setback
    values = setback[setback > 0]
    mean_setback = float(values.mean()) if values.size else 0.0
    max_setback = float(values.max()) if values.size else 0.0
    deactivated = int(band.sum())
    local["setback"] = setback_raster
    local["band"] = band
    local["missing"][:, 0] = ~edge_support
    record = {
        **base, "status": "measured",
        "bezel_px": edge_column,
        "setback_px": setback.astype(int).tolist(),
        "mean_setback_px": round(mean_setback, 3),
        "max_setback_px": round(max_setback, 3),
        "mean_setback_mm": round(mean_setback * scale_across, 3) if scale_across else None,
        "max_setback_mm": round(max_setback * scale_across, 3) if scale_across else None,
        "setback_mm": (setback.astype(float) * scale_across).round(3).tolist()
        if scale_across else None,
        "deactivated_pixels": deactivated,
        "deactivated_area_mm2": round(
            deactivated * scale_along * scale_across, 3
        ) if scale_along and scale_across else None,
        "measured_positions": length - missing_positions,
        "missing_positions": missing_positions,
        "missing_fraction": round(missing_positions / length, 5) if length else 1.0,
        "missing_segments": missing_segments,
        "edge_components": len(kept),
    }
    return record, local


def paint_local(raster: np.ndarray, cell: Cell, local: np.ndarray, vertical: bool, flip: bool):
    """Place an oriented local (edge-on-left) raster back onto the plane cell bounds."""
    if not vertical:
        local = local.T
        if flip:
            local = local[::-1, :]
    elif flip:
        local = local[:, ::-1]
    x0, y0, x1, y1 = cell.bounds
    target = (slice(y0, y1), slice(x0, x1))
    if local.dtype == bool:
        raster[target] |= local
    else:
        existing = raster[target]
        raster[target] = np.where(local > 0, local, existing)


def cut_edge_map(panel: PanelImage, plane: Plane, cells: list[Cell],
                 settings: EdgeSettings, inventory: IntensityMap,
                 dark_fraction: float) -> EdgeMap:
    boundary = edge_boundary(plane, settings)
    scales = millimeter_scale(plane)
    setback_raster = np.full(panel.signal.shape, np.nan, dtype=np.float32)
    missing_raster = np.zeros(panel.signal.shape, dtype=bool)
    deactivation_raster = np.zeros(panel.signal.shape, dtype=bool)
    records = []
    participants = list(edge_participants(cells, plane, settings))
    for cell, face in participants:
        record, local = analyze_edge(
            panel, cell, face, settings, inventory, dark_fraction, scales
        )
        records.append(record)
        vertical = local["vertical"]
        flip = face in ("right", "bottom")
        paint_local(setback_raster, cell, local["setback"], vertical, flip)
        paint_local(deactivation_raster, cell, local["band"], vertical, flip)
        paint_local(missing_raster, cell, local["missing"], vertical, flip)
    records.sort(key=lambda row: (row["cell"], row["face"]))
    total_missing = sum(row["missing_positions"] for row in records)
    total_length = sum(row["edge_length_px"] for row in records)
    total_deactivated = sum(row["deactivated_pixels"] for row in records)
    report = {
        "enabled": settings.enabled,
        "axis": settings.axis,
        "boundary_fraction": settings.boundary,
        "boundary_px": boundary,
        "faces": list(settings.face_names()),
        "cell_edges": len(records),
        "reliable_edges": sum(row["status"] == "measured" for row in records),
        "edge_length_px": total_length,
        "deactivated_pixels": total_deactivated,
        "missing_positions": total_missing,
        "missing_fraction": round(total_missing / total_length, 5) if total_length else None,
        "seam_gap_px": settings.seam_gap_px,
        "edge_footprint": settings.edge_footprint,
        "max_setback_px": settings.max_setback_px,
        "measurement": (
            "Edge-connected dark setback along the configured cut boundary; "
            "detached dark spots are excluded"
        ),
    }
    return EdgeMap(
        setback_raster, missing_raster, deactivation_raster, records, report,
        settings.axis, boundary, settings.face_names(),
    )
