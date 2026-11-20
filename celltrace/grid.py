"""Piecewise seam calibration on top of the regular cell grid.

The declared panel plane starts with an equal-width grid. Real module seams can
wander: a seam may be straight near one panel edge and drift toward the other
half. Each seam is therefore described by a few calibration points sampled
against the original capture. Points are interpolated independently for every
plane row (or column), producing one shared boundary between the two adjacent
cell pieces.

Pixel ownership is resolved once on the full pixel-center grid with half-open
intervals (``left <= x < right``). Every plane pixel therefore belongs to
exactly one cell: the two neighbours of a calibrated seam read the same
boundary, and crossing seams form proper T-junctions without duplicated or
unattended strips.

Cell identifiers (RxxCxx) keep their regular row/column meaning; only the pixel
edges attached to an identifier move.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import Cell
from .model import Issue, Plane, integer, mapping, real, sequence

MAX_SEAM_POINTS = 24
MAX_SEAM_SHIFT_FRACTION = 0.6
MIN_SEAM_GAP = 1.0


@dataclass(frozen=True)
class SeamCalibration:
    """One calibrated seam.

    axis: "vertical" seams separate columns and run along the plane height;
          their samples give the seam x position at several y positions.
          "horizontal" seams separate rows and give y at several x positions.
    index: 1-based grid line index (vertical seam ``k`` separates columns
          ``k - 1`` and ``k``).
    points: (along, across) plane-pixel samples. For vertical seams along is y
          and across is x; for horizontal seams along is x and across is y.
    """

    axis: str
    index: int
    points: tuple[tuple[float, float], ...]

    @classmethod
    def from_dict(cls, raw: dict, plane: Plane):
        raw = mapping(raw, "Seam calibration")
        axis = raw.get("axis")
        if axis not in ("vertical", "horizontal"):
            raise Issue("A seam axis must be vertical or horizontal.")
        line_count = plane.cols if axis == "vertical" else plane.rows
        index = integer(raw.get("index"), "Seam index", 1, line_count)
        entries = sequence(raw.get("points"), "Seam calibration points", 2, MAX_SEAM_POINTS)
        along_limit = plane.height if axis == "vertical" else plane.width
        points = []
        for entry in entries:
            pair = sequence(entry, "Seam calibration point", 2, 2)
            along = real(pair[0], "Seam sample position", 0.0, float(along_limit))
            across = real(pair[1], "Seam boundary position", 0.0,
                          float(plane.width if axis == "vertical" else plane.height))
            points.append((along, across))
        points.sort(key=lambda item: item[0])
        along_values = [item[0] for item in points]
        if any(second - first < 1.0 for first, second in zip(along_values, along_values[1:])):
            raise Issue("Seam calibration points must be at least one plane pixel apart.")
        seam = cls(axis, index, tuple(points))
        validate_seam(seam, plane)
        return seam

    def to_dict(self) -> dict:
        return {"axis": self.axis, "index": self.index,
                "points": [[round(a, 4), round(b, 4)] for a, b in self.points]}


def parse_grid_adjustments(raw, plane: Plane) -> dict:
    """Parse the persisted adjustment document into plain serializable dicts.

    The returned structure stores the same JSON-ready dictionaries that
    :meth:`SeamCalibration.to_dict` produces, so ``asdict(inspection)`` and the
    revision digests keep working. Use :func:`calibration_objects` when code
    needs the interpolating :class:`SeamCalibration` instances.
    """
    if raw in (None, {}):
        return {"vertical": [], "horizontal": []}
    raw = mapping(raw, "Grid adjustments")
    result = {}
    for axis in ("vertical", "horizontal"):
        seams = [SeamCalibration.from_dict(entry, plane).to_dict()
                 for entry in sequence(raw.get(axis, []), f"{axis} seams", 0, 24)]
        indices = [seam["index"] for seam in seams]
        if len(set(indices)) != len(indices):
            raise Issue(f"Each {axis} seam can be calibrated only once.")
        result[axis] = sorted(seams, key=lambda seam: seam["index"])
    validate_adjustments(calibration_objects(result), plane)
    return result


def calibration_objects(adjustments: dict | None) -> dict:
    """Project the persisted dict document into interpolating seam objects."""
    if not adjustments:
        return {"vertical": [], "horizontal": []}
    return {
        axis: [item if isinstance(item, SeamCalibration) else _direct_seam(item)
               for item in adjustments.get(axis, [])]
        for axis in ("vertical", "horizontal")
    }


def _direct_seam(raw: dict) -> "SeamCalibration":
    return SeamCalibration(
        raw["axis"], raw["index"],
        tuple((float(a), float(b)) for a, b in raw["points"]),
    )


def nominal_edges(length: int, divisions: int) -> np.ndarray:
    return np.floor(np.linspace(0, length, divisions + 1) + 0.5).astype(np.float64)


def round_edge(values) -> np.ndarray:
    """Half-up rounding so boundary ownership is deterministic at .5 pixels."""
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def seam_positions(seam: SeamCalibration, samples: np.ndarray) -> np.ndarray:
    """Boundary position at every requested along-axis sample."""
    along = np.array([point[0] for point in seam.points], dtype=np.float64)
    across = np.array([point[1] for point in seam.points], dtype=np.float64)
    return np.interp(samples, along, across)


def boundary_grids(plane: Plane, adjustments: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Boundary position of every grid line at every pixel center.

    Returns ``(x_grid, y_grid)`` with shapes
    ``(plane.height, plane.cols + 1)`` and ``(plane.width, plane.rows + 1)``.
    ``x_grid[y, k]`` is the x coordinate of vertical boundary ``k`` evaluated at
    pixel row ``y``; ``y_grid[x, k]`` is the y coordinate of horizontal
    boundary ``k`` at pixel column ``x``. The outer boundaries stay fixed at the
    panel edges so cells cannot leave the plane.
    """
    row_centers = np.arange(plane.height, dtype=np.float64) + 0.5
    col_centers = np.arange(plane.width, dtype=np.float64) + 0.5
    x_grid = np.repeat(nominal_edges(plane.width, plane.cols)[None, :], plane.height, axis=0)
    y_grid = np.repeat(nominal_edges(plane.height, plane.rows)[None, :], plane.width, axis=0)
    if adjustments:
        for seam in calibration_objects(adjustments).get("vertical", []):
            x_grid[:, seam.index] = seam_positions(seam, row_centers)
        for seam in calibration_objects(adjustments).get("horizontal", []):
            y_grid[:, seam.index] = seam_positions(seam, col_centers)
    x_grid[:, 0] = 0.0
    x_grid[:, -1] = float(plane.width)
    y_grid[:, 0] = 0.0
    y_grid[:, -1] = float(plane.height)
    return x_grid, y_grid


def edge_arrays(plane: Plane, adjustments: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Boundary positions sampled at regular row/column edge coordinates.

    Kept for reporting and rendering helpers that reason about row/column
    boundaries rather than individual pixels.
    """
    row_coords = nominal_edges(plane.height, plane.rows)
    col_coords = nominal_edges(plane.width, plane.cols)
    x_base = nominal_edges(plane.width, plane.cols)
    y_base = nominal_edges(plane.height, plane.rows)
    x_edges = np.repeat(x_base[None, :], plane.rows + 1, axis=0)
    y_edges = np.repeat(y_base[None, :], plane.cols + 1, axis=0)
    if adjustments:
        for seam in calibration_objects(adjustments).get("vertical", []):
            x_edges[:, seam.index] = seam_positions(seam, row_coords)
        for seam in calibration_objects(adjustments).get("horizontal", []):
            y_edges[:, seam.index] = seam_positions(seam, col_coords)
    return x_edges, y_edges


def validate_seam(seam: SeamCalibration, plane: Plane):
    """A seam must stay ordered against its neighbours and within its travel."""
    x_grid, y_grid = boundary_grids(plane, _with_single_seam(plane, seam))
    if seam.axis == "vertical":
        gaps = np.diff(round_edge(x_grid), axis=1)
        nominal_shift = plane.width / plane.cols * MAX_SEAM_SHIFT_FRACTION
        nominal = nominal_edges(plane.width, plane.cols)[seam.index]
        shifts = np.abs(x_grid[:, seam.index] - nominal)
    else:
        gaps = np.diff(round_edge(y_grid), axis=1)
        nominal_shift = plane.height / plane.rows * MAX_SEAM_SHIFT_FRACTION
        nominal = nominal_edges(plane.height, plane.rows)[seam.index]
        shifts = np.abs(y_grid[:, seam.index] - nominal)
    if np.min(gaps) < MIN_SEAM_GAP:
        raise Issue("The calibrated seam would make a cell boundary overlap or cross its neighbour.")
    if np.max(shifts) > nominal_shift + 1e-6:
        raise Issue("A seam can move at most 60% of the nominal cell width from its regular position.")


def _with_single_seam(plane: Plane, seam: SeamCalibration) -> dict:
    adjustments = {"vertical": [], "horizontal": []}
    adjustments[seam.axis].append(seam)
    return adjustments


def validate_adjustments(adjustments: dict, plane: Plane):
    """Cross-seam ordering check once every seam has been parsed."""
    x_grid, y_grid = boundary_grids(plane, calibration_objects(adjustments))
    if np.min(np.diff(round_edge(x_grid), axis=1)) < MIN_SEAM_GAP:
        raise Issue("Calibrated vertical seams must stay ordered without overlapping cell pieces.")
    if np.min(np.diff(round_edge(y_grid), axis=1)) < MIN_SEAM_GAP:
        raise Issue("Calibrated horizontal seams must stay ordered without overlapping cell pieces.")


def pixel_ownership(plane: Plane, adjustments: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Integer row/column index of the cell owning each plane pixel.

    Pixel centers are compared with the half-open boundary intervals, so each
    pixel is assigned exactly once even where a horizontal and vertical seam
    meet at a T-junction.
    """
    x_grid, y_grid = boundary_grids(plane, adjustments)
    row_centers = np.arange(plane.height, dtype=np.float64) + 0.5
    col_centers = np.arange(plane.width, dtype=np.float64) + 0.5
    # Each row/column holds an ordered boundary list; searchsorted locates the
    # half-open interval containing the pixel center.
    col_index = np.empty((plane.height, plane.width), dtype=np.int64)
    for y in range(plane.height):
        col_index[y] = np.searchsorted(x_grid[y], col_centers, side="right") - 1
    row_boundaries = y_grid.T  # (rows+1, width): horizontal boundary per column
    row_index = np.empty((plane.height, plane.width), dtype=np.int64)
    for x in range(plane.width):
        row_index[:, x] = np.searchsorted(row_boundaries[:, x], row_centers, side="right") - 1
    return np.clip(row_index, 0, plane.rows - 1), np.clip(col_index, 0, plane.cols - 1)


def cells_for_plane(plane: Plane, adjustments: dict | None = None) -> list[Cell]:
    """Build cells using calibrated boundaries.

    ``bounds`` is the integer bounding box of the pixels the cell owns; the
    returned ``mask`` marks the exact owned pixels inside that box (the whole
    box for a regular grid). The pixel-level partition from
    :func:`pixel_ownership` guarantees gap-free, duplication-free coverage even
    where vertical and horizontal seams meet at a T-junction.
    """
    row_owner, col_owner = pixel_ownership(plane, adjustments)
    objects = calibration_objects(adjustments)
    has_adjustment = bool(objects["vertical"] or objects["horizontal"])
    cells = []
    for row in range(plane.rows):
        for col in range(plane.cols):
            membership = (row_owner == row) & (col_owner == col)
            ys, xs = np.nonzero(membership)
            if xs.size == 0:
                raise Issue("A calibrated seam leaves a cell without any plane pixels.")
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            ix = int(round((x1 - x0) * plane.inset))
            iy = int(round((y1 - y0) * plane.inset))
            axis_index = col if plane.string_axis == "columns" else row
            local_mask = membership[y0:y1, x0:x1] if has_adjustment else None
            cells.append(Cell(
                row, col, (x0, y0, x1, y1),
                (x0 + ix, y0 + iy, x1 - ix, y1 - iy),
                axis_index // plane.string_size + 1,
                local_mask,
            ))
    return cells


def cell_at(plane: Plane, adjustments: dict | None, x: float, y: float) -> Cell | None:
    if not (0.0 <= x < plane.width and 0.0 <= y < plane.height):
        return None
    row_owner, col_owner = pixel_ownership(plane, adjustments)
    py, px = int(np.floor(y)), int(np.floor(x))
    row, col = int(row_owner[py, px]), int(col_owner[py, px])
    # Recover the owned bounding box directly from the ownership map.
    membership = (row_owner == row) & (col_owner == col)
    if not membership.any():
        return None
    ys, xs = np.nonzero(membership)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ix = int(round((x1 - x0) * plane.inset))
    iy = int(round((y1 - y0) * plane.inset))
    axis_index = col if plane.string_axis == "columns" else row
    objects = calibration_objects(adjustments)
    local_mask = membership[y0:y1, x0:x1] if (objects["vertical"] or objects["horizontal"]) else None
    return Cell(
        row, col, (x0, y0, x1, y1),
        (x0 + ix, y0 + iy, x1 - ix, y1 - iy),
        axis_index // plane.string_size + 1,
        local_mask,
    )


def seam_polylines(plane: Plane, adjustments: dict | None) -> list[dict]:
    """Render-ready polylines for every calibrated seam.

    Lines are sampled per plane pixel so a drifting seam draws as a smooth
    polyline rather than a coarse join between row boundaries.
    """
    if not adjustments:
        return []
    objects = calibration_objects(adjustments)
    lines = []
    for seam in objects.get("vertical", []):
        rows = np.arange(plane.height, dtype=np.float64)
        lines.append({
            "axis": "vertical", "index": seam.index,
            "points": [[round(float(pos), 3), int(row)]
                       for row, pos in zip(rows, seam_positions(seam, rows))],
            "samples": seam.to_dict()["points"],
        })
    for seam in objects.get("horizontal", []):
        cols = np.arange(plane.width, dtype=np.float64)
        lines.append({
            "axis": "horizontal", "index": seam.index,
            "points": [[int(col), round(float(pos), 3)]
                       for col, pos in zip(cols, seam_positions(seam, cols))],
            "samples": seam.to_dict()["points"],
        })
    return lines


def adjusted_seam_keys(adjustments: dict | None) -> set[tuple[str, int]]:
    if not adjustments:
        return set()
    objects = calibration_objects(adjustments)
    return {(seam.axis, seam.index) for seam in objects.get("vertical", [])} | \
        {(seam.axis, seam.index) for seam in objects.get("horizontal", [])}
