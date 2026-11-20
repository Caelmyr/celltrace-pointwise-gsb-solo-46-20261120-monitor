from dataclasses import asdict, replace

import numpy as np

from .calibration import PanelImage, support_reason
from .geometry import millimeter_scale, plane_polygon, polygon_mask, project, transform_for, validate_polygon
from .grid import cell_at, cells_for_plane
from .images import distribution, histogram_rows
from .model import (
    FINDING_CATEGORIES, FINDING_STATES, MAX_FINDINGS, MAX_POLYGON_POINTS, SENSOR_ROLES,
    Calibration, Finding, Inspection, Issue, Placement, Plane, choice, file_id, parse_cell, text, utcnow, xy_list,
)
from .store import data_digest, new_id


def plane_anchor(inspection: Inspection, source_role: str) -> dict:
    active = inspection.selected(source_role)
    placement = inspection.placements.get(active.id)
    dark = inspection.captures.get(inspection.calibration.dark_id)
    flat = inspection.captures.get(inspection.calibration.flat_id)
    values = {
        "image": active.id, "digest": active.digest,
        "placement": asdict(placement) if placement is not None else None,
        "plane": asdict(inspection.plane),
        "calibration": asdict(inspection.calibration),
        "reference_digests": {"dark": dark.digest if dark else None, "flat": flat.digest if flat else None},
    }
    return {"signature": data_digest(values), **values}


def finding_status(inspection: Inspection, finding: Finding) -> str:
    try:
        anchor = plane_anchor(inspection, finding.source_role)
    except Issue:
        return "source unavailable"
    return "current" if anchor["signature"] == finding.anchor.get("signature") else "source changed"


def list_findings(inspection: Inspection) -> list[dict]:
    return [
        {**asdict(finding), "source_status": finding_status(inspection, finding)}
        for finding in inspection.findings
    ]


def validate_source_anchor(inspection: Inspection, finding: Finding):
    anchor = finding.anchor
    try:
        image = inspection.capture(anchor.get("image"))
    except Issue:
        raise Issue("A finding refers to an invalid original image.") from None
    if image.role != finding.source_role or anchor.get("digest") != image.digest:
        raise Issue("A finding refers to an invalid original image.")
    claims = {key: value for key, value in anchor.items() if key != "signature"}
    if data_digest(claims) != anchor.get("signature"):
        raise Issue("Finding source metadata does not match its signature.")
    validate_anchor_geometry(anchor, finding, image)
    refs = anchor.get("reference_digests")
    if not isinstance(refs, dict):
        raise Issue("Finding reference metadata is invalid.")
    calibration = Calibration.from_dict(anchor.get("calibration"))
    for role in ("dark", "flat"):
        reference_id = getattr(calibration, f"{role}_id")
        if reference_id is None and refs.get(role) is None:
            continue
        reference = inspection.captures.get(reference_id)
        if reference is None or reference.role != role or reference.digest != refs.get(role):
            raise Issue("A finding refers to a missing calibration original.")


def validate_anchor_geometry(anchor: dict, finding: Finding, image):
    plane = Plane.from_dict(anchor.get("plane"))
    quad = anchor.get("placement")
    if quad is not None:
        placement = Placement.from_dict(quad, image.width, image.height)
        transform_for(plane, placement)
    points = plane_polygon(finding.points, plane)
    validate_polygon(points)
    if finding.cell is not None:
        parse_cell(finding.cell, plane.rows, plane.cols)
    return plane


def region_source_link(inspection: Inspection, finding: Finding):
    validate_source_anchor(inspection, finding)
    image = inspection.capture(finding.anchor["image"])
    return {
        "image": image.id,
        "digest": image.digest,
        "acquisition": {"exposure_ms": image.exposure_ms, "current_a": image.current_a},
        "state": finding_status(inspection, finding),
        "anchor_sha256": finding.anchor["signature"],
    }


def make_finding(inspection: Inspection, raw: dict, finding_id: str | None = None) -> Finding:
    saved = None
    if finding_id is not None:
        saved = get_finding(inspection, finding_id)
    elif len(inspection.findings) >= MAX_FINDINGS:
        raise Issue("This inspection contains the maximum number of findings.", 409)
    title = text(raw.get("title"), "Finding title", 100)
    category = choice(raw.get("category"), "Finding category", FINDING_CATEGORIES)
    status = choice(raw.get("status", "open"), "Finding state", FINDING_STATES)
    source_role = choice(raw.get("source_role"), "Finding source", SENSOR_ROLES)
    points = xy_list(raw.get("points"), "Finding polygon", 3, MAX_POLYGON_POINTS, (0, 1), (0, 1))
    validate_polygon(plane_polygon(points, inspection.plane))
    cell = raw.get("cell") or None
    if cell is not None:
        parse_cell(cell, inspection.plane.rows, inspection.plane.cols)
    note = text(raw.get("note", ""), "Finding note", 1200, allow_blank=True)
    modified = utcnow()
    if saved is not None and (saved.points, saved.cell, saved.source_role) == (points, cell, source_role):
        return replace(saved, title=title, category=category, status=status, note=note, updated_at=modified)
    current = plane_anchor(inspection, source_role)
    return Finding(
        finding_id or new_id(),
        title,
        category,
        status,
        source_role,
        points,
        cell,
        note,
        current,
        saved.created_at if saved is not None else modified,
        modified,
    )


def get_finding(inspection: Inspection, finding_id: str) -> Finding:
    file_id(finding_id, "Finding ID")
    for finding in inspection.findings:
        if finding.id == finding_id:
            return finding
    raise Issue("This finding does not exist.", 404)


def measure_polygon(inspection: Inspection, panel: PanelImage, points: tuple) -> tuple[dict, np.ndarray]:
    shape = plane_polygon(points, inspection.plane)
    validate_polygon(shape)
    mask = polygon_mask(shape, inspection.plane.width, inspection.plane.height)
    usable = mask & panel.valid
    count = int(mask.sum())
    valid = int(usable.sum())
    stats = distribution(panel.signal[usable])
    scale = millimeter_scale(inspection.plane)
    overlap = []
    for cell in cells_for_plane(inspection.plane, inspection.grid_adjustments):
        view = cell.view
        owned = cell.interior_mask(view)
        count_in = int((mask[view] & owned).sum())
        if count_in:
            overlap.append({"cell": cell.id, "pixels": count_in,
                            "usable": int((usable[view] & owned).sum())})
    rows, cols = np.nonzero(mask)
    center = [round(float(cols.mean()) + 0.5, 2), round(float(rows.mean()) + 0.5, 2)] if count else None
    return {
        "pixels": count, "usable": valid, "usable_fraction": round(valid / count, 5) if count else 0.0,
        "area_mm2": round(count * scale[0] * scale[1], 3) if scale else None,
        "center": center, "cells": overlap, "signal": stats,
        "clipped": int((mask & panel.clipped).sum()),
        "calibration_gap": int((mask & panel.calibration_gap).sum()),
        "outside": int((mask & ~panel.any_support).sum()),
        "histogram": histogram_rows(panel.signal[usable], (0, 1.2), 24),
    }, mask


def pixel_evidence(inspection: Inspection, panel: PanelImage, x: int, y: int) -> dict:
    value = panel.signal[y, x]
    source_x, source_y = project(panel.matrix, np.array(x + 0.5), np.array(y + 0.5))
    scale = millimeter_scale(inspection.plane)
    cell = cell_at(inspection.plane, inspection.grid_adjustments, x + 0.5, y + 0.5)
    return {
        "x": x, "y": y, "cell": cell.id if cell else None,
        "source": [round(float(source_x), 3), round(float(source_y), 3)],
        "physical_mm": [round((x + 0.5) * scale[0], 3), round((y + 0.5) * scale[1], 3)] if scale else None,
        "signal": round(float(value), 6) if np.isfinite(value) else None,
        "usable": bool(panel.valid[y, x]),
        "warnings": support_reason(panel, x, y),
    }
