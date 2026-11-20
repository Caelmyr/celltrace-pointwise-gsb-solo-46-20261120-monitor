"""Per-cell dark-area release gate.

The gate reconciles human-confirmed finding regions with the measured dark-pixel
evidence on the high-current plane. Confirmed polygons are unioned so overlapping
annotations count once, then intersected with the connected dark map. Dark
components only partially covered by a confirmed region extend the cell's worst
case to their full connected extent, and unsupported pixels inside confirmed
regions add to the same upper bound. Cells whose upper bound stays under the
configured per-cell limit may release; crossing it asks for recapture or review.
"""

import numpy as np
from scipy import ndimage

from .findings import finding_status
from .geometry import cells_for, millimeter_scale, plane_polygon, polygon_mask


GATE_SCHEMA = "celltrace-release-gate-v1"
STRUCTURE = np.ones((3, 3), dtype=bool)
ACCOUNTING = ("region", "evidence", "extension", "unreviewed", "missing", "upper")


def confirmed_region(inspection) -> tuple[np.ndarray, list[str], list[dict]]:
    """Union of usable confirmed polygons, plus findings pending verification."""
    plane = inspection.plane
    region = np.zeros((plane.height, plane.width), dtype=bool)
    confirmed = []
    pending = []
    for finding in inspection.findings:
        state = finding_status(inspection, finding)
        if state != "current":
            pending.append({
                "id": finding.id, "title": finding.title, "cell": finding.cell,
                "status": finding.status, "source_status": state,
            })
            continue
        if finding.status != "confirmed" or finding.source_role != "high":
            continue
        region |= polygon_mask(plane_polygon(finding.points, plane), plane.width, plane.height)
        confirmed.append(finding.id)
    return region, confirmed, pending


def cell_accounting(dark, testable, marked, view) -> dict:
    labels, count = ndimage.label(dark.mask[view], structure=STRUCTURE)
    evidence = extension = unreviewed = 0
    if count:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        hits = np.bincount(labels[marked], minlength=count + 1)
        for marker in range(1, count + 1):
            inside = int(hits[marker])
            if inside:
                evidence += inside
                extension += int(sizes[marker]) - inside
            else:
                unreviewed += int(sizes[marker])
    missing = int((marked & ~testable[view]).sum())
    return {
        "region": int(marked.sum()),
        "evidence": evidence,
        "extension": extension,
        "unreviewed": unreviewed,
        "missing": missing,
        "upper": evidence + extension + unreviewed + missing,
    }


def release_assessment(inspection, context, dark) -> dict:
    plane = inspection.plane
    scale = millimeter_scale(plane)
    pixel_area = scale[0] * scale[1] if scale else None
    limit_mm2 = inspection.release.cell_dark_limit_mm2
    limit_px = limit_mm2 / pixel_area if limit_mm2 is not None and pixel_area else None
    region, confirmed, pending = confirmed_region(inspection)
    testable = np.isfinite(dark.reference)
    totals = dict.fromkeys(ACCOUNTING, 0)
    rows = []
    exceeding = []
    for cell in cells_for(plane):
        view = cell.view
        counted = cell_accounting(dark, testable, region[view], view)
        if not counted["region"] and not counted["upper"]:
            continue
        verdict = None
        if limit_px is not None:
            verdict = "exceed" if counted["upper"] > limit_px else "pass"
            if verdict == "exceed":
                exceeding.append(cell.id)
        row = {
            "cell": cell.id, "string": cell.string, "limit_mm2": limit_mm2, "verdict": verdict,
            **{f"{key}_px": counted[key] for key in ACCOUNTING},
        }
        if pixel_area:
            row.update({f"{key}_mm2": round(counted[key] * pixel_area, 3) for key in ACCOUNTING})
        rows.append(row)
        for key in ACCOUNTING:
            totals[key] += counted[key]
    summary = {f"{key}_px": value for key, value in totals.items()}
    if pixel_area:
        summary.update({f"{key}_mm2": round(value * pixel_area, 3) for key, value in totals.items()})
    if limit_mm2 is None:
        result = "unset"
    elif limit_px is None:
        result = "unevaluable"
    else:
        result = "exceed" if exceeding else "pass"
    return {
        "schema": GATE_SCHEMA,
        "role": "high",
        "capture": context.capture.id,
        "capture_digest": context.capture.digest,
        "rule": {"cell_dark_limit_mm2": limit_mm2},
        "scale_mm_per_px": [round(scale[0], 6), round(scale[1], 6)] if scale else None,
        "limit_px": round(limit_px, 3) if limit_px is not None else None,
        "result": result,
        "cells_total": plane.rows * plane.cols,
        "judged_cells": [row["cell"] for row in rows],
        "exceeding_cells": exceeding,
        "confirmed_findings": confirmed,
        "pending_findings": pending,
        "totals": summary,
        "cells": rows,
    }
