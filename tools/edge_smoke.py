#!/usr/bin/env python3
"""End-to-end smoke test for the cut edge inspection feature on synthetic demos."""
import json
import sys
import tempfile

sys.path.insert(0, "/workspace")

from celltrace.inspection import Inspections
from celltrace.store import Store
from celltrace.engine import Engine
from celltrace.model import EdgeSettings
from dataclasses import replace


def run(variant):
    with tempfile.TemporaryDirectory() as data_dir:
        store = Store(data_dir)
        engine = Engine(store)
        service = Inspections(store)
        inspection = service.demo(variant)
        context = engine.panel_context(inspection, "low" if variant == "stress" else "high")
        role = "low" if variant == "stress" else "high"
        edge_map = engine.edges(inspection, context)
        print(f"=== {variant} ({role}) ===")
        print("report:", json.dumps(edge_map.report, indent=2))
        for record in edge_map.records:
            print(
                f"  {record['cell']} {record['face']:6s} {record['status']:16s} "
                f"len={record['edge_length_px']:3d} mean={record['mean_setback_px']} "
                f"max={record['max_setback_px']} deact={record['deactivated_pixels']:4d} "
                f"missing={record['missing_positions']:3d} segs={len(record['missing_segments'])} "
                f"comps={record['edge_components']}"
            )
        # Render layer must not fail.
        from celltrace.render import render_layer, RenderOptions
        png = render_layer(inspection, engine, role, "edges", RenderOptions(True, True))
        print(f"  rendered edges layer: {len(png)} bytes")
        # Table + CSV.
        table = engine.table(inspection, role, "edges")
        from celltrace.tables import csv_data
        csv_data("edges", table["rows"])
        print(f"  table rows: {len(table['rows'])}; CSV ok")
        return inspection, engine, edge_map, role


if __name__ == "__main__":
    for variant in ("standard", "stress"):
        run(variant)
    print("SMOKE OK")
