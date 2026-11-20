#!/usr/bin/env python3
"""HTTP-level verification for the cut edge feature."""
import json
import sys
import tempfile

sys.path.insert(0, "/workspace")

from celltrace.web import build_app
from celltrace.store import Store


def main():
    with tempfile.TemporaryDirectory() as data_dir:
        app = build_app(data_dir)
        client = app.test_client()
        created = client.post("/api/demo/standard")
        assert created.status_code == 200, created.data
        inspection = created.get_json()
        ident = inspection["id"]
        revision = inspection["revision"]

        measure = client.get(f"/api/inspections/{ident}/measure/edges?role=high")
        assert measure.status_code == 200, measure.data
        payload = measure.get_json()
        assert payload["kind"] == "edges"
        assert payload["summary"]["cell_edges"] == 8
        print("measure edges summary keys:", sorted(payload["summary"].keys()))
        print("first row keys:", sorted(payload["rows"][0].keys()))

        csv_resp = client.get(f"/api/inspections/{ident}/measure/edges?role=high&format=csv")
        assert csv_resp.status_code == 200
        print("CSV header:", csv_resp.data.decode().splitlines()[0])

        layer = client.get(f"/api/inspections/{ident}/layer/edges.png?role=high")
        assert layer.status_code == 200 and layer.data[:4] == b"\x89PNG"
        print("edge layer PNG bytes:", len(layer.data))

        # PATCH edge settings: horizontal axis, single face.
        patch = client.patch(
            f"/api/inspections/{ident}/edges",
            json={"revision": revision, "axis": "horizontal", "boundary": 0.5,
                  "top_face": True, "bottom_face": False, "enabled": True},
        )
        assert patch.status_code == 200, patch.data
        updated = patch.get_json()
        assert updated["edges"]["axis"] == "horizontal"
        # For a horizontal seam the two face flags are stored in left_face/right_face.
        assert updated["edges"]["left_face"] is True
        assert updated["edges"]["right_face"] is False
        revision = updated["revision"]
        print("PATCH edges ->", updated["edges"])

        horizontal = client.get(f"/api/inspections/{ident}/measure/edges?role=high")
        assert horizontal.status_code == 200
        hrows = horizontal.get_json()["rows"]
        assert all(row["axis"] == "horizontal" and row["face"] == "top" for row in hrows)
        print("horizontal rows:", len(hrows), "faces:", sorted({r["face"] for r in hrows}))
        # Disable edge inspection.
        patch2 = client.patch(
            f"/api/inspections/{ident}/edges",
            json={"revision": revision, "enabled": False, "axis": "vertical",
                  "left_face": True, "right_face": True},
        )
        assert patch2.status_code == 200
        disabled = client.get(f"/api/inspections/{ident}/measure/edges?role=high")
        assert disabled.status_code == 409
        print("disabled measure status:", disabled.status_code, disabled.get_json()["error"])

        # Invalid: no face selected.
        bad = client.patch(
            f"/api/inspections/{ident}/edges",
            json={"revision": patch2.get_json()["revision"], "enabled": True,
                  "axis": "vertical", "left_face": False, "right_face": False},
        )
        assert bad.status_code == 400
        print("invalid faces rejected:", bad.get_json()["error"])

        # Cell details include edge records.
        cell = client.get(f"/api/inspections/{ident}/cells/R01C03?role=high")
        assert cell.status_code == 200
        assert "edges" in cell.get_json()
        print("cell details edge entries:", len(cell.get_json()["edges"]))

        # Report preview includes edges section when re-enabled.
        rev = patch2.get_json()["revision"]
        client.patch(f"/api/inspections/{ident}/edges",
                     json={"revision": rev, "enabled": True, "axis": "vertical",
                           "left_face": True, "right_face": True})
        report = client.get(f"/api/inspections/{ident}/report")
        assert report.status_code == 200
        assert "edges" in report.get_json()["sections"]
        print("report sections:", sorted(report.get_json()["sections"].keys()))

    print("HTTP OK")


if __name__ == "__main__":
    main()
