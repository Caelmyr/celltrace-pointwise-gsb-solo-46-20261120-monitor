from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import sqlite3
from uuid import uuid4

from .model import Inspection, Issue, MAX_HISTORY, file_id, sha256_id, utcnow


def serial_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def data_digest(value) -> str:
    return sha256(serial_json(value)).hexdigest()


def new_id() -> str:
    return uuid4().hex


class BlobStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def location(self, digest: str) -> Path:
        digest = sha256_id(digest)
        return self.path / digest[:2] / digest[2:]

    def put(self, content: bytes) -> str:
        digest = sha256(content).hexdigest()
        target = self.location(digest)
        if target.exists():
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{secrets.token_hex(5)}")
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def read(self, digest: str) -> bytes:
        target = self.location(digest)
        try:
            return target.read_bytes()
        except FileNotFoundError:
            raise Issue("The original image or report attachment is missing from local storage.", 409) from None

    def size(self, digest: str):
        try:
            return self.location(digest).stat().st_size
        except FileNotFoundError:
            raise Issue("The original image or report attachment is missing from local storage.", 409) from None


class Store:
    def __init__(self, data_dir: str):
        self.path = Path(data_dir).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.database = self.path / "inspection.sqlite3"
        self.blobs = BlobStore(self.path / "objects")
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY,
                    module TEXT NOT NULL,
                    lot TEXT NOT NULL,
                    updated TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    body BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS inspection_updates ON inspections(updated DESC, id ASC);
                CREATE INDEX IF NOT EXISTS inspection_lot_search ON inspections(lot COLLATE NOCASE);
                CREATE TABLE IF NOT EXISTS events (
                    serial INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL,
                    at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details BLOB NOT NULL,
                    UNIQUE(inspection, revision)
                );
                CREATE INDEX IF NOT EXISTS inspection_event_order ON events(inspection, revision DESC);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=5000")
            yield db
        finally:
            db.close()

    @contextmanager
    def write(self):
        with self.connection() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise

    def ping(self) -> bool:
        with self.connection() as db:
            return db.execute("SELECT 1").fetchone()[0] == 1

    def get(self, identifier: str) -> Inspection:
        file_id(identifier, "Inspection ID")
        with self.connection() as db:
            row = db.execute("SELECT body FROM inspections WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise Issue("This inspection does not exist.", 404)
        return Inspection.from_dict(json.loads(row["body"]))

    def list(self, query="", lot="", limit=70) -> list[dict]:
        args = []
        filters = []
        if query:
            escaped = like_pattern(query)
            filters.append("(module LIKE ? ESCAPE '~' OR lot LIKE ? ESCAPE '~')")
            args.extend((escaped, escaped))
        if lot:
            filters.append("lot = ? COLLATE NOCASE")
            args.append(lot)
        condition = f" WHERE {' AND '.join(filters)}" if filters else ""
        args.append(limit)
        with self.connection() as db:
            rows = db.execute(
                f"SELECT body FROM inspections{condition} ORDER BY updated DESC, id ASC LIMIT ?", args
            ).fetchall()
        return [summary(Inspection.from_dict(json.loads(row["body"]))) for row in rows]

    def bundle(self, identifier: str) -> tuple[Inspection, list[dict]]:
        file_id(identifier, "Inspection ID")
        with self.connection() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT body FROM inspections WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise Issue("This inspection does not exist.", 404)
            events = self.read_events(db, identifier, MAX_HISTORY)
            db.commit()
        return Inspection.from_dict(json.loads(row["body"])), events

    def add(self, inspection: Inspection, action: str, details=None) -> Inspection:
        fresh = Inspection.from_dict(inspection.to_dict())
        with self.write() as db:
            self.insert(db, fresh, action, details or {})
        return fresh

    def insert(self, db, inspection: Inspection, action: str, details: dict):
        db.execute(
            "INSERT INTO inspections(id, module, lot, updated, revision, body) VALUES (?, ?, ?, ?, ?, ?)",
            (
                inspection.id, inspection.module, inspection.lot, inspection.updated_at,
                inspection.revision, serial_json(inspection.to_dict()),
            ),
        )
        self.log(db, inspection, action, details)

    def save(self, inspection: Inspection, expected: int, action: str, details=None) -> Inspection:
        if inspection.revision != expected:
            raise Issue("The inspection changed. Reload before applying more changes.", 409)
        inspection.revision = expected + 1
        inspection.updated_at = utcnow()
        checked = Inspection.from_dict(inspection.to_dict())
        with self.write() as db:
            row = db.execute(
                "SELECT revision FROM inspections WHERE id=?", (checked.id,)
            ).fetchone()
            if row is None:
                raise Issue("This inspection does not exist.", 404)
            if row["revision"] != expected:
                raise Issue("The inspection changed. Reload before applying more changes.", 409)
            db.execute(
                "UPDATE inspections SET module=?, lot=?, updated=?, revision=?, body=? WHERE id=?",
                (
                    checked.module, checked.lot, checked.updated_at, checked.revision,
                    serial_json(checked.to_dict()), checked.id,
                ),
            )
            self.log(db, checked, action, details or {})
        return checked

    def events(self, identifier: str, limit=MAX_HISTORY) -> list[dict]:
        file_id(identifier, "Inspection ID")
        with self.connection() as db:
            return self.read_events(db, identifier, limit)

    def read_events(self, db, identifier: str, limit: int) -> list[dict]:
        rows = db.execute(
            """SELECT revision, at, action, details FROM events
               WHERE inspection=? ORDER BY revision DESC LIMIT ?""",
            (identifier, limit),
        ).fetchall()
        return [
            {"revision": row["revision"], "at": row["at"], "action": row["action"],
             "details": json.loads(row["details"])}
            for row in rows
        ]

    def log(self, db, inspection: Inspection, action: str, details: dict):
        db.execute(
            "INSERT INTO events(inspection, revision, at, action, details) VALUES (?, ?, ?, ?, ?)",
            (inspection.id, inspection.revision, inspection.updated_at, action, serial_json(details)),
        )


def like_pattern(query: str) -> str:
    return "%" + query.replace("~", "~~").replace("%", "~%").replace("_", "~_") + "%"


def summary(inspection: Inspection) -> dict:
    active_roles = sorted(inspection.active)
    synthetic = bool(inspection.captures) and all(
        capture.synthetic for capture in inspection.captures.values()
    )
    return {
        "id": inspection.id,
        "module": inspection.module,
        "lot": inspection.lot,
        "operator": inspection.operator,
        "revision": inspection.revision,
        "state": inspection.state,
        "updated_at": inspection.updated_at,
        "image_count": len(inspection.captures),
        "active_roles": active_roles,
        "findings": len(inspection.findings),
        "synthetic": synthetic,
    }
