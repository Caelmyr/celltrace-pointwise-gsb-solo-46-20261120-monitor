from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
import zipfile

from .model import Issue, MAX_PACKAGE_BYTES, MAX_UPLOAD_BYTES, integer, mapping, sha256_id, text


MAX_FILE_COUNT = 70
MAX_TOTAL_UNPACKED = 81 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_HEADER_SIZE = 30 * 1024


@dataclass(frozen=True)
class Member:
    name: str
    size: int
    digest: str
    content_type: str

    def to_dict(self):
        return {"name": self.name, "size": self.size, "sha256": self.digest, "content_type": self.content_type}

    @classmethod
    def from_dict(cls, raw: dict):
        raw = mapping(raw, "Archive member")
        name = validated_path(raw.get("name"))
        return cls(
            name,
            integer(raw.get("size"), "Archive member size", 1, MAX_UPLOAD_BYTES),
            sha256_id(raw.get("sha256")),
            text(raw.get("content_type"), "Archive content type", 80),
        )


def validated_path(value: str) -> str:
    value = text(value, "Archive path", 160)
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or ":" in value:
        raise Issue("The archive contains an unsupported member path.")
    if path.as_posix() != value or not path.parts or any(part in ("..", ".", "") for part in path.parts):
        raise Issue("The archive contains an unsupported member path.")
    if not all(part.isascii() for part in path.parts):
        raise Issue("The archive contains a nonportable member path.")
    return value


def member_metadata(name: str, content: bytes, mime: str) -> Member:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise Issue("An archive member exceeds its size limit.", 413)
    return Member(validated_path(name), len(content), sha256(content).hexdigest(), mime)


class PacketWriter:
    def __init__(self):
        self.buffer = BytesIO()
        self.archive = zipfile.ZipFile(self.buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=3)
        self.members = {}
        self.size = 0

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.archive.close()

    def add(self, name: str, content: bytes, mime: str) -> Member:
        member = member_metadata(name, content, mime)
        if member.name in self.members:
            raise Issue("An archive member name appears twice.")
        if len(self.members) + 1 >= MAX_FILE_COUNT or self.size + member.size > MAX_TOTAL_UNPACKED:
            raise Issue("This package exceeds its archive limits.", 413)
        info = zipfile.ZipInfo(member.name, (2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED if mime.startswith("image/") else zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        self.archive.writestr(info, content, compresslevel=3)
        self.members[member.name] = member
        self.size += member.size
        return member

    def finish(self, manifest: dict) -> bytes:
        payload = json.dumps(
            {**manifest, "files": [member.to_dict() for member in self.members.values()]},
            ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
        ).encode()
        if len(payload) > MAX_MANIFEST_BYTES:
            raise Issue("This package manifest exceeds its size limit.", 413)
        self.add("manifest.json", payload, "application/json")
        self.archive.close()
        result = self.buffer.getvalue()
        if len(result) > MAX_PACKAGE_BYTES:
            raise Issue("This inspection exchange package exceeds 64 MiB.", 413)
        return result


class PacketReader:
    def __init__(self, content: bytes):
        if not content or len(content) > MAX_PACKAGE_BYTES:
            raise Issue("Upload a nonempty inspection archive no larger than 64 MiB.", 413)
        try:
            self.archive = zipfile.ZipFile(BytesIO(content), "r")
        except (zipfile.BadZipFile, RuntimeError):
            raise Issue("The inspection archive is not a valid ZIP file.") from None
        try:
            self.entries = self.check_entries()
            self.manifest = self.read_manifest()
            self.members = self.check_manifest()
        except BaseException:
            self.archive.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.archive.close()

    def check_entries(self) -> dict:
        files = self.archive.infolist()
        if not 1 <= len(files) <= MAX_FILE_COUNT:
            raise Issue("The package has an unsupported number of members.")
        entries = {}
        total = 0
        for info in files:
            name = validated_path(info.filename)
            size_limit = MAX_MANIFEST_BYTES if name == "manifest.json" else MAX_UPLOAD_BYTES
            if info.is_dir() or info.file_size <= 0 or info.file_size > size_limit:
                raise Issue("An archived entry has an unsupported size or type.")
            if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise Issue("The archive uses an unsupported compression or encryption method.")
            if (info.external_attr >> 16) & 0o170000 not in (0, 0o100000):
                raise Issue("Only regular files can be imported from a package.")
            if len(info.extra) > MAX_HEADER_SIZE:
                raise Issue("A member header exceeds the archive limit.")
            if name in entries:
                raise Issue("An archive member name appears twice.")
            entries[name] = info
            total += info.file_size
        if total > MAX_TOTAL_UNPACKED:
            raise Issue("This package exceeds its expanded data limit.", 413)
        return entries

    def read_manifest(self):
        body = self.read_entry("manifest.json")
        try:
            raw = json.loads(body.decode("utf-8"), object_pairs_hook=unique_fields, parse_constant=reject_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise Issue("The archive manifest must contain valid UTF-8 JSON.") from None
        return mapping(raw, "Archive manifest")

    def check_manifest(self):
        raw = self.manifest.get("files")
        if not isinstance(raw, list) or len(raw) > MAX_FILE_COUNT:
            raise Issue("The archive file inventory is invalid.")
        files = {}
        for record in raw:
            member = Member.from_dict(record)
            if member.name in files:
                raise Issue("The manifest lists a member twice.")
            actual = self.entries.get(member.name)
            if actual is None or actual.file_size != member.size:
                raise Issue("A manifest member does not match the ZIP member size.")
            files[member.name] = member
        if set(files) | {"manifest.json"} != set(self.entries) or "manifest.json" in files:
            raise Issue("The package contents do not match the manifest.")
        return files

    def read_entry(self, name: str) -> bytes:
        if name not in self.entries:
            raise Issue(f"The package is missing {name}.")
        try:
            entry = self.entries[name]
            with self.archive.open(entry) as stream:
                content = stream.read(entry.file_size + 1)
        except (zipfile.BadZipFile, RuntimeError, EOFError, OSError):
            raise Issue(f"The archive member {name} could not be verified.") from None
        if len(content) != entry.file_size:
            raise Issue(f"The archive member {name} has a size mismatch.")
        return content

    def read(self, name: str) -> bytes:
        member = self.members.get(name)
        if member is None:
            raise Issue(f"The package is missing {name}.")
        content = self.read_entry(name)
        if sha256(content).hexdigest() != member.digest:
            raise Issue(f"The archive member {name} has a digest mismatch.")
        return content


def unique_fields(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"unsupported JSON constant: {value}")
