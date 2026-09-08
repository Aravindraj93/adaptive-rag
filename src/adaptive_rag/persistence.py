"""Versioned and checksummed persistence helpers for retrieval indexes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1


class IndexFormatError(ValueError):
    """Raised when an index manifest is malformed, incompatible, or corrupted."""


def validate_ranges(records, file_size: int) -> None:
    for record in records:
        if not isinstance(record, dict):
            raise IndexFormatError('invalid mapped record')
        offset, size = record.get('offset'), record.get('size')
        if (type(offset) is not int or type(size) is not int or offset < 0 or size < 1
                or offset+size > file_size):
            raise IndexFormatError('mapped record outside data file')


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("index metadata must be JSON-serializable") from error


def content_checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "checksum_algorithm": "sha256",
        "checksum": content_checksum(payload),
        "payload": payload,
    }
    descriptor, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(canonical_json(envelope) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_manifest(path: str | Path, *, max_bytes: int = 128*1024*1024) -> dict[str, Any]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('max_bytes must be positive')
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant: '+value)
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(max_bytes+1)
        if len(raw) > max_bytes:
            raise ValueError('manifest exceeds size limit')
        envelope = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise IndexFormatError(f"cannot read index manifest: {error}") from error
    if (not isinstance(envelope, dict) or type(envelope.get('schema_version')) is not int
            or envelope.get("schema_version") != SCHEMA_VERSION):
        raise IndexFormatError("unsupported index schema version")
    if envelope.get("checksum_algorithm") != "sha256":
        raise IndexFormatError("unsupported checksum algorithm")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise IndexFormatError("index payload must be an object")
    try:
        checksum = content_checksum(payload)
    except (TypeError, ValueError, RecursionError) as error:
        raise IndexFormatError('invalid JSON payload values') from error
    if envelope.get("checksum") != checksum:
        raise IndexFormatError("index checksum mismatch")
    return payload
