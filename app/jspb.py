"""Helpers for Google's JSPB format: protobuf sent as positional JSON arrays.

In JSPB there are no keys. The array index is the field number, and empty
fields are kept as null so positions never shift.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Iterable

XSSI_PREFIX = ")]}'"


def loads(text: str) -> Any:
    """Strip Google's anti-hijacking prefix and parse the JSON."""
    text = text.lstrip()
    if text.startswith(XSSI_PREFIX):
        text = text[len(XSSI_PREFIX):]
    return json.loads(text)


def get(data: Any, path: Iterable[int], default: Any = None) -> Any:
    """Safely read data[a][b][c]...; return default if any step is missing."""
    cur = data
    for idx in path:
        if not isinstance(cur, list) or idx >= len(cur) or cur[idx] is None:
            return default
        cur = cur[idx]
    return cur


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def decode_proto_strings(b64: str) -> dict[int, str]:
    """Decode a base64 protobuf blob and return its top-level string fields.

    Only length-delimited fields (wire type 2) that are valid UTF-8 are kept.
    """
    raw = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4))
    out: dict[int, str] = {}
    i = 0
    try:
        while i < len(raw):
            key, i = _read_varint(raw, i)
            field, wire = key >> 3, key & 7
            if wire == 0:
                _, i = _read_varint(raw, i)
            elif wire == 2:
                length, i = _read_varint(raw, i)
                chunk = raw[i:i + length]
                i += length
                try:
                    out[field] = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            else:
                break
    except IndexError:
        pass
    return out
