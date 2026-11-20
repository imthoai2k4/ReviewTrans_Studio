from __future__ import annotations

import struct

import pytest

from reviewtrans.core.bytevc2 import normalize_mp4


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _mp4(config: bytes) -> bytes:
    entry = _box(b"bvc2", bytes(78) + _box(b"bv2C", config))
    table = _box(b"stsd", bytes(4) + b"\0\0\0\1" + entry)
    for kind in (b"stbl", b"minf", b"mdia", b"trak", b"moov"):
        table = _box(kind, table)
    # A box with the same names inside media data cannot be mistaken for metadata.
    return _box(b"ftyp", b"isom") + _box(b"mdat", entry) + table


def test_bytevc2_normalizes_only_mp4_sample_description(tmp_path):
    arrays = b"\x02\x8f\0\x01\0\x03\0\x79\x80\x90\0\x01\0\x03\0\x81\x80"
    config = b"\x01\x07" + bytes(16) + arrays
    original = _mp4(config)
    source = tmp_path / "source.mp4"
    source.write_bytes(original)
    target = tmp_path / "source_vvc.mp4"

    assert normalize_mp4(source, target)
    patched = target.read_bytes()
    assert len(patched) == len(original)
    assert patched[:original.index(b"moov")] == original[:original.index(b"moov")]
    assert patched.count(b"bvc2") == 1
    assert b"vvi1" in patched
    assert b"vvcC\0\0\0\0\xfe" + arrays in patched
    assert source.read_bytes() == original
    assert not normalize_mp4(target, tmp_path / "already_normalized.mp4")


def test_bytevc2_rejects_truncated_config(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "source_vvc.mp4"
    source.write_bytes(_mp4(b"\x01\x07" + bytes(16) + b"\x01\x8f"))

    with pytest.raises(ValueError, match="bị cắt ngắn"):
        normalize_mp4(source, target)
    assert not target.exists()
