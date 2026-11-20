"""Normalize the MP4 sample description of supported ByteVC2 files for FFmpeg."""
from __future__ import annotations

import shutil
import struct
from pathlib import Path
from typing import BinaryIO, Iterator


def _boxes(file: BinaryIO, start: int, end: int) -> Iterator[tuple[bytes, int, int, int]]:
    while start + 8 <= end:
        file.seek(start)
        header = file.read(8)
        if len(header) != 8:
            return
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            large_size = file.read(8)
            if len(large_size) != 8:
                return
            size = int.from_bytes(large_size, "big")
            header_size = 16
        elif size == 0:
            size = end - start
        if size < header_size or start + size > end:
            return
        yield kind, start, start + header_size, start + size
        start += size


def _video_descriptions(file: BinaryIO, end: int) -> Iterator[tuple[int, int]]:
    for kind, _, start, end in _boxes(file, 0, end):
        if kind != b"moov":
            continue
        parents = [(start, end)]
        for name in (b"trak", b"mdia", b"minf", b"stbl", b"stsd"):
            parents = [
                (start, end) for left, right in parents
                for kind, _, start, end in _boxes(file, left, right) if kind == name
            ]
        yield from parents
        return


def _vvc_config(data: bytes) -> bytes:
    # This ByteVC2 variant has an 18-byte private header followed by VVC NAL arrays.
    # Reuse the arrays as-is and signal 4-byte NAL lengths with no optional PTL.
    if len(data) < 19 or data[:2] != b"\x01\x07":
        raise ValueError("Cấu hình ByteVC2 không được hỗ trợ.")
    position = 19
    types = set()
    for _ in range(data[18]):
        if position + 3 > len(data):
            raise ValueError("Cấu hình ByteVC2 bị cắt ngắn.")
        nal_type = data[position] & 0x1F
        count = int.from_bytes(data[position + 1:position + 3], "big")
        position += 3
        for _ in range(count):
            size = int.from_bytes(data[position:position + 2], "big")
            position += 2
            if size < 2 or position + size > len(data) or data[position + 1] >> 3 != nal_type:
                raise ValueError("Cấu hình ByteVC2 chứa NAL không hợp lệ.")
            position += size
            types.add(nal_type)
    if position != len(data) or not {15, 16}.issubset(types):
        raise ValueError("Cấu hình ByteVC2 thiếu SPS/PPS hoặc có bố cục chưa được hỗ trợ.")
    return b"\0\0\0\0\xfe" + data[18:]


def normalize_mp4(source: Path, target: Path) -> bool:
    """Copy a ByteVC2 MP4 with a VVC sample entry, preserving samples and offsets."""
    if source.suffix.lower() not in (".mp4", ".m4v", ".mov"):
        return False
    patches: list[tuple[int, bytes]] = []
    with source.open("rb") as file:
        for start, end in _video_descriptions(file, source.stat().st_size):
            for kind, entry, payload, entry_end in _boxes(file, start + 8, end):
                if kind != b"bvc2":
                    continue
                for box_kind, offset, config, config_end in _boxes(file, payload + 78, entry_end):
                    if box_kind == b"bv2C":
                        if config_end - config > 1024 * 1024:
                            raise ValueError("Cấu hình ByteVC2 vượt quá kích thước hỗ trợ.")
                        file.seek(config)
                        replacement = _vvc_config(file.read(config_end - config))
                        size = 8 + len(replacement)
                        remaining = config_end - offset - size
                        if remaining < 8:
                            raise ValueError("Không đủ chỗ cho cấu hình VVC trong MP4.")
                        patches.append((entry + 4, b"vvi1"))
                        patches.append((offset, struct.pack(">I4s", size, b"vvcC") + replacement
                                        + struct.pack(">I4s", remaining, b"free") + bytes(remaining - 8)))
                        break
                else:
                    raise ValueError("Video ByteVC2 thiếu cấu hình bv2C.")
    if not patches:
        return False
    shutil.copyfile(source, target)
    with target.open("r+b") as file:
        for offset, data in patches:
            file.seek(offset)
            file.write(data)
    return True
