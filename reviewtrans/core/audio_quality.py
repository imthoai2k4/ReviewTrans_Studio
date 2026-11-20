"""Kiểm tra WAV TTS bằng PCM, không phụ thuộc thư viện xử lý âm thanh ngoài."""
from __future__ import annotations

import math
import hashlib
import re
import sys
import wave
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class AudioQuality:
    duration: float
    active_duration: float
    rms: float
    peak: float


def inspect_wav(path: Path) -> AudioQuality:
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError("Cần WAV PCM 16-bit để kiểm tra voice")
        rate, channels = audio.getframerate(), audio.getnchannels()
        block = max(1, round(rate * 0.02))
        energy, count, peak = 0.0, 0, 0
        frames: list[tuple[float, float]] = []
        while data := audio.readframes(block):
            values = array("h")
            values.frombytes(data)
            if sys.byteorder != "little":
                values.byteswap()
            power = sum(value * value for value in values)
            n = len(values)
            peak = max(peak, max((abs(value) for value in values), default=0))
            energy += power
            count += n
            frames.append((math.sqrt(power / n) / 32768, n / channels / rate))
        threshold = max(0.002, peak / 32768 * 0.04)
        return AudioQuality(
            count / channels / rate,
            sum(seconds for rms, seconds in frames if rms >= threshold),
            math.sqrt(energy / count) / 32768 if count else 0.0,
            peak / 32768,
        )


def speech_units(text: str, language: str) -> int:
    # Tiếng Việt tách âm tiết bằng khoảng trắng; CJK không cần khoảng trắng.
    cjk = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    rest = re.sub(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", text)
    words = re.findall(r"[^\W_]+", rest, re.UNICODE)
    if language.split("-")[0].lower() == "vi":
        return len(cjk) + len(words)
    # Chặn trường hợp từ rất dài bị đọc thiếu ở các ngôn ngữ dùng chữ Latin.
    return len(cjk) + sum(max(1, math.ceil(len(word) / 7)) for word in words)


@lru_cache(maxsize=2048)
def _quality_error(path: str, mtime_ns: int, size: int, digest: bytes, text: str, speed: float, language: str) -> str | None:
    try:
        quality = inspect_wav(Path(path))
    except (OSError, ValueError, EOFError, wave.Error) as exc:
        return f"Không đọc được WAV TTS: {exc}"
    if quality.duration <= 0 or quality.rms < 0.002:
        return "Voice rỗng hoặc gần im lặng"
    # Ngưỡng bảo thủ: chỉ bắt âm thanh ngắn bất thường, không ép TTS dài bằng ASR.
    units = speech_units(text, language)
    minimum = max(0.16, units * 0.14) / max(0.25, min(4.0, speed))
    active_minimum = minimum * (0.7 if units > 1 else 0.45)
    if quality.duration + 0.005 < minimum or quality.active_duration + 0.005 < active_minimum:
        return f"Voice quá ngắn so với nội dung ({quality.duration:.2f}s, phần có tiếng {quality.active_duration:.2f}s); có thể đọc thiếu"
    return None


def audio_quality_error(path: Path, text: str, speed: float = 1.0, language: str = "vi") -> str | None:
    try:
        stat = path.stat()
        # Windows có thể giữ cùng mtime/kích thước khi ghi lại rất nhanh. Hash
        # PCM bảo đảm lần tạo lại không dùng nhầm kết quả kiểm tra lần trước.
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").digest()
    except OSError as exc:
        return f"Không đọc được file TTS: {exc}"
    return _quality_error(str(path.resolve()), stat.st_mtime_ns, stat.st_size, digest, text, speed, language)
