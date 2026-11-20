"""Kiểm tra bản đóng gói có đủ thư viện và công cụ: ReviewTrans.exe --self-check <file.json>."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

MODULES = [
    "requests", "openai", "anthropic", "google.genai", "py7zr",
    "PyQt6.QtWidgets", "PyQt6.QtMultimedia", "reviewtrans.ui.player.qt_backend",
    "reviewtrans.ui.main_window", "reviewtrans.core.pipeline.runner",
]


def run(output: Path, require: list[str]) -> int:
    from . import APP_VERSION
    from .core.paths import is_portable, register_dll_dirs
    from .core.proc import subprocess_kwargs
    from .core.pipeline.resources import tool_status

    result: dict = {"version": APP_VERSION, "frozen": bool(getattr(sys, "frozen", False)), "modules": {}}
    ok = True
    for name in MODULES:
        try:
            importlib.import_module(name)
            result["modules"][name] = "ok"
        except Exception as exc:  # noqa: BLE001
            result["modules"][name] = f"lỗi: {exc}"
            ok = False
    register_dll_dirs()
    try:
        importlib.import_module("mpv")
        result["libmpv"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["libmpv"] = f"không nạp được: {exc}"
        if "libmpv" in require:
            ok = False
    result["tools"] = tool_status()
    if "funasr" in require:
        from .core.pipeline.funasr import models_ready, worker_command

        asr_report = output.with_name(output.stem + ".funasr.json")
        try:
            code = subprocess.run([*worker_command(), "--check", "--output", str(asr_report)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=120, **subprocess_kwargs()).returncode
            result["funasr"] = json.loads(asr_report.read_text(encoding="utf-8")) if asr_report.is_file() else {}
            result["funasr"]["models_ready"] = models_ready()
            if code or not models_ready():
                ok = False
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            result["funasr"] = {"error": str(exc)}
            ok = False
    result["runs"] = {}
    for tool in require:
        if tool in ("libmpv", "funasr"):
            continue
        path = result["tools"].get(tool)
        if not path:
            ok = False
            continue
        # chạy thử để bắt lỗi thiếu DLL (Windows trả 0xC0000135) — có file chưa chắc đã chạy được
        try:
            code = subprocess.run(
                [path, "-version" if tool in ("ffmpeg", "ffprobe") else "--help"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, **subprocess_kwargs(),
            ).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            code = f"lỗi: {exc}"
        result["runs"][tool] = code
        if not isinstance(code, int) or code < 0 or code > 255:
            ok = False
    result["portable"] = is_portable()
    result["ok"] = ok
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok else 1
