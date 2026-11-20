"""Dò phần cứng tăng tốc: GPU và bộ mã hoá video của ffmpeg.

Mọi thứ đều được *chạy thử thật* thay vì đoán theo tên: bản ffmpeg có thể liệt kê h264_nvenc
nhưng máy không có card NVIDIA, driver cũ có thể thiếu AMF... Kết quả được lưu vào
`hardware.json` trong thư mục dữ liệu để lần sau khỏi dò lại.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .paths import IS_WINDOWS, bin_dirs, find_tool, user_data_dir
from .proc import subprocess_kwargs

LogFn = Callable[[str], None]
CACHE_VERSION = 3
CACHE_MAX_AGE = 7 * 24 * 3600

VENDORS = {"10de": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel"}
VENDOR_LABELS = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "other": "Khác"}
ENCODER_VENDOR = {"nvenc": "nvidia", "amf": "amd", "qsv": "intel", "mf": "", "": ""}

# codec → (nhãn, họ). "auto" chọn bộ mã hoá GPU tốt nhất của họ đó, không có thì dùng CPU.
CODECS: list[tuple[str, str]] = [
    ("auto", "Tự động H.264 (ưu tiên GPU rời + CPU)"),
    ("auto_hevc", "Tự động H.265 (ưu tiên GPU rời + CPU)"),
    ("libx264", "H.264 CPU (x264, tương thích nhất)"),
    ("libx265", "H.265 CPU (x265, nhỏ hơn, chậm)"),
    ("h264_nvenc", "H.264 NVIDIA NVENC"),
    ("hevc_nvenc", "H.265 NVIDIA NVENC"),
    ("av1_nvenc", "AV1 NVIDIA NVENC (RTX 40+)"),
    ("h264_amf", "H.264 AMD AMF"),
    ("hevc_amf", "H.265 AMD AMF"),
    ("av1_amf", "AV1 AMD AMF (RX 7000+)"),
    ("h264_qsv", "H.264 Intel Quick Sync"),
    ("hevc_qsv", "H.265 Intel Quick Sync"),
    ("av1_qsv", "AV1 Intel Quick Sync (Arc)"),
    ("h264_mf", "H.264 Windows Media Foundation (GPU bất kỳ)"),
    ("hevc_mf", "H.265 Windows Media Foundation (GPU bất kỳ)"),
]
CODEC_LABELS = dict(CODECS)
AUTO_ORDER = {
    "auto": ["h264_nvenc", "h264_amf", "h264_qsv", "h264_mf", "libx264"],
    "auto_hevc": ["hevc_nvenc", "hevc_amf", "hevc_qsv", "hevc_mf", "libx265"],
}
SOFTWARE = {"h264": "libx264", "hevc": "libx265", "av1": "libx264"}
HW_CODECS = [c for c, _ in CODECS if c not in ("auto", "auto_hevc", "libx264", "libx265")]

ASR_DEVICES = [
    ("auto", "Tự động (CUDA → CPU)"),
    ("cuda", "NVIDIA CUDA"),
    ("cpu", "Chỉ CPU"),
]
VARIANT_NAMES = {"cuda": "CUDA", "vulkan": "Vulkan", "cpu": "CPU"}


# ------------------------------------------------------------------ dữ liệu


@dataclass
class Gpu:
    name: str
    vendor: str = "other"  # nvidia | amd | intel | other
    adapter: int = -1  # chỉ số DXGI (dùng cho -init_hw_device d3d11va=…:N)
    integrated: bool = False

    @property
    def label(self) -> str:
        return f"{self.name} ({'tích hợp' if self.integrated else 'rời'})"


@dataclass
class EncoderConfig:
    """Cách gọi một bộ mã hoá đã chạy thử thành công."""

    codec: str
    pre_args: list[str] = field(default_factory=list)  # đặt trước các -i
    tail: str = "format=yuv420p"  # cuối filter graph
    args: list[str] = field(default_factory=list)  # tham số riêng ngoài chất lượng (vd. -gpu 1)
    device: str = ""  # mô tả thiết bị để hiển thị
    discrete: bool = False
    ffmpeg: str = ""  # FFmpeg thay thế đã chạy thử với bộ mã hoá này

    @property
    def hardware(self) -> bool:
        return self.codec not in ("libx264", "libx265")


@dataclass
class HardwareInfo:
    version: int = CACHE_VERSION
    detected_at: float = 0.0
    ffmpeg: str = ""  # chữ ký bản ffmpeg đã dò
    gpus: list[Gpu] = field(default_factory=list)
    cuda_version: str = ""  # CUDA tối đa driver NVIDIA hỗ trợ (từ nvidia-smi)
    nvidia_driver: str = ""
    compute_cap: str = ""
    encoders: dict[str, EncoderConfig] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)  # codec → lỗi
    alternate_ffmpeg: str = ""  # chữ ký FFmpeg thay thế đã dò

    def vendors(self) -> set[str]:
        return {g.vendor for g in self.gpus}

    def has_nvidia(self) -> bool:
        return "nvidia" in self.vendors() or bool(self.nvidia_driver)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareInfo":
        info = cls(**{k: v for k, v in data.items() if k in (
            "version", "detected_at", "ffmpeg", "cuda_version", "nvidia_driver", "compute_cap", "failed", "alternate_ffmpeg",
        )})
        info.gpus = [Gpu(**g) for g in data.get("gpus", [])]
        info.encoders = {k: EncoderConfig(**v) for k, v in data.get("encoders", {}).items()}
        return info

    def summary(self) -> str:
        gpus = ", ".join(g.name for g in self.gpus) or "không thấy GPU"
        encoders = ", ".join(self.encoders) or "chỉ CPU"
        return f"GPU: {gpus}. Mã hoá: {encoders}."


# ------------------------------------------------------------------ chạy lệnh


def _run(command: list[str], timeout: float = 30) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, stdin=subprocess.DEVNULL, **subprocess_kwargs(),
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "quá thời gian"
    except OSError as exc:
        return -1, str(exc)


def _ffmpeg_signature(ffmpeg: Path | None) -> str:
    if ffmpeg is None:
        return ""
    try:
        stat = ffmpeg.stat()
    except OSError:
        return ""
    return f"{ffmpeg}|{stat.st_size}|{int(stat.st_mtime)}"


def _system_ffmpeg(primary: Path) -> Path | None:
    """Find an installed Windows FFmpeg other than the one supplied by the app."""
    primary = primary.resolve()
    for directory in map(Path, os.get_exec_path()):
        if directory.is_absolute():
            candidate = directory / "ffmpeg.exe"
            if candidate.is_file() and candidate.resolve() != primary:
                return candidate.resolve()
    return None


# ------------------------------------------------------------------ GPU

_INTEGRATED = re.compile(
    r"(UHD|Iris|HD Graphics|Radeon\(TM\) Graphics|Radeon Graphics|Vega \d+ Graphics|Basic Render)", re.I
)
_DEVICE_LINE = re.compile(r"Using device ([0-9a-fA-F]{4}):[0-9a-fA-F]{4} \((.+)\)\.")


def list_gpus(ffmpeg: Path | None) -> list[Gpu]:
    """Liệt kê adapter DXGI qua ffmpeg (Windows). Chỉ số khớp với -init_hw_device d3d11va=…:N."""
    gpus: list[Gpu] = []
    if not IS_WINDOWS or ffmpeg is None:
        return gpus
    for index in range(6):
        code, output = _run(
            [str(ffmpeg), "-hide_banner", "-v", "verbose", "-init_hw_device", f"d3d11va=dx:{index}",
             "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.04", "-frames:v", "1", "-f", "null", "-"],
            timeout=15,
        )
        match = _DEVICE_LINE.search(output)
        if not match:
            break  # hết adapter
        vendor_id, name = match.group(1).lower(), match.group(2).strip()
        if vendor_id == "1414":  # Microsoft Basic Render Driver (phần mềm)
            continue
        if code == 0:
            gpus.append(Gpu(name=name, vendor=VENDORS.get(vendor_id, "other"), adapter=index,
                            integrated=bool(_INTEGRATED.search(name))))
    return gpus


def nvidia_smi() -> Path | None:
    found = shutil.which("nvidia-smi")
    if found:
        return Path(found)
    for candidate in (Path(r"C:\Windows\System32\nvidia-smi.exe"),
                      Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")):
        if candidate.is_file():
            return candidate
    return None


def nvidia_details() -> tuple[str, str, str, list[str]]:
    """(phiên bản CUDA driver hỗ trợ, driver, compute capability, tên GPU)."""
    smi = nvidia_smi()
    if smi is None:
        return "", "", "", []
    code, output = _run([str(smi)], timeout=15)
    if code != 0:
        return "", "", "", []
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", output)
    code, query = _run([str(smi), "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"], timeout=15)
    names, driver, cap = [], "", ""
    if code == 0:
        for line in query.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0]:
                names.append(parts[0])
                driver = driver or parts[1]
                cap = cap or (parts[2] if len(parts) > 2 else "")
    return (cuda.group(1) if cuda else ""), driver, cap, names


# ------------------------------------------------------------------ bộ mã hoá


def family(codec: str) -> str:
    if codec in AUTO_ORDER:
        return "hevc" if codec == "auto_hevc" else "h264"
    if codec.startswith(("hevc", "libx265")):
        return "hevc"
    if codec.startswith("av1"):
        return "av1"
    return "h264"


def encoder_vendor(codec: str) -> str:
    return codec.split("_", 1)[1] if "_" in codec and not codec.startswith("lib") else ""


def quality_args(codec: str, crf: int, preset: str = "medium") -> list[str]:
    """Quy đổi CRF (thang x264) sang tham số chất lượng của từng bộ mã hoá."""
    crf = max(0, min(51, int(crf)))
    kind = encoder_vendor(codec)
    if kind == "nvenc":
        return ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    if kind == "amf":
        return ["-quality", "quality", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]
    if kind == "qsv":
        return ["-preset", "medium", "-global_quality", str(crf)]
    if kind == "mf":
        return ["-hw_encoding", "1", "-rate_control", "quality", "-quality", str(max(1, min(100, round(100 - crf * 1.5))))]
    return ["-preset", preset or "medium", "-crf", str(crf)]


def _candidates(codec: str, gpus: list[Gpu]) -> list[EncoderConfig]:
    """Các cách gọi cần thử cho một bộ mã hoá (theo thứ tự ưu tiên)."""
    kind = encoder_vendor(codec)
    if not kind:
        return [EncoderConfig(codec)]
    vendors = {g.vendor for g in gpus}
    # thử GPU rời trước GPU tích hợp
    ordered = sorted(gpus, key=lambda g: (g.integrated, g.adapter))
    if kind == "nvenc":
        if IS_WINDOWS and gpus and "nvidia" not in vendors:
            return []
        nvidia = [g for g in ordered if g.vendor == "nvidia"]
        return [EncoderConfig(
            codec, device=nvidia[0].name if len(nvidia) == 1 else "NVIDIA", discrete=True,
        )]
    if kind == "amf":
        if not IS_WINDOWS:
            return []
        # AMF mặc định chạy trên adapter 0 — trên laptop hai GPU AMD thường là GPU tích hợp (driver cũ, dễ lỗi)
        return [
            EncoderConfig(
                codec,
                pre_args=["-init_hw_device", f"d3d11va=rtdx:{g.adapter}", "-filter_hw_device", "rtdx"],
                tail="format=nv12,hwupload",
                device=g.name,
                discrete=not g.integrated,
            )
            for g in ordered if g.vendor == "amd"
        ]
    if kind == "qsv":
        if IS_WINDOWS and gpus and "intel" not in vendors:
            return []
        default = EncoderConfig(codec, tail="format=nv12", device="Intel")
        configs = [default]
        for g in ordered:
            if g.vendor == "intel":
                configs.append(EncoderConfig(
                    codec,
                    pre_args=["-init_hw_device", f"d3d11va=rtdx:{g.adapter}", "-init_hw_device", "qsv=rtqs@rtdx",
                              "-filter_hw_device", "rtqs"],
                    tail="format=nv12,hwupload=extra_hw_frames=64",
                    device=g.name,
                    discrete=not g.integrated,
                ))
        return sorted(configs, key=lambda config: not config.discrete)
    if kind == "mf":
        return [EncoderConfig(codec, tail="format=nv12", device="Media Foundation")] if IS_WINDOWS and gpus else []
    return []


def probe_command(ffmpeg: Path, config: EncoderConfig, crf: int = 23) -> list[str]:
    return [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", *config.pre_args,
        "-f", "lavfi", "-i", "color=black:s=1280x720:r=30:d=0.5", "-frames:v", "8",
        "-vf", config.tail, "-c:v", config.codec, *quality_args(config.codec, crf), *config.args,
        "-f", "null", "-",
    ]


NVENC_API_MISMATCH = "does not support the required nvenc API version"


def probe_encoder(ffmpeg: Path, config: EncoderConfig, attempts: int = 4) -> tuple[bool, str]:
    """Mã hoá thử vài khung hình, thử lại ngay nếu lỗi: GPU rời trên laptop tự tắt sau ~1 giây rảnh và lần gọi
    đầu tiên khi nó vừa thức dậy thường lỗi (AMF báo lỗi 30). Chờ lâu giữa các lần thử chỉ làm GPU ngủ lại."""
    message = ""
    for attempt in range(attempts):
        code, output = _run(probe_command(ffmpeg, config), timeout=40)
        if code == 0:
            return True, ""
        message = next((line.strip() for line in output.splitlines() if line.strip()), f"mã lỗi {code}")
        if "Cannot load" in output or NVENC_API_MISMATCH in output:
            break  # thiếu driver/DLL (vd. nvcuda.dll), thử lại vô ích
        if attempt + 1 < attempts:
            time.sleep(0.2)
    return False, message


def _encoder_list(ffmpeg: Path) -> set[str]:
    code, output = _run([str(ffmpeg), "-hide_banner", "-encoders"], timeout=20)
    if code != 0:
        return set()
    return {parts[1] for parts in (line.split() for line in output.splitlines()) if len(parts) > 1 and parts[0].startswith("V")}


# ------------------------------------------------------------------ dò + cache

_lock = threading.Lock()
_cached: HardwareInfo | None = None


def cache_path() -> Path:
    return user_data_dir() / "hardware.json"


def load_cached() -> HardwareInfo | None:
    """Kết quả dò gần nhất (không chạy gì). None nếu chưa dò hoặc bản ffmpeg đã đổi."""
    global _cached
    if _cached is not None:
        return _cached
    try:
        info = HardwareInfo.from_dict(json.loads(cache_path().read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None
    if info.version != CACHE_VERSION or info.ffmpeg != _ffmpeg_signature(find_tool("ffmpeg")):
        return None
    if info.alternate_ffmpeg:
        if info.alternate_ffmpeg != _ffmpeg_signature(Path(info.alternate_ffmpeg.split("|", 1)[0])):
            return None
    if time.time() - info.detected_at > CACHE_MAX_AGE:
        return None
    _cached = info
    return info


def detect(force: bool = False, log: LogFn | None = None) -> HardwareInfo:
    """Dò GPU và chạy thử từng bộ mã hoá phần cứng. Mất vài giây — gọi trong luồng nền."""
    global _cached
    with _lock:
        if not force:
            cached = load_cached()
            if cached is not None:
                return cached
        say = log or (lambda _m: None)
        ffmpeg = find_tool("ffmpeg")
        info = HardwareInfo(detected_at=time.time(), ffmpeg=_ffmpeg_signature(ffmpeg))
        info.gpus = list_gpus(ffmpeg)
        info.cuda_version, info.nvidia_driver, info.compute_cap, nvidia_names = nvidia_details()
        if nvidia_names and not any(g.vendor == "nvidia" for g in info.gpus):
            info.gpus += [Gpu(name=n, vendor="nvidia") for n in nvidia_names]  # máy không phải Windows
        say("GPU: " + (", ".join(g.label for g in info.gpus) or "không thấy"))
        if ffmpeg is not None:
            available = _encoder_list(ffmpeg)
            alternate = _system_ffmpeg(ffmpeg) if IS_WINDOWS else None
            for codec in HW_CODECS:
                if codec not in available:
                    continue
                configs = _candidates(codec, info.gpus)
                if not configs:
                    continue
                error = ""
                for config in configs:
                    ok, error = probe_encoder(ffmpeg, config)
                    if not ok and alternate and encoder_vendor(codec) == "nvenc" and NVENC_API_MISMATCH in error:
                        ok, alt_error = probe_encoder(alternate, config)
                        if ok:
                            config.ffmpeg = str(alternate)
                            info.alternate_ffmpeg = _ffmpeg_signature(alternate)
                            say(f"  {codec}: dùng FFmpeg tương thích tại {alternate}")
                        else:
                            say(f"  FFmpeg tại {alternate}, {codec}: {alt_error}")
                    if ok:
                        info.encoders[codec] = config
                        say(f"  ✓ {codec} ({config.device})")
                        break
                else:
                    info.failed[codec] = error
                    say(f"  ✕ {codec}: {error}")
        _cached = info
        try:
            cache_path().write_text(json.dumps(info.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return info


def forget() -> None:
    """Xoá kết quả dò (dùng trong test / khi đổi ffmpeg)."""
    global _cached
    _cached = None


def resolve_encoder(requested: str, info: HardwareInfo | None, exclude=()) -> tuple[EncoderConfig, str]:
    """Chọn bộ mã hoá thực tế cho giá trị trong cài đặt. Trả về (config, ghi chú).
    `exclude`: các bộ mã hoá vừa lỗi trong lần xuất này."""
    requested = requested or "auto"
    software = EncoderConfig(SOFTWARE[family(requested)])
    if requested in ("libx264", "libx265"):
        return EncoderConfig(requested), ""
    skip = set(exclude)
    encoders = {k: v for k, v in (info.encoders if info else {}).items() if k not in skip}
    if requested in AUTO_ORDER:
        available = [encoders[codec] for codec in AUTO_ORDER[requested] if codec in encoders]
        if not available:
            return software, "Không có bộ mã hoá GPU dùng được → dùng CPU"
        chosen = next((config for config in available if config.discrete), available[0])
        return chosen, f"Tự động chọn {chosen.codec} ({chosen.device})"
    if requested in encoders:
        return encoders[requested], ""
    if info is None and requested not in skip:  # chưa dò: cứ thử, lỗi sẽ chuyển sang CPU
        return EncoderConfig(requested, tail="format=nv12" if encoder_vendor(requested) in ("qsv", "mf") else "format=yuv420p"), ""
    reason = (info.failed.get(requested) if info else "") or ("vừa lỗi" if requested in skip else "không có trên máy này")
    return software, f"{requested} không dùng được ({reason}) → dùng {software.codec}"
