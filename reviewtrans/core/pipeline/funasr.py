"""Paraformer integration without importing optional ML packages into the UI."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from ..config import asr_thread_count
from ..models import Segment, VideoDoc
from ..hardware import load_cached
from ..paths import IS_WINDOWS, exe_dir, models_dir, resource_path, temp_dir, user_data_dir
from ..proc import ToolMissing, run, run_checked, subprocess_kwargs
from ..srt import render_srt
from . import RunContext
from .asr_boundaries import rebuild_sentences

MODEL_NAMES = ("paraformer-zh", "fsmn-vad", "ct-punc", "campplus")
EVENT_PREFIX = "REVIEWTRANS_FUNASR "


def runtime_dir() -> Path:
    return user_data_dir() / "funasr-runtime"


def runtime_python() -> Path:
    override = os.environ.get("REVIEWTRANS_FUNASR_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
    embedded = runtime_dir() / "python.exe"
    if IS_WINDOWS and embedded.is_file():
        return embedded
    bundled = exe_dir() / "asr-runtime" / "python.exe"
    if getattr(sys, "frozen", False) and bundled.is_file():
        return bundled
    return runtime_dir() / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def funasr_models_dir() -> Path:
    managed = models_dir() / "funasr"
    if all((managed / name / ".reviewtrans-complete").is_file() for name in MODEL_NAMES):
        return managed
    bundled = exe_dir() / "asr-models"
    if getattr(sys, "frozen", False) and bundled.is_dir():
        return bundled
    return managed


def runtime_info() -> dict:
    python = runtime_python()
    root = python.parent.parent if python.parent.name in ("Scripts", "bin") else python.parent
    try:
        return json.loads((root / "runtime-info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def models_ready() -> bool:
    required = {"paraformer-zh": ("seg_dict", "tokens.json", "am.mvn"),
                "ct-punc": ("tokens.json",), "fsmn-vad": ("am.mvn",), "campplus": ()}
    return all((funasr_models_dir() / name / file).is_file()
               for name in MODEL_NAMES
               for file in (".reviewtrans-complete",
                            "campplus_cn_common.bin" if name == "campplus" else "model.pt",
                            "config.yaml", "configuration.json", *required[name]))


def worker_command() -> list[str]:
    python = runtime_python()
    if not python.is_file():
        raise ToolMissing("Chưa cài FunASR. Vào Tài nguyên → Paraformer-zh → Cài bộ nhận dạng + tải model.")
    return [str(python), "-u", str(resource_path("scripts/funasr_worker.py"))]


def _base_python() -> list[str]:
    if not getattr(sys, "frozen", False):
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        candidates = [[str(python)]]
    else:
        candidates = [["py", f"-3.{minor}"] for minor in (11, 12, 10)]
        candidates += [[path] for name in ("python", "python3") if (path := shutil.which(name))]
    for command in candidates:
        try:
            result = subprocess.run(
                [*command, "-c", "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) and sys.maxsize > 2**32 else 1)"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, **subprocess_kwargs(),
            )
            if result.returncode == 0:
                return command
        except (OSError, subprocess.TimeoutExpired):
            continue
    raise ToolMissing("Cần Python 3.10–3.12 (64-bit) để cài FunASR. Cài Python rồi thử lại trong Tài nguyên.")


def _worker(command: list[str], progress, stop=None, log=None, env=None) -> None:
    tail: list[str] = []

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-12]
        if line.startswith(EVENT_PREFIX):
            value = json.loads(line[len(EVENT_PREFIX):])
            progress(float(value["percent"]), value["message"])
            if log:
                log(value["message"])
        elif log:
            log(line)

    code = run(command, stop_event=stop, line_cb=on_line, quiet=True,
               env=env if env is not None else _install_environment())
    if code:
        raise RuntimeError("FunASR thất bại. " + " | ".join(line for line in tail if line))


def torch_flavor(device: str = "auto") -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cu126"
    if device != "auto":
        raise ValueError(f"Kiểu cài FunASR không hợp lệ: {device}")
    info = load_cached()
    if info and any(g.vendor == "nvidia" for g in info.gpus):
        try:
            if tuple(map(int, info.cuda_version.split(".")[:2])) >= (12, 6):
                return "cu126"
        except ValueError:
            pass
    return "cu126" if runtime_info().get("cuda_version") else "cpu"


def _install_environment() -> dict[str, str]:
    work = temp_dir() / "funasr-install"
    work.mkdir(parents=True, exist_ok=True)
    return dict(os.environ, TEMP=str(work), TMP=str(work), TMPDIR=str(work),
                PIP_CACHE_DIR=str(work / "pip-cache"), PIP_NO_CACHE_DIR="1",
                PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE="1", PYTHONUTF8="1",
                HF_HOME=str(user_data_dir() / "cache/huggingface"),
                MODELSCOPE_CACHE=str(user_data_dir() / "cache/modelscope"),
                TORCH_HOME=str(user_data_dir() / "cache/torch"),
                NUMBA_CACHE_DIR=str(user_data_dir() / "cache/numba"))


def check_install_space(flavor: str) -> None:
    # Includes unpacked libraries, the four models and transient wheel files.
    # Conservative headroom, not an exact download/installation size.
    required_gib = 16 if flavor == "cu126" else 8
    destination = user_data_dir()
    free = shutil.disk_usage(destination).free / (1024 ** 3)
    if free < required_gib:
        raise RuntimeError(
            f"Không đủ dung lượng tại {destination}: còn {free:.1f} GiB. "
            f"Cần tối thiểu {required_gib} GiB trống trước khi cài "
            f"Paraformer {'CUDA' if flavor == 'cu126' else 'CPU'} (gồm dung lượng tạm)."
        )


def _embedded_python(progress, stop, env) -> Path:
    """Windows embeddable Python can be moved with the app; venv cannot."""
    from .resources import download_file

    target = runtime_dir()
    python = target / "python.exe"
    if not python.is_file():
        progress(0, "Tải Python độc lập vào thư mục app…")
        archive = temp_dir() / "python-3.11.9-embed-amd64.zip"
        download_file("https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",
                      archive, lambda _pct, msg: progress(1, msg), stop, "Python portable")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for entry in zf.infolist():
                path = (target / entry.filename).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise RuntimeError("Đường dẫn trong gói Python không hợp lệ.")
            zf.extractall(target)
        archive.unlink(missing_ok=True)
        (target / "python311._pth").write_text(
            "python311.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8")
        # Retain existing packages, retire the old venv's absolute C: reference.
        config = target / "pyvenv.cfg"
        if config.is_file():
            config.replace(target / "pyvenv.cfg.legacy")
    if not (target / "Lib/site-packages/pip").is_dir():
        bootstrap = temp_dir() / "funasr-get-pip.py"
        download_file("https://bootstrap.pypa.io/get-pip.py", bootstrap,
                      lambda _pct, msg: progress(2, msg), stop, "pip")
        _worker([str(python), str(bootstrap), "--disable-pip-version-check"], progress, stop, env=env)
        bootstrap.unlink(missing_ok=True)
    return python


def install_funasr(progress=lambda *_: None, stop=None, device: str = "auto") -> Path:
    """Install a relocatable Windows runtime; auto selects compatible NVIDIA CUDA."""
    flavor = torch_flavor(device)
    check_install_space(flavor)
    env = _install_environment()
    python = runtime_dir() / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if IS_WINDOWS:
        python = _embedded_python(progress, stop, env)
    elif not python.is_file():
        progress(0, "Tạo môi trường Python riêng cho FunASR…")
        run_checked([*_base_python(), "-m", "venv", str(runtime_dir())], "Tạo môi trường FunASR", stop_event=stop, env=env)
    progress(2, "Cập nhật pip trong môi trường FunASR…")
    _worker([str(python), "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"], progress, stop, env=env)
    progress(5, f"Cài PyTorch {'CUDA 12.6 (NVIDIA)' if flavor == 'cu126' else 'CPU'} và torchaudio…")
    _worker(
        [str(python), "-m", "pip", "install", f"torch==2.8.0+{flavor}", f"torchaudio==2.8.0+{flavor}",
         "--index-url", f"https://download.pytorch.org/whl/{flavor}",
         "--extra-index-url", "https://pypi.org/simple", "--disable-pip-version-check"],
        progress, stop, env=env,
    )
    progress(35, "Cài thư viện FunASR…")
    _worker(
        [str(python), "-m", "pip", "install", "-r", str(resource_path("requirements-asr.txt")),
         "--disable-pip-version-check"],
        progress, stop, env=env,
    )
    # Check the actual imports, rather than treating the presence of python.exe as success.
    command = [str(python), "-u", str(resource_path("scripts/funasr_worker.py"))]
    _worker([*command, "--check", "--output", str(runtime_dir() / "runtime-info.json")],
            lambda _pct, msg: progress(50, msg), stop, env=env)
    info = json.loads((runtime_dir() / "runtime-info.json").read_text(encoding="utf-8"))
    if device == "cuda" and not info.get("cuda_available"):
        raise RuntimeError("Đã cài PyTorch CUDA nhưng không khởi tạo được GPU. Hãy kiểm tra driver NVIDIA.")
    _worker([*command, "--download", "--models", str(models_dir() / "funasr")],
            lambda pct, msg: progress(50 + pct * .5, msg), stop, env=env)
    return python


def segments_from_funasr(payload: dict, duration: float = 0, *, log=None) -> list[Segment]:
    """FunASR sentence timestamps are milliseconds on the original audio timeline."""
    result = payload.get("result", {})
    if not isinstance(result, dict):
        raise RuntimeError("FunASR trả về kết quả không hợp lệ.")
    sentences = result.get("sentence_info", [])
    if not isinstance(sentences, list) or (not sentences and str(result.get("text", "")).strip()):
        raise RuntimeError("FunASR thiếu timestamp từng câu; không thể tạo SRT. Hãy cài lại bộ nhận dạng trong Tài nguyên.")
    sentences = rebuild_sentences(result, log=log)
    segments = []
    overlaps = 0
    max_overlap = 0.0
    for sentence in sentences:
        text = str(sentence.get("text") or sentence.get("sentence") or "").strip()
        if not text:
            continue
        try:
            start, end = float(sentence["start"]) / 1000, float(sentence["end"]) / 1000
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("FunASR trả về câu thiếu timestamp hợp lệ.") from exc
        # The first utterance can straddle the audio origin (e.g. -90 ms).
        # Clip only that leading boundary; never shift later utterances.
        if not segments and math.isfinite(start) and math.isfinite(end) and start < 0 < end:
            if log:
                log(f"Căn đầu câu FunASR từ {start * 1000:.0f} ms về 0 ms; giữ nguyên nội dung.")
            start = 0.0
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise RuntimeError("FunASR trả về timestamp không hợp lệ.")
        if duration > 0:
            end = min(end, duration)
            if start >= end:
                raise RuntimeError("Timestamp FunASR vượt quá thời lượng video.")
        start, end = round(start, 3), round(end, 3)
        if end <= start:
            raise RuntimeError("FunASR trả về timestamp không hợp lệ ở độ chính xác mili giây.")
        if segments and start < segments[-1].end:
            previous = segments[-1]
            # Adjacent VAD regions can overlap even though both sentence bounds
            # move forward. Keep the next onset and trim only the previous tail;
            # never shift the rest of the episode or discard recognized words.
            if start <= previous.start or end <= previous.end:
                raise RuntimeError("Timestamp các câu FunASR bị chồng lấn hoặc sai thứ tự.")
            overlaps += 1
            max_overlap = max(max_overlap, previous.end - start)
            previous.end = start
        segments.append(Segment(id=len(segments) + 1, start=start, end=end, source=text))
    if overlaps and log:
        log(f"Đã căn lại {overlaps} ranh giới câu FunASR bị chồng lấn "
            f"(tối đa {max_overlap * 1000:.0f} ms); giữ nguyên nội dung và thời điểm bắt đầu câu.")
    return segments


def run_paraformer(ctx: RunContext, doc: VideoDoc, audio: Path) -> list[Segment]:
    ctx.check_stop()
    command = worker_command()
    if not models_ready():
        raise ToolMissing("Chưa tải đủ Paraformer-zh + FSMN-VAD + CT-Punc + CAM++. Vào Tài nguyên → Cài bộ nhận dạng + tải model.")
    cache = ctx.store.cache_dir(doc.id)
    output, srt = cache / "asr.paraformer.json", cache / "asr.srt"
    output.unlink(missing_ok=True)
    srt.unlink(missing_ok=True)
    ctx.log("Nhận dạng Paraformer-zh + FSMN-VAD + CT-Punc; căn ranh giới lời thoại bằng CAM++ và timestamp gốc.")
    _worker(
        [*command, "--input", str(audio), "--output", str(output),
         "--models", str(funasr_models_dir()), "--device", ctx.settings.asr_device,
         "--threads", str(asr_thread_count(ctx.settings.asr_threads))],
        ctx.progress, ctx.stop_event, ctx.log,
    )
    ctx.check_stop()
    payload = json.loads(output.read_text(encoding="utf-8"))
    segments = segments_from_funasr(payload, doc.duration, log=ctx.log)
    partial = srt.with_suffix(".part.srt")
    partial.write_text(render_srt((s.start, s.end, s.source) for s in segments), encoding="utf-8")
    partial.replace(srt)
    ctx.log(f"Paraformer nhận dạng {len(segments)} câu bằng {payload.get('device', '?')}. SRT: {srt}")
    ctx.progress(100, f"Paraformer: {len(segments)} câu")
    return segments
