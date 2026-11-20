from __future__ import annotations

import os
from pathlib import Path

import pytest

from reviewtrans.core import hardware
from reviewtrans.core.config import SETTINGS_VERSION, SettingsStore, upgrade_settings, asr_thread_count
from reviewtrans.core.hardware import EncoderConfig, Gpu, HardwareInfo, quality_args, resolve_encoder
from reviewtrans.core.paths import find_tool
from reviewtrans.core.pipeline import RunContext
from reviewtrans.core.pipeline import asr as asr_mod
from reviewtrans.core.pipeline import render as render_mod
from reviewtrans.core.proc import MediaInfo, probe, run_checked
from reviewtrans.core.store import ProjectStore

needs_ffmpeg = pytest.mark.skipif(not find_tool("ffmpeg"), reason="cần ffmpeg trong bin/")


@pytest.fixture(autouse=True)
def _fresh_hardware():
    hardware.forget()
    yield
    hardware.forget()


def test_quality_args_per_vendor():
    assert quality_args("libx264", 20, "fast") == ["-preset", "fast", "-crf", "20"]
    assert "-cq" in quality_args("h264_nvenc", 20) and "20" in quality_args("h264_nvenc", 20)
    assert quality_args("hevc_amf", 22)[-4:] == ["-qp_i", "22", "-qp_p", "22"]
    assert quality_args("h264_qsv", 24)[-2:] == ["-global_quality", "24"]
    mf = quality_args("h264_mf", 20)
    assert mf[:2] == ["-hw_encoding", "1"] and mf[-1] == "70"


def test_resolve_encoder_prefers_gpu_and_falls_back():
    amf = EncoderConfig("h264_amf", tail="format=nv12,hwupload", device="RX")
    info = HardwareInfo(gpus=[Gpu("RX", "amd", 1)], encoders={"h264_amf": amf, "h264_mf": EncoderConfig("h264_mf")},
                        failed={"h264_nvenc": "Cannot load nvcuda.dll"})
    assert resolve_encoder("auto", info)[0].codec == "h264_amf"
    assert resolve_encoder("auto_hevc", info)[0].codec == "libx265"  # không có hevc GPU → CPU
    config, note = resolve_encoder("h264_nvenc", info)
    assert config.codec == "libx264" and "nvcuda" in note
    assert resolve_encoder("libx265", info)[0].codec == "libx265"
    assert resolve_encoder("auto", info, {"h264_amf"})[0].codec == "h264_mf"  # vừa lỗi trong lần xuất này
    assert resolve_encoder("auto", info)[0].codec == "h264_amf"  # lần sau vẫn thử lại


def test_auto_prefers_discrete_gpu_to_integrated():
    info = HardwareInfo(encoders={
        "h264_amf": EncoderConfig("h264_amf", device="Radeon Graphics"),
        "h264_qsv": EncoderConfig("h264_qsv", device="Intel Arc", discrete=True),
    })
    assert resolve_encoder("auto", info)[0].codec == "h264_qsv"
    assert resolve_encoder("auto", info, {"h264_qsv"})[0].codec == "h264_amf"


def test_detect_nvenc_with_compatible_ffmpeg(tmp_path, monkeypatch):
    primary = tmp_path / "primary" / "ffmpeg.exe"
    alternate = tmp_path / "alternate" / "ffmpeg.exe"
    for exe in (primary, alternate):
        exe.parent.mkdir()
        exe.touch()
    monkeypatch.setattr(hardware, "IS_WINDOWS", True)
    monkeypatch.setenv("PATH", os.pathsep.join(str(exe.parent) for exe in (primary, alternate)))
    monkeypatch.setattr(hardware, "find_tool", lambda name: primary)
    monkeypatch.setattr(hardware, "cache_path", lambda: tmp_path / "hardware.json")
    monkeypatch.setattr(hardware, "list_gpus", lambda exe: [Gpu("Quadro T1000", "nvidia", 0)])
    monkeypatch.setattr(hardware, "nvidia_details", lambda: ("13.0", "581.32", "7.5", []))
    monkeypatch.setattr(hardware, "_encoder_list", lambda exe: {"h264_nvenc"})
    mismatch = hardware.NVENC_API_MISMATCH
    monkeypatch.setattr(hardware, "probe_encoder", lambda exe, config: (False, mismatch) if exe == primary else (True, ""))

    info = hardware.detect(force=True)
    encoder, _ = resolve_encoder("auto", info)
    assert (encoder.codec, encoder.device, encoder.ffmpeg) == ("h264_nvenc", "Quadro T1000", str(alternate))
    hardware.forget()
    assert hardware.load_cached().encoders[encoder.codec] == encoder
    alternate.unlink()
    hardware.forget()
    assert hardware.load_cached() is None


def test_amf_tries_discrete_gpu_first():
    gpus = [Gpu("AMD Radeon(TM) Graphics", "amd", 0, integrated=True), Gpu("Radeon RX 5300M", "amd", 1)]
    configs = hardware._candidates("h264_amf", gpus)
    if not configs:
        pytest.skip("AMF chỉ có trên Windows")
    assert [c.device for c in configs] == ["Radeon RX 5300M", "AMD Radeon(TM) Graphics"]
    assert "d3d11va=rtdx:1" in configs[0].pre_args
    assert hardware._candidates("h264_nvenc", gpus) == []  # không có card NVIDIA thì khỏi thử


def test_settings_upgrade_to_auto(home):
    store = SettingsStore()
    s = store.settings
    s.render.video_codec, s.asr_threads = "libx264", 4
    upgrade_settings(s, 1)
    assert s.render.video_codec == "auto" and s.asr_threads == 0 and s.settings_version == SETTINGS_VERSION
    s.render.video_codec = "libx264"  # người dùng tự chọn lại CPU → giữ nguyên
    upgrade_settings(s, 2)
    assert s.render.video_codec == "libx264"
    assert asr_thread_count(0) >= 1 and asr_thread_count(6) == 6


def _ctx(tmp_path):
    settings = SettingsStore().settings
    store, project = ProjectStore.create(tmp_path / "projects", "HW")
    logs: list[str] = []
    return RunContext(store=store, project=project, settings=settings, log=logs.append), logs


def _video(ctx, tmp_path):
    source = tmp_path / "src.mp4"
    run_checked(
        [str(find_tool("ffmpeg")), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        "tạo video test",
    )
    return ctx.store.add_video(ctx.project, str(source), probe(source))


@needs_ffmpeg
def test_render_gpu_failure_retries_on_cpu(tmp_path, home, monkeypatch):
    ctx, logs = _ctx(tmp_path)
    doc = _video(ctx, tmp_path)
    # cấu hình chắc chắn lỗi trên mọi máy (GPU số 99)
    broken = EncoderConfig("h264_nvenc", args=["-gpu", "99"], device="giả")
    real_choose = render_mod.choose_encoder
    calls = []

    def fake_choose(c, exclude=None):
        calls.append(set(exclude or ()))
        if len(calls) == 1:
            return broken, False
        c.settings.render.video_codec = "libx264"
        return real_choose(c, exclude)

    monkeypatch.setattr(render_mod, "choose_encoder", fake_choose)
    out = render_mod.run_render(ctx, doc, [], None)
    info = probe(out)
    assert info.has_video and abs(info.duration - 3) < 0.5
    assert any("xuất lại bằng libx264" in line for line in logs), logs[-10:]
    assert "h264_nvenc" in calls[-1]


@needs_ffmpeg
def test_render_failure_preserves_previous_output(tmp_path, home, monkeypatch):
    ctx, _ = _ctx(tmp_path)
    ctx.settings.render.video_codec = "libx264"
    doc = _video(ctx, tmp_path)
    sound = tmp_path / "mix.wav"
    sound.touch()
    out = tmp_path / "out.mp4"
    out.write_bytes(b"previous output")

    def fail(command, *, line_cb, **kwargs):
        Path(command[-1]).touch()
        line_cb("[decoder] Invalid video source")
        return 1

    monkeypatch.setattr(render_mod, "run", fail)
    with pytest.raises(RuntimeError, match="Invalid video source"):
        render_mod.run_render(ctx, doc, [], sound, output=out)
    assert out.read_bytes() == b"previous output"
    assert not list(out.parent.glob(".render-*"))


@needs_ffmpeg
def test_render_with_detected_gpu(tmp_path, home):
    info = hardware.detect(force=True)
    if not any(c.startswith("h264") for c in info.encoders):
        pytest.skip("máy này không có bộ mã hoá H.264 phần cứng")
    ctx, logs = _ctx(tmp_path)
    ctx.settings.render.video_codec = "auto"
    doc = _video(ctx, tmp_path)
    out = render_mod.run_render(ctx, doc, [], None)
    result = probe(out)
    assert result.has_video and abs(result.duration - 3) < 0.5
    assert any("Mã hoá bằng GPU" in line for line in logs), logs[-10:]
    assert not any("xuất lại" in line for line in logs)
