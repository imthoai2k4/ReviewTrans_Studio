from __future__ import annotations

import re
import subprocess
import math
from array import array
from pathlib import Path

import pytest

from reviewtrans.core.config import AppSettings
from reviewtrans.core.models import BatchSettings, Segment
from reviewtrans.core.paths import find_tool
from reviewtrans.core.pipeline import RunContext
from reviewtrans.core.pipeline.render import build_render_plan, run_render
from reviewtrans.core.proc import probe, run_checked
from reviewtrans.core.store import ProjectStore


def _mean_volume(path: Path, ffmpeg: Path) -> float:
    result = subprocess.run([
        str(ffmpeg), "-v", "info", "-i", str(path), "-af", "volumedetect", "-f", "null", "-",
    ], capture_output=True, text=True, check=True)
    match = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", result.stderr)
    assert match
    return float(match.group(1))


@pytest.fixture
def audio_render(home, tmp_path):
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        pytest.skip("cần ffmpeg")
    source = tmp_path / "source.mp4"
    run_checked([
        str(ffmpeg), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:size=160x120:rate=25:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264",
        "-c:a", "aac", "-ac", "2", "-shortest", str(source),
    ], "tạo video test")
    store, project = ProjectStore.create(tmp_path, "Render audio")
    doc = store.add_video(project, str(source), probe(source))
    settings = AppSettings()
    settings.render.video_codec = "libx264"
    settings.render.hwaccel_decode = False
    ctx = RunContext(store=store, project=project, settings=settings)
    return ctx, doc, source, ffmpeg


@pytest.mark.parametrize("missing_mix", [False, True])
def test_render_without_mix_respects_ten_percent_gain(audio_render, missing_mix):
    ctx, doc, source, ffmpeg = audio_render
    doc.apply_batch_settings(BatchSettings(original_volume=10))
    mix = ctx.store.cache_dir(doc.id) / "missing.wav" if missing_mix else None
    output = run_render(ctx, doc, [], mix)
    change = _mean_volume(output, ffmpeg) - _mean_volume(source, ffmpeg)
    assert change == pytest.approx(-40, abs=0.7)


@pytest.mark.parametrize("mode", ["keep", "mute"])
def test_render_plan_without_mix_respects_original_audio_settings(audio_render, mode):
    ctx, doc, source, ffmpeg = audio_render
    doc.audio.original_volume = 10
    doc.audio.original_mode = mode
    output = ctx.store.output_dir(ctx.project) / "fallback.mp4"
    plan = build_render_plan(doc, [], ctx.settings.render, ctx.store.cache_dir(doc.id), output, None)
    run_checked(plan.command, "xuất video test", cwd=plan.cwd)
    if mode == "mute":
        assert not probe(output).has_audio
    else:
        assert _mean_volume(output, ffmpeg) - _mean_volume(source, ffmpeg) == pytest.approx(-40, abs=0.7)


def _tone_amplitude(path: Path, ffmpeg: Path, frequency: int) -> float:
    result = subprocess.run([
        str(ffmpeg), "-v", "error", "-ss", "0.5", "-i", str(path), "-t", "0.4",
        "-ac", "1", "-ar", "48000", "-f", "f32le", "-",
    ], capture_output=True, check=True)
    samples = array("f")
    samples.frombytes(result.stdout)
    sine = sum(value * math.sin(2 * math.pi * frequency * i / 48000) for i, value in enumerate(samples))
    cosine = sum(value * math.cos(2 * math.pi * frequency * i / 48000) for i, value in enumerate(samples))
    return 2 * math.hypot(sine, cosine) / len(samples)


def test_missing_mix_rebuilds_audio_and_keeps_dub_volume_independent(audio_render):
    ctx, doc, source, ffmpeg = audio_render
    speech = ctx.store.tts_dir(doc.id) / "speech.wav"
    run_checked([
        str(ffmpeg), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=880:duration=1.5",
        "-ac", "2", str(speech),
    ], "tạo lồng tiếng test")
    segments = [Segment(id=1, start=0.25, end=1.75, tts_file=speech.name, tts_duration=1.5)]
    outputs = []
    for volume in (100, 10):
        doc.audio.original_volume = volume
        output = ctx.store.output_dir(ctx.project) / f"volume_{volume}.mp4"
        outputs.append(run_render(ctx, doc, segments, None, output=output))
    full, reduced = outputs
    assert _tone_amplitude(reduced, ffmpeg, 440) / _tone_amplitude(full, ffmpeg, 440) == pytest.approx(0.01, abs=0.002)
    assert _tone_amplitude(full, ffmpeg, 880) > 0.01
    assert _tone_amplitude(reduced, ffmpeg, 880) / _tone_amplitude(full, ffmpeg, 880) == pytest.approx(1, abs=0.03)
