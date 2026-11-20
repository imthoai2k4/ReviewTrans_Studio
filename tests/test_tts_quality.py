from __future__ import annotations

import math
import os
import shutil
import subprocess
import wave
from array import array
from pathlib import Path

import pytest

from reviewtrans.core.audio_quality import audio_quality_error, inspect_wav
from reviewtrans.core.config import AppSettings, ProviderProfile
from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import RunContext, tts
from reviewtrans.core.proc import MediaInfo
from reviewtrans.core.paths import find_tool
from reviewtrans.core.pipeline.mix import plan_clips, run_mix
from reviewtrans.core.pipeline.runner import VideoPipeline
from reviewtrans.core.store import ProjectStore


def wav(path: Path, seconds: float, *, sound: float | None = None, amplitude=6000, rate=8000):
    active = seconds if sound is None else sound
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(array("h", (
            int(amplitude * math.sin(2 * math.pi * 440 * i / rate)) if i < active * rate else 0
            for i in range(round(seconds * rate))
        )).tobytes())


@pytest.mark.parametrize("text,seconds,sound,amplitude", [
    ("Hương Nhi", 0.31, 0.02, 700),
    ("Ta mở đường", 0.36, 0.06, 2400),
    ("Không được dùng sức", 0.6, 0.2, 16000),
    ("Một câu dài mà chỉ đọc được một tiếng", 0.4, 0.4, 6000),
    ("Xin chào", 2, 2, 10),
    ("Xin chào", 2, 0, 6000),
])
def test_short_or_nearly_silent_responses_are_rejected(tmp_path, text, seconds, sound, amplitude):
    path = tmp_path / "speech.wav"
    wav(path, seconds, sound=sound, amplitude=amplitude)
    assert audio_quality_error(path, text)


def test_short_word_and_natural_pauses_are_allowed_and_cache_tracks_replacement(tmp_path):
    path = tmp_path / "speech.wav"
    wav(path, 0.42, sound=0.1)
    assert audio_quality_error(path, "Được") is None
    wav(path, 1, sound=0.6)
    assert audio_quality_error(path, "Không được dùng sức") is None
    stamp = path.stat()
    wav(path, 1, sound=0)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert audio_quality_error(path, "Không được dùng sức")


def test_quality_accounts_for_tts_speed_and_cjk(tmp_path):
    path = tmp_path / "speech.wav"
    wav(path, 0.4)
    assert audio_quality_error(path, "một hai ba bốn năm", speed=1)
    assert audio_quality_error(path, "một hai ba bốn năm", speed=2) is None
    assert audio_quality_error(path, "一二三四五", language="zh")
    assert inspect_wav(path).duration == pytest.approx(0.4)


@pytest.fixture
def synthesis(tmp_path, monkeypatch):
    store, project = ProjectStore.create(tmp_path, "Quality")
    doc = store.add_video(project, "video.mp4")
    profile = ProviderProfile(kind="openai_speech", options={"voice": "sample"})
    logs = []
    ctx = RunContext(store, project, AppSettings(tts_profiles=[profile], default_tts_profile=profile.id), log=logs.append)
    calls = []
    responses = []

    class Provider:
        voice = "sample"

        def synthesize(self, text, voice, speed, base, stop):
            calls.append(text)
            path = base.with_suffix(".wav")
            seconds, sound = responses[min(len(calls) - 1, len(responses) - 1)]
            wav(path, seconds, sound=sound)
            return path

    monkeypatch.setattr(tts, "create_tts", lambda *args: Provider())
    monkeypatch.setattr(tts, "_convert_to_wav", lambda raw, target, ctx: shutil.copyfile(raw, target))
    monkeypatch.setattr(tts, "probe", lambda path: MediaInfo(duration=inspect_wav(path).duration, has_audio=True))
    return ctx, doc, profile, responses, calls, logs


@pytest.mark.parametrize(("text", "spoken"), [
    ("Ma đạo Tây Hoàng thật lớn trận thế", "Ma đạo Tây Hoàng thật lớn trận thế."),
    ("  Ngươi đoán đúng rồi  ", "Ngươi đoán đúng rồi."),
    ("Ngươi tìm ta?", "Ngươi tìm ta?"),
    ("Không được!", "Không được!"),
    ("Nhưng mà…", "Nhưng mà…"),
])
def test_tts_finishes_sentence_on_first_attempt(synthesis, text, spoken):
    ctx, doc, _profile, responses, calls, _logs = synthesis
    responses.append((3, 2.7))
    seg = Segment(id=16, text=text)
    assert tts.run_tts(ctx, doc, [seg]) == (1, 0)
    assert calls == [spoken]
    assert seg.text == text
    # Kết quả tạo với dấu kết câu có thể dùng lại khi chạy phần còn thiếu.
    assert tts.run_tts(ctx, doc, [seg]) == (0, 0)
    assert calls == [spoken]


def test_invalid_tts_is_retried_before_publishing(synthesis):
    ctx, doc, _profile, responses, calls, logs = synthesis
    responses.extend([(0.6, 0.2), (1.4, 1.2)])
    seg = Segment(id=8, text="Không được dùng sức")
    assert tts.run_tts(ctx, doc, [seg]) == (1, 0)
    assert len(calls) == 2 and seg.tts_duration == pytest.approx(1.4)
    assert tts.tts_path(ctx, doc, seg) is not None
    assert any("tạo lại" in line for line in logs)
    assert len(list(ctx.store.tts_dir(doc.id).iterdir())) == 1


def test_old_unpunctuated_cache_is_regenerated_in_only_missing_mode(synthesis, monkeypatch):
    ctx, doc, profile, responses, calls, _logs = synthesis
    seg = Segment(id=16, text="Ngươi đoán đúng rồi", tts_file="old.wav", tts_duration=1.2)
    with monkeypatch.context() as legacy:
        legacy.setattr(tts, "synthesis_text", lambda text: text.strip())
        seg.tts_key = tts.segment_key(seg, profile, "sample", doc.audio.speed)
    wav(ctx.store.tts_dir(doc.id) / seg.tts_file, 1.2)
    responses.append((1.4, 1.2))
    assert tts.run_tts(ctx, doc, [seg]) == (1, 0)
    assert calls == ["Ngươi đoán đúng rồi."]
    assert seg.tts_file != "old.wav"


def test_existing_bad_audio_is_regenerated_in_only_missing_mode(synthesis):
    ctx, doc, profile, responses, calls, _logs = synthesis
    responses.append((1.2, 1))
    seg = Segment(id=1, text="Ta mở đường", tts_file="old.wav", tts_duration=0.36)
    seg.tts_key = tts.segment_key(seg, profile, "sample", doc.audio.speed)
    wav(ctx.store.tts_dir(doc.id) / seg.tts_file, 0.36, sound=0.06)
    assert tts.run_tts(ctx, doc, [seg]) == (1, 0)
    assert len(calls) == 1 and seg.tts_file != "old.wav"
    assert not (ctx.store.tts_dir(doc.id) / "old.wav").exists()


def test_all_bad_attempts_keep_previous_good_audio_and_report_failure(synthesis):
    ctx, doc, _profile, responses, calls, _logs = synthesis
    responses.append((0.36, 0.06))
    seg = Segment(id=1, text="Ta mở đường", tts_file="old.wav", tts_duration=1.2, tts_key="old-key")
    old = ctx.store.tts_dir(doc.id) / seg.tts_file
    wav(old, 1.2)
    original = old.read_bytes()
    assert tts.run_tts(ctx, doc, [seg], force=True) == (0, 1)
    assert len(calls) == 3 and calls[-1] == "Ta mở đường."
    assert (seg.tts_file, seg.tts_duration, seg.tts_key) == ("old.wav", 1.2, "old-key")
    assert old.read_bytes() == original
    assert list(ctx.store.tts_dir(doc.id).iterdir()) == [old]


def test_mix_preserves_voice_at_video_end(synthesis):
    ctx, doc, _profile, _responses, _calls, logs = synthesis
    if not find_tool("ffmpeg"):
        pytest.skip("cần ffmpeg")
    doc.has_audio = False
    doc.duration = 2
    path = ctx.store.tts_dir(doc.id) / "voice.wav"
    wav(path, 1)
    segments = [
        Segment(id=1, start=0, end=0.3, tts_file=path.name, tts_duration=1),
        Segment(id=2, start=0.3, end=1.8, tts_file=path.name, tts_duration=1),
    ]
    assert run_mix(ctx, doc, segments).exists()
    assert not any("chồng" in line for line in logs)
    mix = run_mix(ctx, doc, [Segment(id=3, start=1.9, end=1.99, tts_file=path.name, tts_duration=1)])
    with wave.open(str(mix)) as audio:
        assert audio.getnframes() / audio.getframerate() > 2.4
        audio.setpos(round(2.2 * audio.getframerate()))
        assert any(audio.readframes(1000))


def test_render_continues_after_one_bad_sentence_and_retry_only_fills_gap(synthesis, qapp, tmp_path):
    ctx, doc, profile, responses, calls, logs = synthesis
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        pytest.skip("cần ffmpeg")
    source = tmp_path / "sample.mp4"
    subprocess.run([str(ffmpeg), "-v", "error", "-f", "lavfi", "-i",
                    "color=size=160x120:rate=12:duration=4", "-c:v", "libx264", str(source)], check=True)
    doc.source_path = str(source)
    ctx.store.save_video(doc)
    profile.options["concurrency"] = 1
    ctx.settings.render.video_codec = "libx264"
    ctx.settings.render.hwaccel_decode = False
    segments = [Segment(id=i, start=i-1, end=i, text=f"Câu {i}") for i in (1, 2, 3)]
    segments[1].tts_file, segments[1].tts_duration = "obsolete.wav", 1
    ctx.store.save_segments(doc.id, segments)
    responses.extend([(1, 1), (0.1, 0), (0.1, 0), (0.1, 0), (1, 1)])
    pipeline = VideoPipeline(ctx, doc.id)
    pipeline.run(["tts", "render"])
    saved = ctx.store.load_segments(doc.id)
    assert calls == ["Câu 1.", "Câu 2.", "Câu 2.", "Câu 2.", "Câu 3."]
    assert saved[1].tts_error and saved[1].text == "Câu 2"
    assert [clip.segment.id for clip in plan_clips(doc, saved)] == [1, 3]
    assert Path(pipeline.doc.last_output).stat().st_size > 0
    assert any("Bỏ qua lồng tiếng câu #2" in msg for msg in logs)
    VideoPipeline(ctx, doc.id).run(["tts"])
    assert calls[5:] == ["Câu 2."]
    assert not ctx.store.load_segments(doc.id)[1].tts_error


def test_fit_delays_next_voice_then_recovers_original_timing(synthesis):
    _, doc, *_ = synthesis
    doc.duration = 4
    doc.audio.max_speed = 1.8
    segments = [
        Segment(id=1, start=0, end=.4, tts_file="a", tts_duration=1.8),
        Segment(id=2, start=.4, end=1.5, tts_file="b", tts_duration=.9),
        Segment(id=3, start=1.5, end=3, tts_file="c", tts_duration=.6),
    ]
    clips = plan_clips(doc, segments)
    assert [c.start for c in clips] == pytest.approx([0, 1, 1.5])
    assert [c.tempo for c in clips] == pytest.approx([1.8, 1.8, 1])
    doc.audio.fit_mode = "natural"
    assert [c.start for c in plan_clips(doc, segments)] == [s.start for s in segments]
