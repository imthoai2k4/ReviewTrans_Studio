from __future__ import annotations

import json
import re
import sys
import threading
import time
import wave
from array import array
from pathlib import Path

import pytest
from PyQt6 import QtCore

from reviewtrans.core.config import ProviderProfile, SettingsStore
from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import RunContext, asr, resources, translate, tts
from reviewtrans.core.pipeline.runner import VideoPipeline
from reviewtrans.core.proc import MediaInfo, StopRequested, run
from reviewtrans.core.providers.translate.base import LLMTranslator
from reviewtrans.core.store import ProjectStore
from reviewtrans.ui.jobs import DONE, PAUSED, PENDING
from reviewtrans.ui.pages.queue import QueuePage
from reviewtrans.ui.state import AppState


def wait_until(qapp, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(10, loop.quit)
        loop.exec()
    qapp.processEvents()
    assert condition(), "Timed out waiting for queue"


@pytest.mark.parametrize("quick_resume", [False, True])
def test_stop_and_resume_keep_current_video_and_queue(qapp, home, tmp_path, monkeypatch, quick_resume):
    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Queue")
    docs = [state.store.add_video(state.project, f"episode-{i}.mp4") for i in range(3)]
    for doc in docs:
        doc.duration, doc.width, doc.height = 2, 320, 240
        state.store.save_video(doc)
    page = QueuePage(state)
    calls, entered = [], threading.Event()
    interrupted = False

    def step(pipeline, name, only_missing):
        nonlocal interrupted
        calls.append((pipeline.video_id, name))
        if pipeline.video_id == docs[0].id and name == "translate" and not interrupted:
            interrupted = True
            entered.set()
            pipeline.ctx.stop_event.wait(3)
            pipeline.ctx.check_stop()

    monkeypatch.setattr(VideoPipeline, "_run_step", step)
    queue = state.jobs
    try:
        assert not page.stop_button.isEnabled() and not page.resume_button.isEnabled()
        jobs = state.enqueue([d.id for d in docs[:2]], ["asr", "translate", "tts"])
        wait_until(qapp, entered.is_set)
        page.stop_button.click()
        assert queue.is_paused() and page.resume_button.isEnabled()
        assert not page.stop_button.isEnabled()
        if quick_resume:
            page.resume_button.click()
        else:
            wait_until(qapp, lambda: not queue.is_busy())
            assert jobs[0].state == PAUSED and jobs[1].state == PENDING
            assert calls == [(docs[0].id, "asr"), (docs[0].id, "translate")]
            assert state.is_video_locked(docs[0].id)
            elapsed = jobs[0].elapsed()
            added = state.enqueue([docs[2].id], ["tts"])[0]
            assert added.state == PENDING
            page._clear()
            assert len(queue.jobs) == page.table.rowCount() == 3
            assert jobs[0].elapsed() == elapsed
            page.resume_button.click()
        wait_until(qapp, lambda: all(j.state == DONE for j in queue.jobs) and not queue.is_busy())
        assert calls[:4] == [
            (docs[0].id, "asr"), (docs[0].id, "translate"),
            (docs[0].id, "translate"), (docs[0].id, "tts"),
        ]
        assert calls[4:7] == [(docs[1].id, s) for s in ("asr", "translate", "tts")]
        assert not page.stop_button.isEnabled() and not page.resume_button.isEnabled()
        assert not queue.locked_videos()
    finally:
        queue.shutdown()
        qapp.processEvents()
        page.close()


def make_run(home, tmp_path):
    settings = SettingsStore().settings
    profile = ProviderProfile(name="Fake", kind="openai", api_keys=["test"])
    settings.translate_profiles = [profile]
    settings.default_translate_profile = profile.id
    settings.translate_batch_size = 1
    store, project = ProjectStore.create(tmp_path, "Resume")
    project.translate_profile = profile.id
    doc = store.add_video(project, "video.mp4")
    doc.duration, doc.width, doc.height = 2, 320, 240
    store.save_video(doc)
    segments = [Segment(id=i, start=i, end=i + 1, source=f"source-{i}", text=f"old-{i}") for i in (1, 2, 3)]
    store.save_segments(doc.id, segments)
    return RunContext(store=store, project=project, settings=settings), doc, profile


def test_stop_during_translation_ignores_late_api_and_resumes_unfinished_batches(home, tmp_path, monkeypatch):
    ctx, doc, profile = make_run(home, tmp_path)
    release, returned = threading.Event(), threading.Event()
    calls = []

    class SlowLLM(LLMTranslator):
        def _call(self, system, prompt, key):
            item = json.loads(re.search(r"Items \(JSON\):\n(.*)\n\nReturn", prompt, re.S).group(1))[0]
            calls.append(item["text"])
            if len(calls) == 2:
                ctx.stop_event.set()
                release.wait(3)
                returned.set()
            return json.dumps([{"id": 1, "text": "new-" + item["text"]}])

    monkeypatch.setattr(translate, "create_translator", lambda *args: SlowLLM(profile))
    pipeline = VideoPipeline(ctx, doc.id)
    started = time.monotonic()
    try:
        with pytest.raises(StopRequested):
            pipeline.run(["translate"], only_missing=False)
        assert time.monotonic() - started < 1
        assert pipeline.checkpoint.translated_ids == {1}
        saved = ctx.store.load_segments(doc.id)
        assert [s.text for s in saved] == ["new-source-1", "old-2", "old-3"]
        assert ctx.store.load_video(doc.id).stage("translate") == "stale"
    finally:
        release.set()
        assert returned.wait(1)
    assert ctx.store.load_segments(doc.id)[1].text == "old-2"
    ctx.stop_event = threading.Event()
    resumed = VideoPipeline(ctx, doc.id)
    resumed.checkpoint = pipeline.checkpoint
    resumed.run(["translate"], only_missing=False)
    assert calls == ["source-1", "source-2", "source-2", "source-3"]
    assert [s.text for s in ctx.store.load_segments(doc.id)] == [f"new-source-{i}" for i in (1, 2, 3)]


def test_stop_during_tts_saves_partial_results_and_resumes_force_mode(home, tmp_path, monkeypatch):
    ctx, doc, _profile = make_run(home, tmp_path)
    profile = ProviderProfile(name="TTS", kind="openai_speech", options={"concurrency": 1})
    ctx.settings.tts_profiles = [profile]
    ctx.settings.default_tts_profile = profile.id
    release, discarded = threading.Event(), threading.Event()
    calls = []

    class Provider:
        voice = "voice"

        def synthesize(self, text, voice, speed, base, stop):
            calls.append(text)
            if len(calls) == 2:
                # Đợi câu đầu được ghi vào danh sách kết quả trước khi bấm dừng.
                deadline = time.monotonic() + 2
                while not pipeline.checkpoint.tts_ids and time.monotonic() < deadline:
                    time.sleep(0.005)
                ctx.stop_event.set()
                release.wait(3)
            path = base.with_suffix(".wav")
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(array("h", [5000, -5000] * 4000).tobytes())
            return path

    monkeypatch.setattr(tts, "create_tts", lambda *args: Provider())
    monkeypatch.setattr(tts, "_convert_to_wav", lambda raw, target, ctx: target.write_bytes(raw.read_bytes()))
    monkeypatch.setattr(tts, "probe", lambda path: MediaInfo(duration=1, has_audio=True))
    unlink = Path.unlink

    def remove(path, *args, **kwargs):
        result = unlink(path, *args, **kwargs)
        if release.is_set() and path.name.startswith("raw_2_"):
            discarded.set()
        return result

    monkeypatch.setattr(Path, "unlink", remove)
    pipeline = VideoPipeline(ctx, doc.id)
    started = time.monotonic()
    try:
        with pytest.raises(StopRequested):
            pipeline.run(["tts"], only_missing=False)
        assert time.monotonic() - started < 1
        saved = ctx.store.load_segments(doc.id)
        assert saved[0].tts_file and not saved[1].tts_file
        assert pipeline.checkpoint.tts_ids == {1}
        first_file = saved[0].tts_file
    finally:
        release.set()
        assert discarded.wait(1)
    ctx.stop_event = threading.Event()
    resumed = VideoPipeline(ctx, doc.id)
    resumed.checkpoint = pipeline.checkpoint
    resumed.run(["tts"], only_missing=False)
    saved = ctx.store.load_segments(doc.id)
    assert saved[0].tts_file == first_file
    assert all(s.tts_file for s in saved)
    assert calls == ["old-1.", "old-2.", "old-2.", "old-3."]


def test_stop_terminates_silent_subprocess(tmp_path):
    stop = threading.Event()
    ready = tmp_path / "ready"

    def cancel():
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()

    watcher = threading.Thread(target=cancel)
    watcher.start()
    started = time.monotonic()
    try:
        with pytest.raises(StopRequested):
            run([sys.executable, "-c", "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(30)", str(ready)], stop_event=stop)
        assert time.monotonic() - started < 3
    finally:
        stop.set()
        watcher.join(3)


def test_interrupted_audio_extraction_is_not_reused(home, tmp_path, monkeypatch):
    ctx, doc, _profile = make_run(home, tmp_path)
    source = tmp_path / "source.mp4"
    source.touch()
    doc.source_path = str(source)
    monkeypatch.setattr(asr, "require_tool", lambda name: Path(name))
    calls = []

    def extract(command, *args, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"partial" if len(calls) == 1 else b"complete")
        if len(calls) == 1:
            raise StopRequested()

    monkeypatch.setattr(asr, "run_checked", extract)
    with pytest.raises(StopRequested):
        asr.extract_audio(ctx, doc)
    target = asr.extract_audio(ctx, doc)
    assert len(calls) == 2 and target.read_bytes() == b"complete"


@pytest.mark.parametrize("during_stream", [False, True])
def test_model_download_can_stop_without_late_file_writes(tmp_path, monkeypatch, during_stream):
    stop, release, returned, closed = (threading.Event() for _ in range(4))
    target = tmp_path / "model.bin"

    class Response:
        headers = {"content-length": "8"}

        def raise_for_status(self):
            pass

        def close(self):
            closed.set()

        def iter_content(self, chunk_size):
            yield b"first"
            stop.set()
            release.wait(3)
            returned.set()
            yield b"end"

    def get(*args, **kwargs):
        if not during_stream:
            stop.set()
            release.wait(3)
            returned.set()
        return Response()

    monkeypatch.setattr(resources.requests, "get", get)
    started = time.monotonic()
    try:
        with pytest.raises(StopRequested):
            resources.download_file("https://example.invalid/model", target, stop=stop)
        assert time.monotonic() - started < 1
        assert not target.exists()
        partial = target.with_suffix(".bin.part")
        before = partial.read_bytes() if partial.exists() else None
    finally:
        release.set()
        assert returned.wait(1) and closed.wait(1)
    assert not target.exists()
    assert (partial.read_bytes() if partial.exists() else None) == before
