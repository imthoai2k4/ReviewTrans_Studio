from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reviewtrans.core.config import ProviderProfile, SettingsStore
from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import RunContext
from reviewtrans.core.pipeline import runner as runner_module
from reviewtrans.core.pipeline import translate as translate_module
from reviewtrans.core.pipeline.runner import VideoPipeline
from reviewtrans.core.proc import StopRequested
from reviewtrans.core.providers import ProviderError
from reviewtrans.core.providers.translate import base as base_module
from reviewtrans.core.providers.translate.base import LLMTranslator, TranslateJob
from reviewtrans.core.store import ProjectStore


@pytest.fixture
def context_run(home, tmp_path):
    settings = SettingsStore().settings
    profile = ProviderProfile(name="Context API", kind="openai", api_keys=["test"])
    settings.translate_profiles = [profile]
    store, project = ProjectStore.create(tmp_path, "Context")
    project.translate_profile = profile.id
    doc = store.add_video(project, "episode.mp4")
    doc.duration, doc.width, doc.height = 2, 320, 240
    doc.set_stage("translate", "done")
    store.save_video(doc)
    store.save_segments(doc.id, [Segment(id=1, start=0, end=1, source="source", text="translated")])
    logs, progress = [], []
    ctx = RunContext(
        store=store, project=project, settings=settings,
        log=logs.append, progress=lambda pct, msg: progress.append((pct, msg)),
    )
    return ctx, doc, profile, logs, progress


def test_context_timeout_keeps_context_and_continues_tts(context_run, monkeypatch):
    ctx, doc, profile, logs, progress = context_run
    release, returned = threading.Event(), threading.Event()

    class SlowLLM(LLMTranslator):
        def _call(self, *args):
            release.wait(3)
            returned.set()
            return json.dumps({"summary": "late response"})

    monkeypatch.setattr(translate_module, "create_translator", lambda *args: SlowLLM(profile))
    monkeypatch.setattr(translate_module, "CONTEXT_UPDATE_TIMEOUT", 0.08)
    monkeypatch.setattr(translate_module, "CONTEXT_PROGRESS_INTERVAL", 0.01)
    monkeypatch.setattr(translate_module, "CONTEXT_LOG_INTERVAL", 0.01)
    tts_calls = []
    monkeypatch.setattr(runner_module, "run_tts", lambda *args, **kw: (tts_calls.append(True) or 1, 0))
    started = time.monotonic()
    try:
        VideoPipeline(ctx, doc.id).run(["context", "tts"])
        assert time.monotonic() - started < 1
        assert tts_calls == [True]
        saved = ctx.store.load_video(doc.id)
        assert saved.stage("tts") == "done" and saved.stage("translate") == "done"
        assert ctx.store.load_segments(doc.id)[0].text == "translated"
        assert any("Giữ ngữ cảnh cũ" in line for line in logs)
        assert any("Hoàn tất Lồng tiếng sau" in line for line in logs)
        assert not ctx.store.load_context().processed_videos
    finally:
        release.set()
        assert returned.wait(1)
    # API về muộn cũng không được ghi đè ngữ cảnh.
    assert ctx.store.load_context().summary == ""


def test_context_wait_reports_progress(context_run, monkeypatch):
    ctx, doc, profile, logs, progress = context_run

    class SlowLLM(LLMTranslator):
        def _call(self, *args):
            time.sleep(0.35)
            return '{"summary": "updated"}'

    monkeypatch.setattr(translate_module, "create_translator", lambda *args: SlowLLM(profile))
    monkeypatch.setattr(translate_module, "CONTEXT_PROGRESS_INTERVAL", 0.01)
    monkeypatch.setattr(translate_module, "CONTEXT_LOG_INTERVAL", 0.01)
    translate_module.update_context(ctx, doc, ctx.store.load_segments(doc.id))
    assert any("chờ API" in message for _pct, message in progress)
    assert any("chờ API" in line for line in logs)
    assert ctx.store.load_context().summary == "updated"


def test_stopping_during_context_does_not_start_tts(context_run, monkeypatch):
    ctx, doc, profile, logs, progress = context_run
    release = threading.Event()

    class SlowLLM(LLMTranslator):
        def _call(self, *args):
            release.wait(3)
            return '{"summary": "cancelled"}'

    monkeypatch.setattr(translate_module, "create_translator", lambda *args: SlowLLM(profile))
    tts = Mock(return_value=(1, 0))
    monkeypatch.setattr(runner_module, "run_tts", tts)
    timer = threading.Timer(0.03, ctx.stop_event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(StopRequested):
            VideoPipeline(ctx, doc.id).run(["context", "tts"])
        assert time.monotonic() - started < 1
        tts.assert_not_called()
        assert ctx.store.load_context().summary == ""
    finally:
        release.set()
        timer.cancel()
        timer.join()


@pytest.mark.parametrize("force", [False, True])
def test_explicit_context_failure_remains_an_error(context_run, monkeypatch, force):
    ctx, doc, profile, logs, progress = context_run

    class FailingLLM(LLMTranslator):
        def _call(self, *args):
            raise ProviderError("API failed")

    monkeypatch.setattr(translate_module, "create_translator", lambda *args: FailingLLM(profile))
    monkeypatch.setattr(translate_module, "CONTEXT_ATTEMPTS", 1)
    tts = Mock(return_value=(1, 0))
    monkeypatch.setattr(runner_module, "run_tts", tts)
    with pytest.raises(ProviderError, match="API failed"):
        VideoPipeline(ctx, doc.id).run(["context", "tts"] if force else ["context"], force_context=force)
    tts.assert_not_called()


def test_completion_logs_retries_and_stops_after_final_attempt(monkeypatch):
    provider = LLMTranslator(ProviderProfile(name="Test"))
    logs = []
    call = Mock(side_effect=[TimeoutError("slow"), "ok"])
    monkeypatch.setattr(provider, "_call", call)
    sleep = Mock()
    monkeypatch.setattr(base_module.time, "sleep", sleep)
    assert provider.complete("system", "prompt", job=TranslateJob(log=logs.append), attempts=2) == "ok"
    assert call.call_count == 2 and sleep.call_count == 1
    assert "lần 2/2" in logs[0]
    call.reset_mock(side_effect=True)
    call.side_effect = TimeoutError("slow")
    sleep.reset_mock()
    logs.clear()
    with pytest.raises(ProviderError):
        provider.complete("system", "prompt", job=TranslateJob(log=logs.append), attempts=2)
    assert call.call_count == 2 and sleep.call_count == 1
    assert "dừng thử" in logs[-1]


def test_retry_backoff_can_be_cancelled():
    stop = threading.Event()
    provider = LLMTranslator(ProviderProfile(name="Test"))
    provider._call = Mock(side_effect=TimeoutError("slow"))
    job = TranslateJob(log=lambda _msg: stop.set(), stop_event=stop)
    started = time.monotonic()
    with pytest.raises(StopRequested):
        provider.complete("system", "prompt", job=job, attempts=3)
    assert time.monotonic() - started < 0.5
    assert provider._call.call_count == 1


def test_context_request_timeout_reaches_openai_client(monkeypatch):
    import openai

    from reviewtrans.core.providers.translate.llm import OpenAICompatTranslator

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    client = Mock()
    client.chat.completions.create.return_value = response
    factory = Mock(return_value=client)
    monkeypatch.setattr(openai, "OpenAI", factory)
    provider = OpenAICompatTranslator(ProviderProfile(kind="openai"))
    provider.request_timeout = translate_module.CONTEXT_REQUEST_TIMEOUT
    assert provider.complete("system", "prompt", attempts=1) == "ok"
    assert factory.call_args.kwargs["timeout"] == 60
    assert factory.call_args.kwargs["max_retries"] == 0


def test_fast_stage_transition_is_not_throttled(qapp, home, tmp_path, monkeypatch):
    from reviewtrans.ui import jobs as jobs_module
    from reviewtrans.ui.state import AppState

    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Queue")
    doc = state.store.add_video(state.project, "episode.mp4")
    monkeypatch.setattr(state.jobs, "_start_next", lambda: None)
    job = state.enqueue([doc.id], ["asr", "translate", "context", "tts", "mix", "render"])[0]
    changed = []
    state.jobs.jobChanged.connect(lambda j: changed.append((j.progress, j.message)))
    monkeypatch.setattr(jobs_module.time, "time", lambda: 100.0)

    def run(pipeline, *args, **kwargs):
        ctx = pipeline.ctx
        ctx.log("── episode: Dịch")
        ctx.progress(100, "Đã dịch 63/63 câu")
        # Cùng thời điểm, cùng phần trăm tổng thể 33% như trong lỗi thực tế.
        ctx.log("── episode: Cập nhật ngữ cảnh")
        ctx.progress(0, "Cập nhật ngữ cảnh")
        assert changed[-1][1] == "Cập nhật ngữ cảnh"
        assert changed[-1][0] == pytest.approx(100 / 3)

    monkeypatch.setattr(jobs_module.VideoPipeline, "run", run)
    jobs_module._JobThread(state.jobs, job).run()
    assert job.state == jobs_module.DONE
