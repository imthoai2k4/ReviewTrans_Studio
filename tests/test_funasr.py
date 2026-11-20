from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reviewtrans.core.config import AppSettings, SettingsStore
from reviewtrans.core.models import Project, VideoDoc, from_dict
from reviewtrans.core.pipeline import RunContext, asr, funasr
from reviewtrans.core.proc import StopRequested
from reviewtrans.core.srt import parse_srt
from reviewtrans.core.store import ProjectStore


def context(tmp_path, **settings):
    return RunContext(ProjectStore(tmp_path), Project(), AppSettings(**settings))


@pytest.mark.parametrize("language,expected", [("zh", "paraformer"), ("zh-tw", "paraformer"),
                                                ("en", "paraformer"), ("auto", "paraformer")])
def test_auto_engine_uses_explicit_source_language(tmp_path, language, expected):
    ctx = context(tmp_path)
    ctx.project.source_language = language
    assert asr.resolve_asr_engine(ctx, VideoDoc()) == expected


def test_engine_overrides_and_old_project_compatibility(tmp_path):
    ctx = context(tmp_path, default_asr_engine="whisper")
    ctx.project = from_dict(Project, {"source_language": "zh"})
    doc = from_dict(VideoDoc, {})
    assert asr.resolve_asr_engine(ctx, doc) == "paraformer"
    ctx.project.asr_engine = "paraformer"
    assert asr.resolve_asr_engine(ctx, doc) == "paraformer"
    doc.asr_engine = "whisper"
    assert asr.resolve_asr_engine(ctx, doc) == "paraformer"
    doc.asr_engine, doc.source_language = "paraformer", "ja"
    with pytest.raises(ValueError, match="Paraformer"):
        asr.resolve_asr_engine(ctx, doc)


def payload():
    return {"device": "cpu", "result": {"text": "你好。Hello world!", "sentence_info": [
        {"start": 1230, "end": 2340, "text": "你好。"},
        {"start": 35200, "end": 36900, "sentence": "Hello world!"},
    ]}}


def test_sentence_times_keep_original_silence_and_mixed_text():
    segments = funasr.segments_from_funasr(payload(), 40)
    assert [(s.id, s.start, s.end, s.source) for s in segments] == [
        (1, 1.23, 2.34, "你好。"), (2, 35.2, 36.9, "Hello world!")]


@pytest.mark.parametrize("sentence", [
    {"text": "你好"}, {"text": "你好", "start": 100, "end": 10},
    {"text": "你好", "start": -100, "end": -1},
    {"text": "你好", "start": float("nan"), "end": 100},
])
def test_invalid_timestamps_are_not_fabricated(sentence):
    with pytest.raises(RuntimeError, match="timestamp"):
        funasr.segments_from_funasr({"result": {"sentence_info": [sentence]}})


def test_empty_audio_result_and_missing_sentence_times():
    assert funasr.segments_from_funasr({"result": {"text": "", "sentence_info": []}}) == []
    with pytest.raises(RuntimeError, match="timestamp"):
        funasr.segments_from_funasr({"result": {"text": "你好", "timestamp": [[100, 200]]}})
    data = payload()
    data["result"]["sentence_info"][1]["start"] = 1000
    with pytest.raises(RuntimeError, match="chồng lấn"):
        funasr.segments_from_funasr(data)


def test_leading_negative_timestamp_clips_to_audio_origin_only():
    data = {"result": {"sentence_info": [
        {"start": -90, "end": 1290, "text": "仙岛现在还不允许登岛，"},
        {"start": 1290, "end": 4109, "text": "我看到不少海妖海兽想冲上去，"},
    ]}}
    original = json.dumps(data)
    log = Mock()
    segments = funasr.segments_from_funasr(data, 10, log=log)
    assert [(s.start, s.end) for s in segments] == [(0, 1.29), (1.29, 4.109)]
    assert [s.source for s in segments] == [s["text"] for s in data["result"]["sentence_info"]]
    assert json.dumps(data) == original
    log.assert_called_once()
    data["result"]["sentence_info"][1]["start"] = -20
    with pytest.raises(RuntimeError, match="timestamp"):
        funasr.segments_from_funasr(data)


@pytest.mark.parametrize("times", [
    [(108210, 109079), (109000, 111519)],
    [(135979, 137989), (137850, 139590)],
    [(1000, 2001), (2000, 3000), (2990, 4000)],
])
def test_overlapping_sentence_boundaries_keep_text_and_onsets(times):
    data = {"result": {"sentence_info": [
        {"start": start, "end": end, "text": f"句{i}。"}
        for i, (start, end) in enumerate(times)
    ]}}
    original = json.dumps(data)
    log = Mock()
    segments = funasr.segments_from_funasr(data, log=log)
    assert [s.source for s in segments] == [f"句{i}。" for i in range(len(times))]
    assert [s.start for s in segments] == [start / 1000 for start, _ in times]
    assert all(a.start < a.end <= b.start for a, b in zip(segments, segments[1:]))
    assert segments[-1].end == times[-1][1] / 1000
    assert json.dumps(data) == original
    log.assert_called_once()


@pytest.mark.parametrize("times", [
    [(1000, 3000), (500, 4000)],
    [(1000, 3000), (1000, 4000)],
    [(1000, 3000), (2000, 2500)],
])
def test_reversed_or_nested_sentence_times_still_fail(times):
    data = {"result": {"sentence_info": [
        {"start": start, "end": end, "text": "你好。"} for start, end in times
    ]}}
    with pytest.raises(RuntimeError, match="sai thứ tự"):
        funasr.segments_from_funasr(data)


def test_routing_to_worker_writes_source_srt(tmp_path, monkeypatch):
    ctx = context(tmp_path, asr_device="cpu", asr_threads=3)
    doc = VideoDoc(duration=40)
    audio = tmp_path / "audio.wav"
    monkeypatch.setattr(asr, "extract_audio", lambda *_: audio)
    monkeypatch.setattr(funasr, "worker_command", lambda: ["python", "worker.py"])
    monkeypatch.setattr(funasr, "models_ready", lambda: True)
    monkeypatch.setattr(funasr, "funasr_models_dir", lambda: tmp_path / "models")

    def run(command, **kwargs):
        assert command[command.index("--input") + 1] == str(audio)
        assert command[command.index("--device") + 1] == "cpu"
        assert command[command.index("--threads") + 1] == "3"
        Path(command[command.index("--output") + 1]).write_text(json.dumps(payload()), encoding="utf-8")
        return 0

    monkeypatch.setattr(funasr, "run", run)
    segments = asr.run_asr(ctx, doc)
    srt = ctx.store.cache_dir(doc.id) / "asr.srt"
    assert parse_srt(srt.read_text(encoding="utf-8")) == [(s.start, s.end, s.source) for s in segments]


def test_worker_failure_does_not_reuse_stale_srt_or_switch_engine(tmp_path, monkeypatch):
    ctx, doc = context(tmp_path), VideoDoc()
    cache = ctx.store.cache_dir(doc.id)
    for filename in ("asr.srt", "asr.paraformer.json"):
        (cache / filename).write_text("stale", encoding="utf-8")
    monkeypatch.setattr(funasr, "worker_command", lambda: ["python", "worker.py"])
    monkeypatch.setattr(funasr, "models_ready", lambda: True)
    monkeypatch.setattr(funasr, "run", lambda *a, **kw: 1)
    with pytest.raises(RuntimeError, match="FunASR"):
        funasr.run_paraformer(ctx, doc, tmp_path / "audio.wav")
    assert not (cache / "asr.srt").exists()
    assert not (cache / "asr.paraformer.json").exists()


def test_stop_propagates_without_saving_results(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.stop_event.set()
    worker = Mock()
    monkeypatch.setattr(funasr, "worker_command", worker)
    with pytest.raises(StopRequested):
        asr.run_asr(ctx, VideoDoc())
    worker.assert_not_called()


def test_srt_input_bypasses_all_asr_runtime(tmp_path, monkeypatch):
    srt = tmp_path / "input.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n你好。\n", encoding="utf-8")
    monkeypatch.setattr(asr, "run_paraformer", Mock(side_effect=AssertionError("must not run")))
    segments = asr.run_asr(context(tmp_path), VideoDoc(source_mode="srt", srt_path=str(srt)))
    assert [(s.start, s.end, s.source) for s in segments] == [(1, 2, "你好。")]


def test_worker_requests_real_timestamps_and_reports_cpu_fallback(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("funasr_worker", Path(__file__).parents[1] / "scripts/funasr_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    model = Mock()
    model.generate.return_value = [payload()["result"]]
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=factory))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), set_num_threads=Mock()))
    samples = type("Samples", (), {"ndim": 1, "__len__": lambda self: 16000})()
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=lambda *a, **kw: (samples, 16000)))
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda *_: "test")
    progress = Mock()
    monkeypatch.setattr(worker, "event", progress)
    output = worker.transcribe(SimpleNamespace(device="cuda", models=tmp_path, threads=3, input=tmp_path / "audio.wav"))
    assert output["device"] == "cpu"
    assert factory.call_args.kwargs["trust_remote_code"] is False
    assert factory.call_args.kwargs["vad_kwargs"]["max_single_segment_time"] == 15000
    assert model.generate.call_args.kwargs["sentence_timestamp"] is True
    assert model.generate.call_args.kwargs["pred_timestamp"] is True
    assert any("CPU" in call.args[1] for call in progress.call_args_list)


def test_ui_engine_overrides_persist(qapp, tmp_path, home, monkeypatch):
    from PyQt6 import QtCore
    from reviewtrans.ui.pages.editor import VideoSettingsPanel
    from reviewtrans.ui.pages.projects import ProjectSettingsTab
    from reviewtrans.ui.pages.settings import SettingsPage
    from reviewtrans.ui.state import AppState
    from reviewtrans.ui.widgets.provider_field import ProviderField

    monkeypatch.setattr(ProviderField, "fetch_remote", lambda *a, **kw: None)
    state = AppState(SettingsStore())
    state.create_project(tmp_path, "ASR")
    settings = SettingsPage(state)
    settings.asr_engine.set_value("paraformer")
    settings.commit()
    assert state.settings.default_asr_engine == "paraformer"
    project = ProjectSettingsTab(state)
    project.load()
    project.asr_engine.set_value("paraformer")
    project.commit()
    assert state.store.load_project().asr_engine == "paraformer"
    video = VideoSettingsPanel()
    doc = VideoDoc()
    video.set_document(doc)
    video.asr_engine.set_value("paraformer")
    assert doc.asr_engine == "paraformer"
    qapp.processEvents()  # Drain the provider fields' deferred initial refreshes.
    for widget in (settings, project, video):
        for timer in widget.findChildren(QtCore.QTimer):
            timer.stop()
        widget.close()
        widget.deleteLater()
    state.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
