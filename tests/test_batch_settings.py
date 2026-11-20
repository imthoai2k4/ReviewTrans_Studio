from __future__ import annotations

import pytest

from reviewtrans.core.config import SettingsStore, builtin_presets
from reviewtrans.core.models import BatchSettings, Layer, Project, SubtitleStyle, VideoDoc, from_dict, to_dict
from reviewtrans.ui.state import AppState


def test_batch_settings_roundtrip_and_old_projects():
    assert from_dict(Project, {"name": "Old"}).batch_settings is None
    project = Project(batch_settings=BatchSettings(
        original_volume=45, style=SubtitleStyle(position="top", text_color="#FFE14D"),
    ))
    restored = from_dict(Project, to_dict(project))
    assert restored.batch_settings == project.batch_settings


def test_batch_settings_preserve_other_editor_settings():
    doc = VideoDoc(layers=[Layer(type="text", text="Title")], flip_horizontal=True)
    doc.audio.bgm_path = "music.wav"
    doc.audio.dub_volume = 135
    doc.audio.speed = 1.3
    doc.audio.duck_volume = 15
    doc.tracks.original_audio = False
    doc.stages = {s: "done" for s in ["asr", "translate", "tts", "mix", "render"]}
    settings = BatchSettings(original_mode="duck", original_volume=60, style=SubtitleStyle(position="top"))
    doc.apply_batch_settings(settings)
    assert doc.audio.original_mode == "duck" and doc.audio.original_volume == 60
    assert doc.tracks.original_audio
    assert doc.audio.bgm_path == "music.wav" and doc.audio.dub_volume == 135
    assert doc.audio.speed == 1.3 and doc.audio.duck_volume == 15
    assert doc.layers[0].text == "Title" and doc.flip_horizontal
    assert doc.stage("tts") == "done" and doc.stage("mix") == doc.stage("render") == "stale"
    doc.style.position = "middle"
    assert settings.style.position == "top"


def test_subtitle_changes_only_invalidate_render():
    doc = VideoDoc(stages={"mix": "done", "render": "done"})
    doc.apply_batch_settings(BatchSettings(style=SubtitleStyle(position="middle"), subtitles=False))
    assert not doc.tracks.subtitles
    assert doc.stage("mix") == "done" and doc.stage("render") == "stale"
    before = to_dict(doc)
    doc.apply_batch_settings(BatchSettings(enabled=False, original_volume=0))
    assert to_dict(doc) == before


def test_batch_dialog_presets_and_custom_style(qapp):
    from reviewtrans.ui.widgets.batch_settings import BatchSettingsDialog

    settings = BatchSettings(style=SubtitleStyle(outline_width=7, bg_opacity=35))
    presets = builtin_presets()
    dialog = BatchSettingsDialog(settings, presets)
    dialog.position.set_value("top")
    dialog.size.setValue(64)
    dialog.original_volume.setValue(40)
    result = dialog.settings()
    assert result.style.position == "top" and result.style.font_size == 64
    assert result.style.outline_width == 7 and result.style.bg_opacity == 35
    assert result.original_volume == 40
    assert settings.style.position == "bottom" and settings.original_volume == 20
    assert dialog.bg_opacity.value() == 35 and not dialog.bg_opacity.isEnabled()
    dialog.bg_enabled.setChecked(True)
    assert dialog.bg_opacity.isEnabled()
    for opacity in (0, 50, 100):
        dialog.bg_opacity.setValue(opacity)
        assert dialog.settings().style.bg_enabled
        assert dialog.settings().style.bg_opacity == opacity
        assert dialog.preview.style_.bg_opacity == opacity
    restored = from_dict(Project, to_dict(Project(batch_settings=dialog.settings())))
    assert restored.batch_settings.style.bg_opacity == 100
    dialog.preset.set_value("1")
    assert dialog.settings().style == presets[1]
    assert not dialog.bg_enabled.isChecked() and not dialog.bg_opacity.isEnabled()
    dialog.preset.set_value("2")
    assert dialog.bg_enabled.isChecked() and dialog.bg_opacity.value() == presets[2].bg_opacity
    dialog.preset.set_value("1")
    dialog.position.set_value("middle")
    assert dialog.preset.value() == ""
    assert dialog.settings().style.text_color == presets[1].text_color
    dialog.original_mode.set_value("mute")
    dialog.hide_subtitles.setChecked(True)
    assert not dialog.original_volume.isEnabled() and not dialog.style_box.isEnabled()
    assert not dialog.settings().subtitles
    assert dialog.preview.text == ""
    hidden_settings = from_dict(Project, to_dict(Project(batch_settings=dialog.settings()))).batch_settings
    reopened = BatchSettingsDialog(hidden_settings, presets)
    assert reopened.hide_subtitles.isChecked() and reopened.preview.text == ""
    reopened.hide_subtitles.setChecked(False)
    assert reopened.settings().subtitles and reopened.style_box.isEnabled()
    assert reopened.preview.text
    assert reopened.settings().style == dialog.settings().style
    reopened.reject()
    dialog.reject()


@pytest.mark.parametrize("selected", [False, True])
def test_project_batch_run_captures_settings(qapp, home, tmp_path, monkeypatch, selected):
    from reviewtrans.ui.pages.projects import VideosTab

    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Batch")
    first = state.store.add_video(state.project, "ep1.mp4")
    second = state.store.add_video(state.project, "ep2.mp4")
    tab = VideosTab(state)
    tab.refresh()
    tab.table.selectRow(1)
    tab.batch_enabled.setChecked(True)
    state.project.batch_settings.original_volume = 35
    state.project.batch_settings.style.position = "top"
    monkeypatch.setattr("reviewtrans.ui.pages.projects.confirm_run", lambda *args: True)
    monkeypatch.setattr(state.jobs, "_start_next", lambda: None)
    tab.run(selected=selected)
    jobs = state.jobs.jobs
    assert [job.video_id for job in jobs] == ([second.id] if selected else [first.id, second.id])
    assert all(job.batch_settings.original_volume == 35 for job in jobs)
    assert all(job.batch_settings.style.position == "top" for job in jobs)
    assert all(job.only_missing for job in jobs)
    state.project.batch_settings.original_volume = 90
    state.project.batch_settings.style.position = "bottom"
    assert all(job.batch_settings.original_volume == 35 for job in jobs)
    assert all(job.batch_settings.style.position == "top" for job in jobs)
    retried = state.jobs.retry(jobs[0])
    assert retried.batch_settings == jobs[0].batch_settings
    assert retried.batch_settings is not jobs[0].batch_settings
    tab.batch_enabled.setChecked(False)
    tab.run(selected=selected)
    assert state.jobs.jobs[-1].batch_settings is None
    state.open_project(state.store.root)
    assert state.project.batch_settings.enabled is False
    assert not tab.batch_enabled.isChecked()
    # Chạy từ Editor giữ thiết lập riêng, dù project bật cấu hình hàng loạt.
    tab.batch_enabled.setChecked(True)
    editor_job = state.enqueue([first.id], ["render"])[0]
    assert editor_job.batch_settings is None
    tab.close()


def test_worker_applies_batch_settings_before_pipeline(qapp, home, tmp_path, monkeypatch):
    from reviewtrans.ui import jobs as jobs_module

    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Worker")
    doc = state.store.add_video(state.project, "ep.mp4")
    doc.audio.original_volume = 95
    doc.style.position = "bottom"
    state.store.save_video(doc)
    monkeypatch.setattr(state.jobs, "_start_next", lambda: None)
    settings = BatchSettings(original_mode="mute", original_volume=0, style=SubtitleStyle(position="top"))
    job = state.enqueue([doc.id], ["mix", "render"], batch_settings=settings)[0]
    observed = []

    def fake_run(pipeline, steps, **kwargs):
        observed.append(pipeline.ctx.store.load_video(doc.id))
        assert steps == ["mix", "render"] and kwargs["only_missing"]

    monkeypatch.setattr(jobs_module.VideoPipeline, "run", fake_run)
    jobs_module._JobThread(state.jobs, job).run()
    assert job.state == jobs_module.DONE
    assert observed[0].audio.original_volume == 0
    assert observed[0].audio.original_mode == "mute" and not observed[0].tracks.original_audio
    assert observed[0].style.position == "top"


@pytest.mark.parametrize("original_volume", [0, 10, 100])
def test_batch_render_uses_shared_audio_and_subtitles(qapp, home, tmp_path, original_volume):
    import re
    import subprocess
    import time

    from PyQt6 import QtCore

    from reviewtrans.core.models import Segment
    from reviewtrans.core.paths import find_tool
    from reviewtrans.core.proc import probe, run_checked

    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        pytest.skip("cần ffmpeg")
    source = tmp_path / "source.mp4"
    run_checked([
        str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=25:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-ac", "2", "-shortest", str(source),
    ], "tạo video test")
    state = AppState(SettingsStore())
    state.settings.render.video_codec = "libx264"
    state.settings.render.hwaccel_decode = False
    state.create_project(tmp_path, "Render")
    docs = [state.store.add_video(state.project, str(source), probe(source)) for _ in range(2)]
    for index, doc in enumerate(docs):
        doc.name = f"ep{index}"
        doc.audio.original_volume = 100
        state.store.save_video(doc)
        state.store.save_segments(doc.id, [Segment(start=0, end=2, text="Phụ đề chung")])
    jobs = state.enqueue([doc.id for doc in docs], ["render"], batch_settings=BatchSettings(
        original_volume=original_volume, style=SubtitleStyle(position="top", font_size=64, text_color="#FFE14D"),
    ))
    deadline = time.monotonic() + 40
    try:
        while any(job.state in ("pending", "running") for job in jobs) and time.monotonic() < deadline:
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(50, loop.quit)
            loop.exec()
        assert all(job.state == "done" for job in jobs), "\n".join(line for job in jobs for line in job.logs)
        def mean_volume(path):
            volume = subprocess.run([
                str(ffmpeg), "-i", str(path), "-af", "volumedetect", "-f", "null", "-",
            ], capture_output=True, text=True).stderr
            match = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", volume)
            assert match
            return float(match.group(1))

        source_level = mean_volume(source)
        for doc in docs:
            saved = state.store.load_video(doc.id)
            assert saved.style.position == "top" and saved.audio.original_volume == original_volume
            ass = (state.store.cache_dir(doc.id) / "render" / "subs.ass").read_text(encoding="utf-8")
            assert "Phụ đề chung" in ass and "&H004DE1FF" in ass
            output_level = mean_volume(saved.last_output)
            if original_volume == 0:
                assert output_level < -80
            else:
                import math

                expected_change = 40 * math.log10(original_volume / 100)
                assert output_level - source_level == pytest.approx(expected_change, abs=0.7)
    finally:
        state.jobs.shutdown()
