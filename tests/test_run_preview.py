from unittest.mock import Mock

import pytest
from PyQt6 import QtCore, QtWidgets

from reviewtrans.core.config import SettingsStore
from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import RunContext, runner
from reviewtrans.ui.pages import projects
from reviewtrans.ui.state import AppState
from reviewtrans.ui.widgets.run_preview import confirm_run, summarize_run


@pytest.fixture
def batch(qapp, home, tmp_path):
    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Preview")
    docs = [state.store.add_video(state.project, f"{i}.mp4") for i in range(8)]
    for doc in docs[2:]:
        state.store.save_segments(doc.id, [Segment(id=1, source="original", text="câu đã sửa", locked=True)])
    return state, [doc.id for doc in docs], RunContext(state.store, state.project, state.settings)


def test_preview_counts_and_cancel(batch, qapp):
    state, ids, ctx = batch
    summary = summarize_run(ctx, ids, ["asr", "translate", "tts", "render"], True)
    assert "8 video · 2 cần nhận dạng · 6 đã có phụ đề · 2 cần dịch · 8 cần lồng tiếng" in summary
    assert "Trộn âm" in summary
    QtCore.QTimer.singleShot(0, lambda: qapp.activeModalWidget().reject())
    assert not confirm_run(None, ctx, ids, ["asr"], True)
    assert not state.jobs.jobs


def test_preview_enables_submit_after_counting(batch, qapp):
    _state, ids, ctx = batch
    summaries = []
    timer = QtCore.QTimer()

    def submit_when_ready():
        dialog = qapp.activeModalWidget()
        if dialog:
            button = next(b for b in dialog.findChildren(QtWidgets.QPushButton) if b.text() == "Đưa vào hàng đợi")
            if button.isEnabled():
                summaries.append(dialog.findChild(QtWidgets.QLabel).text())
                button.click()

    timer.timeout.connect(submit_when_ready)
    timer.start(10)
    watchdog = QtCore.QTimer(singleShot=True)
    watchdog.timeout.connect(lambda: qapp.activeModalWidget().reject())
    watchdog.start(3000)
    try:
        assert confirm_run(None, ctx, ids, ["asr"], True)
    finally:
        timer.stop()
        watchdog.stop()
    assert "2 cần nhận dạng" in summaries[0]


@pytest.mark.parametrize("only_missing", [True, False])
def test_run_preserves_existing_segments(batch, monkeypatch, only_missing):
    state, ids, ctx = batch
    monkeypatch.setattr(runner, "ensure_media_info", lambda doc: None)
    recognize = Mock(return_value=[Segment(id=1, source="mới")])
    monkeypatch.setattr(runner, "run_asr", recognize)
    pipeline = runner.VideoPipeline(ctx, ids[-1])
    original = state.store.load_segments(ids[-1])
    pipeline.run(["asr"], only_missing=only_missing)
    recognize.assert_not_called()
    assert state.store.load_segments(ids[-1]) == original
    pipeline.force_asr = True
    pipeline.run(["asr"])
    assert state.store.load_segments(ids[-1]) == recognize.return_value
    runner.VideoPipeline(ctx, ids[0]).run(["asr"])
    assert recognize.call_count == 2


def test_project_confirms_scope_and_replacement(batch, monkeypatch):
    state, ids, _ctx = batch
    tab = projects.VideosTab(state)
    tab.refresh()
    tab.table.selectRow(5)
    monkeypatch.setattr(state.jobs, "_start_next", lambda: None)
    preview = Mock(return_value=False)
    monkeypatch.setattr(projects, "confirm_run", preview)
    tab.run(selected=True)
    assert preview.call_args.args[2] == [ids[5]]
    assert not state.jobs.jobs
    preview.return_value = True
    tab.run(selected=False)
    assert preview.call_args.args[2] == ids
    assert all(not job.force_asr for job in state.jobs.jobs)
    state.jobs.jobs.clear()
    confirm = Mock(return_value=False)
    monkeypatch.setattr(projects, "confirm", confirm)
    tab.recognize_again()
    assert not state.jobs.jobs
    confirm.return_value = True
    tab.recognize_again()
    job, = state.jobs.jobs
    assert job.video_id == ids[5] and job.force_asr and job.steps == ["asr"]
    tab.close()
