from __future__ import annotations

from PyQt6 import QtCore, QtTest

from reviewtrans.core.config import ProviderProfile, SettingsStore
from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import translate
from reviewtrans.core.pipeline.runner import VideoPipeline
from reviewtrans.ui.jobs import DONE, ERROR
from reviewtrans.ui.pages.queue import QueuePage
from reviewtrans.ui.state import AppState


def make_project(tmp_path, count):
    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Queue recovery")
    docs = [state.store.add_video(state.project, f"episode-{index}.mp4") for index in range(count)]
    for doc in docs:
        doc.duration, doc.width, doc.height = 10, 320, 240
        state.store.save_video(doc)
    return state, docs


def wait_for_job(state, job, status):
    for _ in range(300):
        QtTest.QTest.qWait(10)
        if job.state == status and not state.jobs.is_busy():
            return
    assert job.state == status and not state.jobs.is_busy(), job.logs


def test_retry_button_restarts_render_without_repeating_implicit_mix(qapp, home, tmp_path, monkeypatch):
    state, (doc,) = make_project(tmp_path, 1)
    calls = []

    def run_step(pipeline, step, only_missing):
        calls.append(step)
        if step == "render" and calls.count("render") == 1:
            raise RuntimeError("Lỗi bộ mã hoá")

    monkeypatch.setattr(VideoPipeline, "_run_step", run_step)
    page = QueuePage(state)
    try:
        job, = state.enqueue([doc.id], ["asr", "render"])
        wait_for_job(state, job, ERROR)
        assert job.step == "render"
        action = page.table.cellWidget(page.rows[job.id], page.table.columnCount() - 1)
        assert action.isEnabled()
        action.click()
        wait_for_job(state, job, DONE)
        assert calls == ["asr", "mix", "render", "render"]
        assert state.jobs.jobs == [job]
    finally:
        state.jobs.shutdown()
        page.close()


def test_switch_api_in_error_row_preserves_translated_sentences(qapp, home, tmp_path, monkeypatch):
    state, (doc,) = make_project(tmp_path, 1)
    first = ProviderProfile(name="API chính", kind="openai", model="primary")
    second = ProviderProfile(name="API dự phòng", kind="openai", model="backup")
    state.settings.translate_profiles = [first, second]
    state.settings.default_translate_profile = first.id
    state.settings.translate_batch_size = 1
    state.store.save_segments(doc.id, [Segment(id=i, start=i, end=i + 1, source=str(i)) for i in (1, 2)])
    calls = []

    class Translator:
        max_batch = 1
        uses_context = False

        def __init__(self, profile, model):
            self.profile = profile
            self.model = model or profile.model

        def thinking_mode(self):
            return ""

        def translate(self, lines, job):
            calls.append((self.model, lines))
            if self.profile.id == first.id and lines == ["2"]:
                raise RuntimeError("Hết hạn mức")
            return [f"{self.model}-{line}" for line in lines]

    monkeypatch.setattr(translate, "create_translator", Translator)
    monkeypatch.setattr(
        "reviewtrans.ui.pages.queue.QtWidgets.QInputDialog.getItem",
        lambda parent, title, label, items, *args: (next(item for item in items if "backup" in item), True),
    )
    page = QueuePage(state)
    try:
        job, = state.enqueue([doc.id], ["translate"])
        wait_for_job(state, job, ERROR)
        action = page.table.cellWidget(page.rows[job.id], page.table.columnCount() - 1)
        action.menu().actions()[0].trigger()
        wait_for_job(state, job, DONE)
        assert calls == [("primary", ["1"]), ("primary", ["2"]), ("backup", ["2"])]
        assert [seg.text for seg in state.store.load_segments(doc.id)] == ["primary-1", "backup-2"]
        assert state.store.load_video(doc.id).translate_profile == second.id
        assert state.settings.default_translate_profile == first.id
    finally:
        state.jobs.shutdown()
        page.close()


def test_queue_can_launch_selected_then_all_project_videos(qapp, home, tmp_path, monkeypatch):
    state, docs = make_project(tmp_path, 3)
    state.store.save_segments(docs[0].id, [Segment(id=1, start=0, end=1, source="Giữ câu có sẵn")])
    monkeypatch.setattr(state.jobs, "_start_next", lambda: None)
    choices = []

    def choose(dialog):
        if not dialog.all_videos:
            dialog.videos.item(1).setCheckState(QtCore.Qt.CheckState.Checked)
        choices.append(dialog.video_ids())
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr("reviewtrans.ui.pages.queue.QueueRunDialog.exec", choose)
    page = QueuePage(state)
    try:
        page.run_selected_button.click()
        assert [job.video_id for job in state.jobs.jobs] == [docs[1].id]
        page.run_all_button.click()
        assert choices == [[docs[1].id], [docs[0].id, docs[2].id]]
        assert [job.video_id for job in state.jobs.jobs] == [doc.id for doc in (docs[1], docs[0], docs[2])]
        assert "asr" not in state.jobs.jobs[1].steps and "asr" in state.jobs.jobs[2].steps
    finally:
        state.jobs.shutdown()
        page.close()
