from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6 import QtCore, QtTest, QtWidgets

from reviewtrans.core.config import SettingsStore
from reviewtrans.core.pipeline.runner import VideoPipeline
from reviewtrans.ui.jobs import CANCELLED, DONE, ERROR, PAUSED, PAUSING, PENDING, RUNNING, STOPPED
from reviewtrans.ui.pages.queue import QueuePage
from reviewtrans.ui.state import AppState


def wait_until(qapp, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        QtTest.QTest.qWait(10)
    qapp.processEvents()
    assert condition(), "Timed out waiting for queue"


@pytest.fixture
def queue_page(qapp, home, tmp_path):
    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Queue selection")
    docs = []
    for i in range(4):
        source = tmp_path / f"episode-{i}.mp4"
        source.write_bytes(b"original video")
        doc = state.store.add_video(state.project, str(source))
        (state.store.tts_dir(doc.id) / "sample.wav").write_bytes(b"tts")
        docs.append(doc)
    state.reload_project()
    state.jobs.pause()
    jobs = state.enqueue([doc.id for doc in docs], ["tts"])
    page = QueuePage(state)
    page.clear_selection()
    yield state, page, docs, jobs
    state.jobs.shutdown()
    qapp.processEvents()
    page.timer.stop()
    page.close()


def test_checkboxes_select_all_clear_and_no_implicit_reselection(queue_page):
    state, page, _docs, jobs = queue_page
    assert not page.remove_button.isEnabled()
    page.table.selectRow(1)
    page.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    page.table.item(2, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    assert page._selected_jobs() == [jobs[0], jobs[2]]
    state.jobs._progress(jobs[0], 40, "Progress", force=True)
    assert page._selected_jobs() == [jobs[0], jobs[2]]
    page.select_all_button.click()
    assert page._selected_jobs() == jobs
    assert page.selection_summary.text() == "Đã chọn 4/4 video"
    page.clear_selection_button.click()
    state.jobs._log(jobs[0], "New log after clearing selection")
    assert not page._selected_jobs() and not page.remove_button.isEnabled()
    assert page.log_title.text() == "Log" and not page.log.toPlainText()
    page.table.selectRow(0)
    page.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    page.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not page._selected_jobs()


@pytest.mark.parametrize("use_checks", [False, True])
def test_remove_multiple_jobs_preserves_files_and_remaining_rows(queue_page, use_checks):
    state, page, docs, jobs = queue_page
    removed, lock_changes = [], []
    state.jobs.jobsRemoved.connect(removed.append)
    state.jobs.videoLocksChanged.connect(lambda: lock_changes.append(True))
    if use_checks:
        for row in (0, 2):
            page.table.item(row, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    else:
        for row in (0, 2):
            page.table.selectionModel().select(
                page.table.model().index(row, 0),
                QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
    page.remove_button.click()
    assert state.jobs.jobs == [jobs[1], jobs[3]]
    assert removed == [[jobs[0].id, jobs[2].id]] and len(lock_changes) == 1
    assert page.table.rowCount() == 2
    assert page.rows == {jobs[1].id: 0, jobs[3].id: 1}
    assert set(page.bars) == {jobs[1].id, jobs[3].id}
    assert not state.is_video_locked(docs[0].id) and state.is_video_locked(docs[1].id)
    assert len(state.store.load_project().videos) == 4
    for doc in docs:
        assert state.store.load_video(doc.id).id == doc.id
        assert (state.store.tts_dir(doc.id) / "sample.wav").read_bytes() == b"tts"
        assert state.store.load_video(doc.id).source_path == doc.source_path
        assert Path(doc.source_path).read_bytes() == b"original video"
    page.clear_selection()
    state.jobs._log(jobs[0], "Late log for removed job")
    state.jobs._progress(jobs[0], 50, "Late progress", force=True)
    assert not page.log.toPlainText() and not page._selected_jobs()
    page.table.selectRow(1)
    state.jobs._progress(jobs[3], 75, "Remaining job", force=True)
    assert page.bars[jobs[3].id].value() == 750
    state.jobs._log(jobs[3], "Correct remaining log")
    assert page.log_title.text() == f"Log — {jobs[3].video_name}"
    assert "Correct remaining log" in page.log.toPlainText()


@pytest.mark.parametrize("job_state", [PENDING, PAUSED, DONE, ERROR, STOPPED, CANCELLED])
def test_remove_supported_states_and_empty_queue_controls(queue_page, job_state):
    state, page, _docs, jobs = queue_page
    for job in jobs:
        job.state = job_state
        state.jobs.jobChanged.emit(job)
    page.select_all_button.click()
    page.remove_button.click()
    assert not state.jobs.jobs and not page.rows and not page.bars
    assert page.table.rowCount() == 0 and not state.jobs.locked_videos()
    assert not page.remove_button.isEnabled() and not page.select_all_button.isEnabled()
    assert not page.stop_button.isEnabled()
    assert page.selection_summary.text() == "Đã chọn 0/0 video"
    assert page.log_title.text() == "Log" and not page.log.toPlainText()
    assert not state.jobs.remove([jobs[0].id])


@pytest.mark.parametrize("job_state", [RUNNING, PAUSING, PAUSED, DONE])
def test_current_worker_is_protected_until_thread_finishes(queue_page, monkeypatch, job_state):
    state, page, _docs, jobs = queue_page
    jobs[0].state = job_state
    with monkeypatch.context() as patch:
        patch.setattr(state.jobs, "_thread", SimpleNamespace(job=jobs[0]))
        page.select_all_button.click()
        assert not page.remove_button.isEnabled()
        page._remove_selected()
        assert len(state.jobs.jobs) == 4  # Không xoá một phần lựa chọn.
        assert state.jobs.remove([job.id for job in jobs]) == [job.id for job in jobs[1:]]
        assert state.jobs.jobs == [jobs[0]]
        page._clear()
        assert state.jobs.jobs == [jobs[0]] and page.table.rowCount() == 1


def test_clear_finished_and_direct_removal_update_all_queue_views(queue_page):
    state, page, _docs, jobs = queue_page
    other_page = QueuePage(state)
    try:
        jobs[0].state, jobs[1].state, jobs[2].state = DONE, PAUSED, ERROR
        page._clear()
        assert state.jobs.jobs == [jobs[1], jobs[3]]
        assert page.table.rowCount() == other_page.table.rowCount() == 2
        state.jobs.remove([jobs[1].id])
        assert state.jobs.jobs == [jobs[3]]
        assert page.rows == other_page.rows == {jobs[3].id: 0}
    finally:
        other_page.timer.stop()
        other_page.close()


def test_remove_only_selected_job_for_same_video_keeps_other_job_and_lock(queue_page):
    state, page, docs, jobs = queue_page
    duplicate = state.jobs.retry(jobs[0])
    page.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    page.remove_button.click()
    assert jobs[0] not in state.jobs.jobs and duplicate in state.jobs.jobs
    assert state.is_video_locked(docs[0].id)
    assert page.rows[duplicate.id] == page.table.rowCount() - 1


def test_log_follows_current_video_with_multiple_checked_jobs(queue_page):
    state, page, _docs, jobs = queue_page
    state.jobs._log(jobs[3], "Log for fourth video")
    page.select_all_button.click()
    page.table.setCurrentCell(3, 3, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
    assert page.log_title.text() == f"Log — {jobs[3].video_name}"
    assert "Log for fourth video" in page.log.toPlainText()
    assert page._selected_jobs() == jobs


def test_context_menu_preserves_checked_group_and_targets_clicked_job(queue_page, monkeypatch):
    state, page, _docs, jobs = queue_page
    page.resize(1200, 700)
    page.show()
    page.select_all_button.click()
    menus = []

    def choose_delete(menu, pos):
        menus.append(menu)
        next(action for action in menu.actions() if action.text() == "Xoá đã chọn khỏi hàng đợi").trigger()

    monkeypatch.setattr(QtWidgets.QMenu, "exec", choose_delete)
    pos = page.table.visualItemRect(page.table.item(2, 3)).center()
    page._menu(pos)
    assert menus and not state.jobs.jobs and page.table.rowCount() == 0


def test_delete_and_select_all_shortcuts(queue_page, qapp):
    state, page, _docs, _jobs = queue_page
    page.show()
    page.activateWindow()
    page.table.setFocus()
    qapp.processEvents()
    QtTest.QTest.keyClick(page.table, QtCore.Qt.Key.Key_A, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert len(page._selected_jobs()) == 4
    QtTest.QTest.keyClick(page.table, QtCore.Qt.Key.Key_Delete)
    assert not state.jobs.jobs and page.table.rowCount() == 0


def test_remove_waiting_then_paused_video_does_not_run_them_on_resume(qapp, home, tmp_path, monkeypatch):
    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Running queue")
    docs = [state.store.add_video(state.project, f"episode-{i}.mp4") for i in range(3)]
    for doc in docs:
        doc.duration, doc.width, doc.height = 2, 320, 240
        state.store.save_video(doc)
    calls, entered = [], threading.Event()

    def step(pipeline, name, only_missing):
        calls.append(pipeline.video_id)
        if pipeline.video_id == docs[0].id:
            entered.set()
            pipeline.ctx.stop_event.wait(3)
            pipeline.ctx.check_stop()

    monkeypatch.setattr(VideoPipeline, "_run_step", step)
    page = QueuePage(state)
    try:
        jobs = state.enqueue([doc.id for doc in docs], ["tts"])
        wait_until(qapp, entered.is_set)
        page.table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        page.remove_button.click()
        assert state.jobs.current() is jobs[0] and jobs[0].state == RUNNING
        assert state.jobs.jobs == [jobs[0], jobs[2]]
        page.select_all_button.click()
        assert not page.remove_button.isEnabled()
        page.stop_button.click()
        wait_until(qapp, lambda: not state.jobs.is_busy())
        assert jobs[0].state == PAUSED and page.remove_button.isEnabled()
        page.clear_selection_button.click()
        page.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        page.remove_button.click()
        assert state.jobs.jobs == [jobs[2]] and not state.is_video_locked(docs[0].id)
        page.resume_button.click()
        wait_until(qapp, lambda: jobs[2].state == DONE and not state.jobs.is_busy())
        assert calls == [docs[0].id, docs[2].id]
    finally:
        state.jobs.shutdown()
        qapp.processEvents()
        page.timer.stop()
        page.close()
