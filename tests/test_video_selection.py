from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest
from PyQt6 import QtCore, QtWidgets

from reviewtrans.core.config import SettingsStore
from reviewtrans.core.models import Segment, VideoRef
from reviewtrans.core.proc import MediaInfo
from reviewtrans.core.store import ProjectStore
from reviewtrans.ui.jobs import PAUSED, PENDING, RUNNING, Job
from reviewtrans.ui.pages import projects as projects_module
from reviewtrans.ui.pages.projects import VideosTab
from reviewtrans.ui.state import AppState


def wait_until(qapp, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(10, loop.quit)
        loop.exec()
    qapp.processEvents()
    assert condition(), "Timed out waiting for video cleanup"


@pytest.fixture
def videos(qapp, home, tmp_path, monkeypatch):
    state = AppState(SettingsStore())
    state.create_project(tmp_path, "Videos")
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original video")
    docs = [state.store.add_video(state.project, str(source)) for _ in range(3)]
    for doc in docs:
        (state.store.tts_dir(doc.id) / "sample.wav").write_bytes(b"tts")
    tab = VideosTab(state)
    tab.refresh()
    monkeypatch.setattr(projects_module, "confirm", lambda *args: True)
    yield state, tab, docs, source
    wait_until(qapp, lambda: not tab._deleting)
    tab.close()


def test_checkboxes_select_all_clear_and_preserve_selection(videos):
    state, tab, docs, _source = videos
    assert not tab.remove_button.isEnabled()
    tab.table.selectRow(2)
    tab.table.item(2, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    tab.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    assert tab.selected_ids() == [docs[0].id, docs[2].id]
    state.project.videos.reverse()
    state.save_project()
    assert tab.checked_ids() == [docs[2].id, docs[0].id]
    tab.select_all_button.click()
    assert tab.selected_ids() == [doc.id for doc in reversed(docs)]
    assert tab.selection_summary.text() == "Đã chọn 3/3 video"
    tab.clear_selection_button.click()
    assert not tab.selected_ids() and not tab.remove_button.isEnabled()
    tab.table.selectRow(0)
    tab.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    tab.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not tab.selected_ids()  # bỏ ô cuối cùng không ngầm chọn lại dòng vừa bấm


def test_checked_videos_are_used_for_batch_run(videos, monkeypatch):
    state, tab, docs, _source = videos
    monkeypatch.setattr(projects_module, "confirm_run", lambda *args: True)
    monkeypatch.setattr(state.jobs, "_start_next", lambda: None)
    for row in (0, 2):
        tab.table.item(row, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    tab.run(selected=True)
    assert [job.video_id for job in state.jobs.jobs] == [docs[0].id, docs[2].id]
    assert tab.checked_ids() == [docs[0].id, docs[2].id]


@pytest.mark.parametrize("use_checks", [True, False])
def test_delete_all_is_asynchronous_and_keeps_originals(qapp, videos, monkeypatch, use_checks):
    state, tab, docs, source = videos
    docs.extend(state.store.add_video(state.project, str(source)) for _ in range(17))
    tab.refresh()
    release, cleaning = threading.Event(), threading.Event()
    delete = state.store.delete_video_files

    def slow_delete(video_id):
        cleaning.set()
        release.wait(3)
        delete(video_id)

    monkeypatch.setattr(state.store, "delete_video_files", slow_delete)
    save = Mock(wraps=state.store.save_project)
    monkeypatch.setattr(state.store, "save_project", save)
    changed = []
    state.projectChanged.connect(lambda: changed.append(True))
    if use_checks:
        tab.select_all_button.click()
    else:
        tab.table.selectAll()
    try:
        tab.remove_button.click()
        wait_until(qapp, cleaning.is_set)
        assert tab._deleting and tab.table.rowCount() == 0
        assert not state.project.videos and not state.store.load_project().videos
        assert save.call_count == 1 and len(changed) == 1
        assert all(state.store.video_dir(doc.id).exists() for doc in docs)
        heartbeat = []
        QtCore.QTimer.singleShot(0, lambda: heartbeat.append(True))
        wait_until(qapp, lambda: bool(heartbeat))
    finally:
        release.set()
        wait_until(qapp, lambda: not tab._deleting)
    assert all(not state.store.video_dir(doc.id).exists() for doc in docs)
    assert source.read_bytes() == b"original video"


def test_delete_bulk_subset_keeps_rows_and_resolves_paths_off_ui(qapp, videos, monkeypatch):
    state, tab, docs, source = videos
    docs.extend(state.store.add_video(state.project, str(source)) for _ in range(21))
    tab.refresh()
    kept = docs[::3]
    for row in range(len(docs)):
        if row % 3:
            tab.table.item(row, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    reads = Mock(wraps=state.store.load_segments)
    monkeypatch.setattr(state.store, "load_segments", reads)
    resolve = state.store._video_delete_path
    threads = []

    def check_path(video_id):
        threads.append(threading.get_ident())
        return resolve(video_id)

    monkeypatch.setattr(state.store, "_video_delete_path", check_path)
    tab.remove_button.click()
    wait_until(qapp, lambda: not tab._deleting)
    assert len(threads) == 16 and threading.get_ident() not in threads
    reads.assert_not_called()
    assert [tab.table.item(i, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(tab.table.rowCount())] == [doc.id for doc in kept]
    assert [tab.table.item(i, 1).text() for i in range(len(kept))] == [str(i + 1) for i in range(len(kept))]
    assert [doc for doc in docs if state.store.video_dir(doc.id).exists()] == kept
    assert source.exists()


def test_delete_only_checked_videos_and_handle_locked_cache(qapp, videos, monkeypatch):
    state, tab, docs, source = videos
    delete = state.store.delete_video_files

    def locked_cache(video_id):
        if video_id == docs[0].id:
            raise PermissionError("file is in use")
        delete(video_id)

    monkeypatch.setattr(state.store, "delete_video_files", locked_cache)
    for row in (0, 2):
        tab.table.item(row, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    tab.remove_button.click()
    wait_until(qapp, lambda: not tab._deleting)
    assert [ref.id for ref in state.project.videos] == [docs[1].id]
    assert state.store.video_dir(docs[1].id).exists()
    assert not state.store.video_dir(docs[2].id).exists()
    assert "1 thư mục chưa dọn" in tab.delete_status.text()
    assert source.exists()


@pytest.mark.parametrize("job_state", [PENDING, RUNNING, PAUSED])
def test_queue_locked_videos_cannot_be_deleted(videos, monkeypatch, job_state):
    state, tab, docs, _source = videos
    monkeypatch.setattr(state.jobs, "_start_next", lambda: None)
    state.jobs.add(Job(str(state.store.root), state.project.name, docs[0].id, docs[0].name, ["tts"], state=job_state))
    errors = []
    monkeypatch.setattr(projects_module, "error", lambda *args: errors.append(args[1]))
    tab.select_all_button.click()
    tab.remove_button.click()
    assert errors and len(state.project.videos) == 3
    assert all(state.store.video_dir(doc.id).exists() for doc in docs)


def test_failed_metadata_save_leaves_project_and_editor_untouched(videos, monkeypatch):
    state, tab, docs, _source = videos
    monkeypatch.setattr(state.store, "save_project", Mock(side_effect=PermissionError("read only")))
    before = list(state.project.videos)
    closed, errors = [], []
    state.videosAboutToBeRemoved.connect(closed.append)
    monkeypatch.setattr(projects_module, "error", lambda *args: errors.append(args[1]))
    tab.select_all_button.click()
    tab.remove_button.click()
    assert state.project.videos == before and not closed and errors
    assert all(state.store.video_dir(doc.id).exists() for doc in docs)


def test_invalid_video_path_cannot_delete_outside_data(tmp_path):
    store, project = ProjectStore.create(tmp_path, "Project")
    outside = store.root / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    project.videos.append(VideoRef(id="../outside", name="Invalid"))
    with pytest.raises(ValueError):
        store.remove_videos(project, ["../outside"])
    assert len(project.videos) == 1 and (outside / "keep.txt").exists()


def test_delete_open_video_waits_for_preview_and_ignores_late_result(qapp, videos, monkeypatch):
    from reviewtrans.ui.pages import editor as editor_module

    state, tab, docs, _source = videos
    state.settings.player_backend = "qt"
    doc = docs[0]
    doc.source_path = "missing.mp4"
    doc.duration, doc.width, doc.height = 2, 320, 240
    doc.has_audio = True
    state.store.save_video(doc)
    state.store.save_segments(doc.id, [Segment(id=1, start=0, end=1, text="Test")])
    window = QtWidgets.QMainWindow()
    editor = editor_module.EditorPage(state, window)
    window.setCentralWidget(editor)
    entered, release = threading.Event(), threading.Event()

    def slow_mix(ctx, doc, segments, target):
        entered.set()
        release.wait(3)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"late preview")
        return target

    monkeypatch.setattr(editor_module, "run_mix", slow_mix)
    monkeypatch.setattr(editor_module, "error", lambda *args: pytest.fail(str(args)))
    editor.open_video(doc.id)
    editor.remix_timer.stop()
    editor._auto_remix()
    wait_until(qapp, entered.is_set)
    override = Mock(wraps=editor.backend.set_audio_override)
    monkeypatch.setattr(editor.backend, "set_audio_override", override)
    editor.mark_dirty("render")
    tab.refresh()
    tab.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    try:
        tab.remove_button.click()
        assert editor.doc is None and editor.canvas.doc is None
        assert not editor.save_timer.isActive() and not editor.remix_timer.isActive()
        assert editor.backend.ext_player.source().isEmpty() and editor.clip_player.source().isEmpty()
        assert tab._deleting and state.store.video_dir(doc.id).exists()
    finally:
        release.set()
        wait_until(qapp, lambda: not tab._deleting)
        editor.shutdown()
        window.close()
    assert not override.called
    assert not state.store.video_dir(doc.id).exists()
    editor.save_now()
    assert not state.store.video_dir(doc.id).exists()


def test_mpv_stop_and_reopen_use_async_commands_and_ignore_old_callbacks(qapp, monkeypatch):
    from reviewtrans.ui.player import mpv_backend

    player = Mock()
    callbacks = {}
    commands = []
    player.property_observer.side_effect = lambda name: lambda fn: fn

    def command_async(*args):
        reply = Future()
        commands.append((args, reply))
        return reply

    player.command_async.side_effect = command_async

    def event(name):
        def register(fn):
            callbacks[name] = fn
            return fn
        return register

    player.event_callback.side_effect = event
    monkeypatch.setattr(mpv_backend.mpv, "MPV", lambda **kwargs: player)
    backend = mpv_backend.MpvPlayerBackend()
    try:
        backend.load("old.mp4")
        backend.set_audio_override("old.wav")
        release = backend.unload()
        backend.load("new.mp4")
        assert len(commands) == 1 and not release.done()
        for index, expected in enumerate([
            ("set", "pause", "yes"), ("loadfile", "old.mp4", "replace"), ("stop",),
            ("set", "pause", "yes"), ("loadfile", "new.mp4", "replace"),
        ]):
            args, reply = commands[index]
            assert args == expected
            reply.set_result(None)
            qapp.processEvents()
        assert release.result(timeout=0) is None

        player.path = "old.mp4"
        callbacks["file-loaded"](None)
        qapp.processEvents()
        assert not backend._loaded
        player.path = "new.mp4"
        callbacks["file-loaded"](None)
        qapp.processEvents()
        assert backend._loaded
        assert backend._override is None and backend._pending_override is None
        player.command.assert_not_called()
    finally:
        backend.close()


def test_add_and_open_new_video_while_deleted_player_is_releasing_files(qapp, videos, monkeypatch):
    from reviewtrans.ui.pages.editor import EditorPage

    state, tab, docs, source = videos
    docs.extend(state.store.add_video(state.project, str(source)) for _ in range(12))
    tab.refresh()
    state.settings.player_backend = "qt"
    window = QtWidgets.QMainWindow()
    container = QtWidgets.QWidget(window)
    window.setCentralWidget(container)
    layout = QtWidgets.QVBoxLayout(container)
    layout.addWidget(tab)
    editor = EditorPage(state, container)
    layout.addWidget(editor)
    release = Future()
    load = Mock()
    monkeypatch.setattr(editor.backend, "load", load)
    monkeypatch.setattr(editor.backend, "unload", lambda: release)
    monkeypatch.setattr(projects_module, "probe", lambda _path: MediaInfo(duration=2, width=320, height=240))
    editor.open_video(docs[0].id)
    try:
        tab.select_all()
        tab.remove_button.click()
        assert editor.doc is None and tab._deleting
        tab.add_videos()
        dialog = tab._video_picker
        assert dialog.testOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        dialog.setDirectory(str(source.parent))
        dialog.selectFile(source.name)
        QtCore.QMetaObject.invokeMethod(dialog, "accept")
        wait_until(qapp, lambda: editor.doc is not None)
        assert tab._video_picker is None
        assert editor.doc.id not in {doc.id for doc in docs}
        assert state.project.videos[0].id == editor.doc.id
        assert load.call_count == 2
        assert state.store.video_dir(docs[0].id).exists()
        assert tab._deleting
    finally:
        release.set_result(None)
        wait_until(qapp, lambda: not tab._deleting)
        editor.shutdown()
        tab.setParent(None)
        window.close()
    assert all(not state.store.video_dir(doc.id).exists() for doc in docs)
    assert source.exists()
