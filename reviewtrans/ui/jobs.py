from __future__ import annotations

import itertools
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from PyQt6 import QtCore

from ..core.config import SettingsStore
from ..core.pipeline import RunContext
from ..core.pipeline.runner import STEP_LABELS, VideoPipeline, ordered_steps
from ..core.proc import StopRequested
from ..core.store import ProjectStore

# ---------------------------------------------------------------- tác vụ nền đơn lẻ

_live_threads: set["FunctionThread"] = set()


class FunctionThread(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(float, str)

    def __init__(self, fn: Callable[..., Any], parent=None) -> None:
        super().__init__(parent)
        self.fn = fn
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            result = self.fn(lambda pct, msg="": self.progress.emit(float(pct), str(msg)), self.stop_event)
            self.done.emit(result)
        except StopRequested:
            self.failed.emit("Đã dừng")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)


def run_background(
    fn: Callable[[Callable[[float, str], None], threading.Event], Any],
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> FunctionThread:
    """Chạy fn(progress, stop_event) ở thread nền, callback trả về main thread."""
    thread = FunctionThread(fn)
    if on_done:
        thread.done.connect(on_done)
    if on_error:
        thread.failed.connect(on_error)
    if on_progress:
        thread.progress.connect(on_progress)
    _live_threads.add(thread)
    thread.finished.connect(lambda: _live_threads.discard(thread))
    thread.start()
    return thread


# ---------------------------------------------------------------- hàng đợi pipeline

_ids = itertools.count(1)

PENDING, RUNNING, DONE, ERROR, STOPPED, CANCELLED = "pending", "running", "done", "error", "stopped", "cancelled"
JOB_STATE_LABELS = {
    PENDING: "Chờ", RUNNING: "Đang chạy", DONE: "Xong", ERROR: "Lỗi", STOPPED: "Đã dừng", CANCELLED: "Đã huỷ",
}


@dataclass
class Job:
    project_root: str
    project_name: str
    video_id: str
    video_name: str
    steps: list[str]
    only_missing: bool = True
    force_context: bool = False
    id: int = field(default_factory=lambda: next(_ids))
    state: str = PENDING
    progress: float = 0.0
    step: str = ""
    message: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    logs: list[str] = field(default_factory=list)
    retry_from: str = ""
    force_asr: bool = False
    retry_translate_profile: str = ""

    @property
    def steps_label(self) -> str:
        return " → ".join(STEP_LABELS[s] for s in ordered_steps(self.steps))

    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at


class _JobThread(QtCore.QThread):
    def __init__(self, queue: "JobQueue", job: Job) -> None:
        super().__init__()
        self.queue = queue
        self.job = job
        self.stop_event = threading.Event()

    def run(self) -> None:
        job = self.job
        queue = self.queue
        store = ProjectStore(job.project_root)
        try:
            project = store.load_project()
            ctx = RunContext(
                store=store,
                project=project,
                settings=queue.settings_store.snapshot(),
                log=lambda text: queue._log(job, text),
                progress=lambda pct, msg: queue._progress(job, pct, msg),
                video_changed=lambda vid: queue.videoChanged.emit(job.project_root, vid),
                context_changed=lambda: queue.contextChanged.emit(job.project_root),
                stop_event=self.stop_event,
            )
            pipeline = VideoPipeline(ctx, job.video_id)
            pipeline.force_asr = job.force_asr
            if job.retry_translate_profile:
                profile = ctx.settings.find_translate(job.retry_translate_profile)
                if profile is None:
                    raise ValueError("API dịch được chọn không còn trong cấu hình.")
                pipeline.doc.translate_profile = profile.id
                pipeline.doc.translate_model = ""
                pipeline.save()
                queue._log(job, f"Dùng API dịch {profile.name} cho phần còn lại của video.")
            steps = ordered_steps(job.steps)
            steps_by_label = {STEP_LABELS[step]: step for step in steps}
            last_progress_step = None

            def step_progress(pct: float, msg: str) -> None:
                nonlocal last_progress_step
                # quy đổi tiến độ từng bước ra tiến độ tổng của job
                current = job.step if job.step in steps else ""
                index = steps.index(current) if current else 0
                queue._progress(
                    job, (index + pct / 100.0) * 100.0 / max(1, len(steps)), msg,
                    force=current != last_progress_step,
                )
                last_progress_step = current

            ctx.progress = step_progress
            original_log = ctx.log

            def log_and_track(text: str) -> None:
                if text.startswith("── "):
                    step = steps_by_label.get(text.rsplit(": ", 1)[-1])
                    if step:
                        job.step = step
                original_log(text)

            ctx.log = log_and_track
            pipeline.run(
                steps, only_missing=job.only_missing, force_context=job.force_context, start_at=job.retry_from,
            )
            queue._finish(job, DONE)
        except StopRequested:
            queue._finish(job, STOPPED)
        except Exception as exc:  # noqa: BLE001
            queue._log(job, traceback.format_exc(limit=4))
            queue._finish(job, ERROR, str(exc) or type(exc).__name__)


class JobQueue(QtCore.QObject):
    jobAdded = QtCore.pyqtSignal(object)
    jobChanged = QtCore.pyqtSignal(object)
    jobLog = QtCore.pyqtSignal(object, str)
    videoChanged = QtCore.pyqtSignal(str, str)  # project_root, video_id
    contextChanged = QtCore.pyqtSignal(str)
    busyChanged = QtCore.pyqtSignal(bool)
    videoLocksChanged = QtCore.pyqtSignal()

    def __init__(self, settings_store: SettingsStore, parent=None) -> None:
        super().__init__(parent)
        self.settings_store = settings_store
        self.jobs: list[Job] = []
        self._thread: _JobThread | None = None
        self._last_emit = 0.0

    # ------------------------------------------------------------ API

    def add(self, job: Job) -> Job:
        self.jobs.append(job)
        self.jobAdded.emit(job)
        self.videoLocksChanged.emit()
        self._start_next()
        return job

    def is_busy(self) -> bool:
        return self._thread is not None

    def current(self) -> Job | None:
        return self._thread.job if self._thread else None

    def locked_videos(self) -> set[tuple[str, str]]:
        return {(j.project_root, j.video_id) for j in self.jobs if j.state in (PENDING, RUNNING)}

    def is_locked(self, project_root: str, video_id: str) -> bool:
        return (project_root, video_id) in self.locked_videos()

    def stop_current(self) -> None:
        if self._thread:
            self._thread.stop_event.set()

    def cancel(self, job: Job) -> None:
        if job.state == PENDING:
            job.state = CANCELLED
            self.jobChanged.emit(job)
            self.videoLocksChanged.emit()
        elif job.state == RUNNING:
            self.stop_current()

    def cancel_all(self) -> None:
        for job in self.jobs:
            if job.state == PENDING:
                job.state = CANCELLED
                self.jobChanged.emit(job)
        self.stop_current()
        self.videoLocksChanged.emit()

    def clear_finished(self) -> None:
        self.jobs = [j for j in self.jobs if j.state in (PENDING, RUNNING)]

    def can_retry_failed(self, job: Job) -> bool:
        return (
            job.state == ERROR and job is not self.current() and job in self.jobs
            and not self.is_locked(job.project_root, job.video_id)
        )

    def retry_failed(self, job: Job, translate_profile: str = "") -> Job | None:
        if not self.can_retry_failed(job) or (translate_profile and job.step != "translate"):
            return None
        job.retry_from = job.step
        job.retry_translate_profile = translate_profile
        label = STEP_LABELS.get(job.step, "bước đầu tiên")
        job.error = ""
        job.finished_at = 0.0
        job.started_at = 0.0
        job.message = f"Chờ tiếp tục: {label}"
        job.state = PENDING
        self._log(job, f"Đã xếp lại hàng đợi từ {label}.")
        self.jobChanged.emit(job)
        self.videoLocksChanged.emit()
        self._start_next()
        return job

    def retry(self, job: Job) -> Job | None:
        if job.state == ERROR:
            return self.retry_failed(job)
        return self.add(Job(
            project_root=job.project_root, project_name=job.project_name, video_id=job.video_id,
            video_name=job.video_name, steps=list(job.steps), only_missing=job.only_missing,
            force_context=job.force_context,
        ))

    def shutdown(self) -> None:
        self.cancel_all()
        if self._thread:
            self._thread.wait(5000)

    # ------------------------------------------------------------ nội bộ

    def _start_next(self) -> None:
        if self._thread is not None:
            return
        job = next((j for j in self.jobs if j.state == PENDING), None)
        if job is None:
            self.busyChanged.emit(False)
            return
        job.state = RUNNING
        job.started_at = time.time()
        self.jobChanged.emit(job)
        self.busyChanged.emit(True)
        thread = _JobThread(self, job)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        thread.start()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self.videoLocksChanged.emit()
        QtCore.QTimer.singleShot(0, self._start_next)

    def _log(self, job: Job, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        job.logs.append(line)
        if len(job.logs) > 5000:
            del job.logs[:1000]
        self.jobLog.emit(job, line)

    def _progress(self, job: Job, pct: float, message: str, force: bool = False) -> None:
        job.progress = max(0.0, min(100.0, pct))
        job.message = message
        now = time.time()
        if force or now - self._last_emit > 0.15 or pct >= 100:
            self._last_emit = now
            self.jobChanged.emit(job)

    def _finish(self, job: Job, state: str, error: str = "") -> None:
        job.state = state
        job.error = error
        job.finished_at = time.time()
        if state == DONE:
            job.progress = 100.0
            job.message = "Hoàn tất"
        elif error:
            job.message = error
        self._log(job, f"Kết thúc: {JOB_STATE_LABELS.get(state, state)} {error}".strip())
        self.jobChanged.emit(job)
