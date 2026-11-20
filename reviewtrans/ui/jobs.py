from __future__ import annotations

import itertools
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from PyQt6 import QtCore

from ..core.config import SettingsStore
from ..core.models import BatchSettings, clone
from ..core.pipeline import RunContext
from ..core.pipeline.runner import STEP_LABELS, PipelineCheckpoint, VideoPipeline, ordered_steps
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
            if self.stop_event.is_set():
                raise StopRequested()
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
PAUSING, PAUSED = "pausing", "paused"
JOB_STATE_LABELS = {
    PENDING: "Chờ", RUNNING: "Đang chạy", DONE: "Xong", ERROR: "Lỗi", STOPPED: "Đã dừng", CANCELLED: "Đã huỷ",
    PAUSING: "Đang dừng", PAUSED: "Đã dừng",
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
    batch_settings: BatchSettings | None = None
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
    checkpoint: PipelineCheckpoint = field(default_factory=PipelineCheckpoint)
    settings_applied: bool = False
    paused_at: float = 0.0
    paused_duration: float = 0.0

    @property
    def steps_label(self) -> str:
        return " → ".join(STEP_LABELS[s] for s in ordered_steps(self.steps))

    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or self.paused_at or time.time()) - self.started_at - self.paused_duration


class _JobThread(QtCore.QThread):
    def __init__(self, queue: "JobQueue", job: Job) -> None:
        super().__init__()
        self.queue = queue
        self.job = job
        self.stop_event = threading.Event()
        self.pause_requested = False

    def run(self) -> None:
        job = self.job
        queue = self.queue
        store = ProjectStore(job.project_root)
        try:
            if self.stop_event.is_set():
                raise StopRequested()
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
            pipeline.checkpoint = job.checkpoint
            if not job.settings_applied and job.batch_settings and job.batch_settings.enabled:
                pipeline.doc.apply_batch_settings(job.batch_settings)
                pipeline.save()
            job.settings_applied = True
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
            queue._finish(job, PAUSED if self.pause_requested else STOPPED)
        except Exception as exc:  # noqa: BLE001
            queue._log(job, traceback.format_exc(limit=4))
            queue._finish(job, ERROR, str(exc) or type(exc).__name__)


class JobQueue(QtCore.QObject):
    jobAdded = QtCore.pyqtSignal(object)
    jobsRemoved = QtCore.pyqtSignal(object)  # job IDs
    jobChanged = QtCore.pyqtSignal(object)
    jobLog = QtCore.pyqtSignal(object, str)
    videoChanged = QtCore.pyqtSignal(str, str)  # project_root, video_id
    contextChanged = QtCore.pyqtSignal(str)
    busyChanged = QtCore.pyqtSignal(bool)
    pausedChanged = QtCore.pyqtSignal(bool)
    videoLocksChanged = QtCore.pyqtSignal()

    def __init__(self, settings_store: SettingsStore, parent=None) -> None:
        super().__init__(parent)
        self.settings_store = settings_store
        self.jobs: list[Job] = []
        self._thread: _JobThread | None = None
        self._last_emit = 0.0
        self._paused = False
        self._state_lock = threading.RLock()

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

    def is_paused(self) -> bool:
        return self._paused

    def locked_videos(self) -> set[tuple[str, str]]:
        return {(j.project_root, j.video_id) for j in self.jobs if j.state in (PENDING, RUNNING, PAUSING, PAUSED)}

    def is_locked(self, project_root: str, video_id: str) -> bool:
        return (project_root, video_id) in self.locked_videos()

    def stop_current(self) -> None:
        if self._thread:
            self._thread.pause_requested = False
            self._thread.stop_event.set()

    def pause(self) -> None:
        if self._paused:
            return
        # Chặn việc kế tiếp trước khi báo dừng thread hiện tại.
        self._paused = True
        with self._state_lock:
            if self._thread and self._thread.job.state == RUNNING:
                self._thread.pause_requested = True
                self._thread.job.state = PAUSING
                self._thread.stop_event.set()
                self._log(self._thread.job, "Đang dừng; giữ tiến trình để tiếp tục.")
                self.jobChanged.emit(self._thread.job)
        self.pausedChanged.emit(True)

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        self.pausedChanged.emit(False)
        # Không xoá stop_event cũ: chờ thread dừng hẳn rồi tiếp tục bằng thread mới.
        self._start_next()

    def cancel(self, job: Job) -> None:
        if job.state == PENDING:
            job.state = CANCELLED
            self.jobChanged.emit(job)
            self.videoLocksChanged.emit()
        elif job.state in (RUNNING, PAUSING):
            self.stop_current()
        elif job.state == PAUSED:
            job.state = CANCELLED
            job.finished_at = job.paused_at
            self.jobChanged.emit(job)
            self.videoLocksChanged.emit()

    def cancel_all(self) -> None:
        self._paused = True
        for job in self.jobs:
            if job.state in (PENDING, PAUSED):
                job.state = CANCELLED
                self.jobChanged.emit(job)
        self.stop_current()
        self.pausedChanged.emit(True)
        self.videoLocksChanged.emit()

    def can_remove(self, job: Job) -> bool:
        # Giữ việc hiện tại đến khi thread đã dừng hẳn, kể cả khi vừa báo kết thúc.
        return job is not self.current() and job.state not in (RUNNING, PAUSING)

    def remove(self, job_ids: list[int]) -> list[int]:
        """Chỉ gỡ việc khỏi hàng đợi, không xoá video hoặc dữ liệu project."""
        selected = set(job_ids)
        with self._state_lock:
            removed = [j.id for j in self.jobs if j.id in selected and self.can_remove(j)]
            if not removed:
                return []
            removed_ids = set(removed)
            self.jobs = [j for j in self.jobs if j.id not in removed_ids]
        self.jobsRemoved.emit(removed)
        self.videoLocksChanged.emit()
        return removed

    def clear_finished(self) -> None:
        self.remove([j.id for j in self.jobs if j.state not in (PENDING, RUNNING, PAUSING, PAUSED)])

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
        job.paused_duration = 0.0
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
            batch_settings=clone(job.batch_settings),
        ))

    def shutdown(self) -> None:
        self.cancel_all()
        if self._thread:
            self._thread.wait(5000)

    # ------------------------------------------------------------ nội bộ

    def _start_next(self) -> None:
        if self._thread is not None:
            return
        if self._paused:
            self.busyChanged.emit(False)
            return
        job = next((j for j in self.jobs if j.state == PAUSED), None)
        if job is None:
            job = next((j for j in self.jobs if j.state == PENDING), None)
        if job is None:
            self.busyChanged.emit(False)
            return
        now = time.time()
        if job.paused_at:
            job.paused_duration += now - job.paused_at
            job.paused_at = 0.0
            self._log(job, f"Tiếp tục từ bước {STEP_LABELS.get(job.step, job.step) or 'đầu tiên'}.")
        job.state = RUNNING
        if not job.started_at:
            job.started_at = now
        thread = _JobThread(self, job)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self.jobChanged.emit(job)
        self.busyChanged.emit(True)
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
        with self._state_lock:
            job.state = state
            job.error = error
            if state == PAUSED:
                job.paused_at = time.time()
                job.message = f"Đã dừng · {STEP_LABELS.get(job.step, job.step)}"
            else:
                job.finished_at = time.time()
            if state == DONE:
                job.progress = 100.0
                job.message = "Hoàn tất"
            elif error:
                job.message = error
        self._log(job, f"Kết thúc: {JOB_STATE_LABELS.get(state, state)} {error}".strip())
        self.jobChanged.emit(job)
