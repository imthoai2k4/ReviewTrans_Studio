from __future__ import annotations

import time

from PyQt6 import QtCore, QtGui, QtWidgets

from ...core.pipeline.runner import STEP_LABELS
from ...core.resolve import resolve
from ..jobs import CANCELLED, DONE, ERROR, JOB_STATE_LABELS, PENDING, RUNNING, STOPPED, Job
from ..state import AppState
from ..theme import ACCENT, DANGER, SUCCESS, TEXT_DIM, WARNING
from ..widgets.common import LogView, info, push_button, title_label
from ..widgets.queue_run import QueueRunDialog

COLORS = {PENDING: TEXT_DIM, RUNNING: ACCENT, DONE: SUCCESS, ERROR: DANGER, STOPPED: WARNING, CANCELLED: TEXT_DIM}


class QueuePage(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.queue = state.jobs
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(title_label("Hàng đợi xử lý", "Các video chạy lần lượt theo thứ tự thêm vào; ngữ cảnh project được cập nhật sau mỗi video."), 1)
        header.addWidget(push_button("Dừng việc đang chạy", self.queue.stop_current, "stop", "Danger"))
        header.addWidget(push_button("Huỷ tất cả", self.queue.cancel_all, "delete"))
        header.addWidget(push_button("Xoá việc đã xong", self._clear, "refresh"))
        layout.addLayout(header)

        launch = QtWidgets.QHBoxLayout()
        self.run_selected_button = push_button("Chạy video đã chọn…", lambda: self._run_project(False), "run")
        self.run_all_button = push_button("Chạy tất cả video…", lambda: self._run_project(True), "run", "Primary")
        for button in (self.run_selected_button, self.run_all_button):
            launch.addWidget(button)
        launch.addStretch()
        layout.addLayout(launch)
        state.projectOpened.connect(self._update_run_buttons)
        state.projectChanged.connect(self._update_run_buttons)
        self._update_run_buttons()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["#", "Project", "Video", "Các bước", "Trạng thái", "Tiến độ", "Thời gian", "Thao tác"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(5, 220)
        self.table.setColumnWidth(7, 183)
        self.table.itemSelectionChanged.connect(self._show_log)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        splitter.addWidget(self.table)
        log_box = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_title = QtWidgets.QLabel("Log")
        self.log_title.setStyleSheet(f"color: {TEXT_DIM};")
        log_layout.addWidget(self.log_title)
        self.log = LogView()
        log_layout.addWidget(self.log)
        splitter.addWidget(log_box)
        splitter.setSizes([300, 400])
        layout.addWidget(splitter, 1)

        self.rows: dict[int, int] = {}
        self.bars: dict[int, QtWidgets.QProgressBar] = {}
        self.queue.jobAdded.connect(self._add_job)
        self.queue.jobChanged.connect(self._update_job)
        self.queue.jobLog.connect(self._on_log)
        self.queue.videoLocksChanged.connect(self._update_recovery_actions)
        for job in self.queue.jobs:
            self._add_job(job)
        self.timer = QtCore.QTimer(self, interval=1000, timeout=self._tick)
        self.timer.start()

    def _add_job(self, job: Job) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.rows[job.id] = row
        for col, text in enumerate([str(job.id), job.project_name, job.video_name, job.steps_label, "", "", "", ""]):
            item = QtWidgets.QTableWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, job.id)
            self.table.setItem(row, col, item)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 1000)
        self.bars[job.id] = bar
        self.table.setCellWidget(row, 5, bar)
        self._update_job(job)
        if self.table.currentRow() < 0:
            self.table.selectRow(row)

    def _update_job(self, job: Job) -> None:
        row = self.rows.get(job.id)
        if row is None:
            return
        state_item = self.table.item(row, 4)
        state_text = JOB_STATE_LABELS.get(job.state, job.state)
        if job.state == ERROR and job.step:
            state_text += f" · {STEP_LABELS[job.step]}"
        state_item.setText(state_text)
        state_item.setForeground(QtGui.QColor(COLORS.get(job.state, TEXT_DIM)))
        state_item.setToolTip(job.error)
        bar = self.bars[job.id]
        bar.setValue(int(job.progress * 10))
        bar.setFormat(f"{job.progress:.0f}% {job.message[:40]}")
        self.table.item(row, 6).setText(self._elapsed(job))
        self._update_recovery_action(job)

    def _update_run_buttons(self) -> None:
        has_videos = bool(self.state.project and self.state.project.videos)
        for button in (self.run_selected_button, self.run_all_button):
            button.setEnabled(has_videos)
            button.setToolTip("Chọn từ video của project đang mở." if has_videos else "Mở project có video để chạy.")

    def _run_project(self, all_videos: bool) -> None:
        project, store = self.state.project, self.state.store
        if not (project and store):
            return
        root = str(store.root)
        selected = self._selected_job()
        selected_ids = {selected.video_id} if selected and selected.project_root == root else set()
        latest = {job.video_id: job for job in self.queue.jobs if job.project_root == root}
        failed = {vid: job for vid, job in latest.items() if self.queue.can_retry_failed(job)}
        videos = [video for video in project.videos if not self.state.is_video_locked(video.id)]
        if not videos:
            info(self, "Các video của project đã có trong hàng đợi.")
            return
        dialog = QueueRunDialog(project.name, videos, selected_ids, failed, all_videos, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if self.state.store is not store:
            return

        steps, only_missing = dialog.steps(), dialog.only_missing.isChecked()
        for video_id in dialog.video_ids():
            if self.state.is_video_locked(video_id):
                continue
            job = failed.get(video_id)
            if job is not None:
                self._retry_failed(job)
            else:
                video_steps = steps
                if only_missing and "asr" in steps and store.load_segments(video_id):
                    video_steps = [step for step in steps if step != "asr"]
                if video_steps:
                    self.state.enqueue([video_id], video_steps, only_missing=only_missing)

    def _update_recovery_actions(self) -> None:
        for job in self.queue.jobs:
            if job.state == ERROR:
                self._update_recovery_action(job)

    def _update_recovery_action(self, job: Job) -> None:
        row = self.rows.get(job.id)
        if row is None:
            return
        col = self.table.columnCount() - 1
        button = self.table.cellWidget(row, col)
        if job.state != ERROR:
            if button is not None:
                self.table.removeCellWidget(row, col)
            return
        if button is None:
            button = QtWidgets.QToolButton()
            button.setText("Tiếp tục từ bước lỗi" if job.step else "Thử lại")
            button.clicked.connect(lambda: self._retry_failed(job))
            if job.step == "translate":
                menu = QtWidgets.QMenu(button)
                menu.addAction("Đổi API và dịch tiếp…", lambda: self._change_api_and_retry(job))
                button.setMenu(menu)
                button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            self.table.setCellWidget(row, col, button)
        button.setToolTip(job.error)
        button.setEnabled(self.queue.can_retry_failed(job))

    def _retry_failed(self, job: Job, profile_id: str = "") -> None:
        if self.state.store and str(self.state.store.root) == job.project_root:
            self.state.flushRequested.emit()
            self.state.save_project()
        self.queue.retry_failed(job, profile_id)

    def _change_api_and_retry(self, job: Job) -> None:
        if not self.queue.can_retry_failed(job):
            return
        store = self.state.store if self.state.store and str(self.state.store.root) == job.project_root else None
        if store:
            doc = store.load_video(job.video_id)
            current = resolve(self.state.settings, "translate", self.state.project, doc).profile
        else:
            current = None
        choices = [profile for profile in self.state.settings.translate_profiles if profile != current]
        if not choices:
            info(self, "Thêm một cấu hình dịch khác ở trang API rồi thử lại.")
            return
        names = [f"{i}. {profile.name} · {profile.model or profile.kind}" for i, profile in enumerate(choices, 1)]
        name, ok = QtWidgets.QInputDialog.getItem(
            self, "Đổi API và dịch tiếp", "API dịch mới cho video này", names, 0, False,
        )
        if ok:
            self._retry_failed(job, choices[names.index(name)].id)

    def _tick(self) -> None:
        current = self.queue.current()
        if current:
            row = self.rows.get(current.id)
            if row is not None:
                self.table.item(row, 6).setText(self._elapsed(current))

    @staticmethod
    def _elapsed(job: Job) -> str:
        seconds = int(job.elapsed())
        return time.strftime("%H:%M:%S", time.gmtime(seconds)) if seconds else ""

    def _selected_job(self) -> Job | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        job_id = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        return next((j for j in self.queue.jobs if j.id == job_id), None)

    def _show_log(self) -> None:
        job = self._selected_job()
        self.log.clear()
        if job:
            self.log_title.setText(f"Log — {job.video_name}")
            self.log.setPlainText("\n".join(job.logs))
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_log(self, job: Job, line: str) -> None:
        selected = self._selected_job()
        if selected is None or selected.id == job.id:
            if selected is None:
                row = self.rows.get(job.id)
                if row is not None:
                    self.table.selectRow(row)
                    return
            self.log.append_line(line)

    def _menu(self, pos) -> None:
        job = self._selected_job()
        if not job:
            return
        menu = QtWidgets.QMenu(self)
        if job.state in (PENDING, RUNNING):
            menu.addAction("Huỷ / dừng", lambda: self.queue.cancel(job))
        if job.state == ERROR:
            action = menu.addAction("Tiếp tục từ bước lỗi" if job.step else "Thử lại", lambda: self._retry_failed(job))
            action.setEnabled(self.queue.can_retry_failed(job))
        if job.state in (STOPPED, CANCELLED, DONE):
            menu.addAction("Chạy lại", lambda: self.queue.retry(job))
        menu.addAction("Chép log", lambda: QtWidgets.QApplication.clipboard().setText("\n".join(job.logs)))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _clear(self) -> None:
        self.queue.clear_finished()
        keep = {j.id for j in self.queue.jobs}
        for row in reversed(range(self.table.rowCount())):
            job_id = self.table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            if job_id not in keep:
                self.table.removeRow(row)
                self.bars.pop(job_id, None)
        self.rows = {self.table.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole): r for r in range(self.table.rowCount())}
        self.log.clear()
