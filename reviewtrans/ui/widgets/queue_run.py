from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from ...core.pipeline.runner import PIPELINE_ORDER, STEP_LABELS
from .common import hint


class QueueRunDialog(QtWidgets.QDialog):
    def __init__(self, project_name, videos, selected_ids, failed_jobs, all_videos, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chạy tất cả video" if all_videos else "Chạy video đã chọn")
        self.resize(500, 460)
        self.all_videos = all_videos
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(hint(f"Project: {project_name}"))
        self.videos = QtWidgets.QListWidget()
        for video in videos:
            text = video.name
            if job := failed_jobs.get(video.id):
                text += f" · Tiếp tục từ {STEP_LABELS.get(job.step, 'bước đầu tiên')}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, video.id)
            if not all_videos:
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked if video.id in selected_ids else QtCore.Qt.CheckState.Unchecked
                )
            self.videos.addItem(item)
        layout.addWidget(self.videos, 1)

        steps_box = QtWidgets.QGroupBox("Các bước cho video chạy mới")
        steps_layout = QtWidgets.QGridLayout(steps_box)
        self.step_checks = {}
        for index, step in enumerate(PIPELINE_ORDER):
            check = QtWidgets.QCheckBox(STEP_LABELS[step])
            check.setChecked(True)
            steps_layout.addWidget(check, index // 2, index % 2)
            self.step_checks[step] = check
        self.only_missing = QtWidgets.QCheckBox("Chỉ phần còn thiếu")
        self.only_missing.setChecked(True)
        self.only_missing.setToolTip("Giữ câu thoại hiện có, chỉ dịch và lồng tiếng phần cần làm.")
        steps_layout.addWidget(self.only_missing, 3, 0, 1, 2)
        layout.addWidget(steps_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        run = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        run.setText("Đưa vào hàng đợi")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Hủy")
        layout.addWidget(buttons)

        def update_run():
            ids = self.video_ids()
            needs_steps = any(vid not in failed_jobs for vid in ids)
            steps_box.setEnabled(needs_steps)
            run.setEnabled(bool(ids) and (not needs_steps or bool(self.steps())))

        self.videos.itemChanged.connect(update_run)
        for check in self.step_checks.values():
            check.toggled.connect(update_run)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        update_run()

    def video_ids(self) -> list[str]:
        items = (self.videos.item(row) for row in range(self.videos.count()))
        return [
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in items
            if self.all_videos or item.checkState() == QtCore.Qt.CheckState.Checked
        ]

    def steps(self) -> list[str]:
        return [step for step, check in self.step_checks.items() if check.isChecked()]
