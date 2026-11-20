from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from ...core.paths import fonts_dir, user_data_dir
from ...core.pipeline.funasr import funasr_models_dir, install_funasr, models_ready, runtime_dir, runtime_info, runtime_python
from ...core.pipeline.resources import (
    download_ffmpeg,
    download_libmpv,
    tool_status,
)
from ...core.store import ProjectStore
from ..jobs import run_background
from ..state import AppState
from ..theme import DANGER, SUCCESS, TEXT_DIM
from ..widgets.common import ComboBox, confirm, error, hint, human_size, open_path, push_button, title_label

OPTIONAL = set()
TOOLS = [
    ("ffmpeg", "FFmpeg + FFprobe", "Bắt buộc: tách âm, trộn âm, xuất video.", download_ffmpeg),
    ("libmpv", "libmpv (player)", "Player mượt, seek chính xác, xem trước blur thật. Không có sẽ dùng Qt Multimedia.", download_libmpv),
]


class ResourcesPage(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.threads = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(title_label("Tài nguyên", "Tải và quản lý công cụ, model nhận dạng giọng nói, dung lượng cache."))
        self.data_path_label = hint(f"Thư mục dữ liệu và tài nguyên: {user_data_dir()}")
        self.data_path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.data_path_label)

        tools_box = QtWidgets.QGroupBox("Công cụ")
        grid = QtWidgets.QGridLayout(tools_box)
        self.tool_labels: dict[str, QtWidgets.QLabel] = {}
        self.tool_progress: dict[str, QtWidgets.QProgressBar] = {}
        for row, (key, name, desc, fn) in enumerate(TOOLS):
            title = QtWidgets.QLabel(f"<b>{name}</b><br><span style='color:#9aa0a8'>{desc}</span>")
            title.setWordWrap(True)
            status = QtWidgets.QLabel()
            status.setWordWrap(True)
            status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            progress = QtWidgets.QProgressBar()
            progress.setRange(0, 100)
            progress.setVisible(False)
            button = push_button("Tải / cập nhật", lambda _=False, k=key, f=fn: self.download_tool(k, f), "download")
            grid.addWidget(title, row, 0)
            grid.addWidget(status, row, 1)
            grid.addWidget(progress, row, 2)
            grid.addWidget(button, row, 3)
            self.tool_labels[key] = status
            self.tool_progress[key] = progress
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        layout.addWidget(tools_box)

        paraformer_box = QtWidgets.QGroupBox("Paraformer-zh · Nhận dạng tiếng Trung")
        paraformer_layout = QtWidgets.QVBoxLayout(paraformer_box)
        paraformer_layout.addWidget(hint(
            "Paraformer-zh + FSMN-VAD + CT-Punc + CAM++: nhận dạng và chia lời thoại theo khoảng nghỉ, giọng nói. "
            "Cài một lần trong app; cần Internet và tối thiểu 8 GiB trống cho CPU hoặc 16 GiB cho CUDA, "
            "gồm dung lượng tạm. Không giữ cache tải pip. "
            "Bản portable lưu bộ nhận dạng và model trong data cạnh app. "
            "Tự động ưu tiên CUDA nếu driver NVIDIA phù hợp."
        ))
        self.paraformer_paths = QtWidgets.QLabel()
        self.paraformer_paths.setWordWrap(True)
        self.paraformer_paths.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        paraformer_layout.addWidget(self.paraformer_paths)
        paraformer_row = QtWidgets.QHBoxLayout()
        self.paraformer_status = QtWidgets.QLabel()
        self.paraformer_status.setWordWrap(True)
        paraformer_row.addWidget(self.paraformer_status, 1)
        self.paraformer_device = ComboBox([("auto", "Tự động CPU / CUDA"), ("cuda", "NVIDIA CUDA"), ("cpu", "CPU")])
        paraformer_row.addWidget(self.paraformer_device)
        self.paraformer_button = push_button("Cài bộ nhận dạng + tải model", self.install_paraformer, "download")
        paraformer_row.addWidget(self.paraformer_button)
        paraformer_layout.addLayout(paraformer_row)
        self.paraformer_progress = QtWidgets.QProgressBar()
        self.paraformer_progress.setVisible(False)
        paraformer_layout.addWidget(self.paraformer_progress)
        layout.addWidget(paraformer_box)

        layout.addStretch(1)
        storage = QtWidgets.QGroupBox("Lưu trữ")
        storage_layout = QtWidgets.QHBoxLayout(storage)
        self.cache_label = QtWidgets.QLabel()
        storage_layout.addWidget(self.cache_label, 1)
        storage_layout.addWidget(push_button("Tính dung lượng cache", self.compute_cache, "refresh"))
        storage_layout.addWidget(push_button("Dọn cache project (giữ lồng tiếng)", self.clear_cache, "delete"))
        storage_layout.addWidget(push_button("Thư mục dữ liệu", lambda: open_path(str(user_data_dir())), "folder"))
        storage_layout.addWidget(push_button("Thư mục font", lambda: open_path(str(fonts_dir())), "folder"))
        layout.addWidget(storage)

        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        self.paraformer_paths.setText(
            f"Nơi cài bộ nhận dạng: {runtime_dir()}\nNơi lưu model: {funasr_models_dir()}")
        python = runtime_python()
        info = runtime_info()
        ready = python.is_file() and bool(info.get("torch_version")) and models_ready()
        backend = f"CUDA {info['cuda_version']}" if info.get("cuda_version") else "CPU"
        if info.get("cuda_available") and info.get("gpu"):
            backend += f" · {info['gpu']}"
        self.paraformer_status.setText(
            f"Đã có 4 model · PyTorch {info.get('torch_version', '?')} · {backend}"
            if ready else "Chưa cài đủ bộ nhận dạng / model")
        self.paraformer_status.setStyleSheet(f"color: {SUCCESS if ready else TEXT_DIM};")
        status = tool_status()
        for key, _name, _desc, _fn in TOOLS:
            if key == "ffmpeg":
                ok = bool(status["ffmpeg"] and status["ffprobe"])
                text = status["ffmpeg"] if ok else "Chưa có"
            else:
                ok = bool(status[key])
                text = status[key] or "Chưa có"
            label = self.tool_labels[key]
            optional = key in OPTIONAL and not ok
            label.setText(("✓ " if ok else "– " if optional else "✕ ") + (text if not optional else "Chưa tải (tuỳ chọn)"))
            label.setStyleSheet(f"color: {SUCCESS if ok else TEXT_DIM if optional else DANGER};")
    def install_paraformer(self) -> None:
        self.paraformer_button.setEnabled(False)
        self.paraformer_device.setEnabled(False)
        self.paraformer_progress.setVisible(True)
        self.paraformer_progress.setValue(0)

        def finish(message: str) -> None:
            self.paraformer_button.setEnabled(True)
            self.paraformer_device.setEnabled(True)
            self.paraformer_progress.setVisible(False)
            self.refresh()
            if message:
                error(self, message, "Cài Paraformer thất bại")

        def progress(pct, message) -> None:
            self.paraformer_progress.setValue(int(pct))
            self.paraformer_status.setText(message)

        device = self.paraformer_device.value()
        run_background(lambda progress, stop: install_funasr(progress, stop, device=device),
                       lambda _result: finish(""), finish, progress)

    def download_tool(self, key: str, fn) -> None:
        bar = self.tool_progress[key]
        bar.setVisible(True)
        bar.setValue(0)

        def done(_result) -> None:
            bar.setVisible(False)
            self.refresh()
            note = " Khởi động lại ứng dụng để dùng player libmpv." if key == "libmpv" else ""
            self.window().statusBar().showMessage(f"Đã tải xong {key}.{note}", 8000)

        def fail(message: str) -> None:
            bar.setVisible(False)
            error(self, message, f"Tải {key} thất bại")

        run_background(lambda progress, stop: fn(progress, stop), done, fail, lambda pct, _m: bar.setValue(int(pct)))


    def compute_cache(self) -> None:
        store = self.state.store
        if not store:
            self.cache_label.setText("Mở một project để xem dung lượng cache.")
            return
        run_background(
            lambda _p, _s: store.cache_size(),
            lambda size: self.cache_label.setText(f"Cache project “{store.root.name}”: {human_size(size)}"),
        )

    def clear_cache(self) -> None:
        store: ProjectStore | None = self.state.store
        if not store:
            return
        if not confirm(self, "Dọn cache", "Xoá file tạm (âm thanh tách, bản trộn, file render) của project hiện tại? File lồng tiếng từng câu được giữ lại."):
            return
        store.clear_cache(keep_tts=True)
        self.compute_cache()

