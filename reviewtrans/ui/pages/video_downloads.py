from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from ...core.video_downloads import (
    HONGGUO_HOME, Media, detail_url, download_video, episode_links, http_url, is_hongguo,
    player_url, series_id,
)
from ..jobs import CANCELLED, DONE, ERROR, PENDING, RUNNING, FunctionThread
from ..widgets.common import open_path, push_button, title_label
from ..widgets.hongguo_browser import HongguoBrowser

_RESOLVING = "resolving"
_STOPPING = "stopping"
_STATUS = {
    PENDING: "Đang chờ", _RESOLVING: "Đang tìm nguồn", RUNNING: "Đang tải",
    _STOPPING: "Đang hủy", DONE: "Đã tải", ERROR: "Lỗi", CANCELLED: "Đã hủy",
}
_PARALLEL_DOWNLOADS = 3


@dataclass
class _Download:
    url: str
    label: str
    folder: Path
    state: str = PENDING
    progress: int = 0
    message: str = ""
    path: Path | None = None


class VideoDownloadsPage(QtWidgets.QWidget):
    importRequested = QtCore.pyqtSignal(list)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._browser: HongguoBrowser | None = None
        self._downloads: list[_Download] = []
        self._workers: dict[int, FunctionThread] = {}
        self._resolving: int | None = None
        self._closing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addWidget(title_label("Tải video"))
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_browse(), "Browse")
        self.tabs.addTab(self._build_links(), "Links")
        self.tabs.addTab(self._build_downloads(), "Download")
        layout.addWidget(self.tabs, 1)
        state.projectOpened.connect(self._update_actions)
        self._update_actions()

    def _build_browse(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        self._browse_layout = layout = QtWidgets.QVBoxLayout(tab)
        toolbar = QtWidgets.QHBoxLayout()
        self._home = push_button("Trang chủ", lambda: self._open_browser(HONGGUO_HOME), "projects")
        self._back = push_button("Quay lại", icon_name="prev")
        self._forward = push_button("Tiến", icon_name="next")
        self._reload = push_button("Tải lại", icon_name="refresh")
        self._choose = push_button("Chọn tập để tải", self._choose_episodes, "download", "Primary")
        for button in (self._home, self._back, self._forward, self._reload):
            toolbar.addWidget(button)
        toolbar.addStretch()
        toolbar.addWidget(self._choose)
        layout.addLayout(toolbar)
        self._browse_status = QtWidgets.QLabel("Tìm phim trên Hongguo, mở trang phim rồi bấm Chọn tập để tải.")
        self._browse_status.setWordWrap(True)
        self._browse_status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self._browse_status)
        return tab

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._ensure_browser()

    def _ensure_browser(self) -> HongguoBrowser:
        if self._browser is None:
            browser = HongguoBrowser(self.tabs.widget(0))
            # Adding the view can dispatch another show event before this call returns.
            self._browser = browser
            browser.attach_view(self._browse_layout)
            view = browser.view
            if view is not None:
                self._back.clicked.connect(view.back)
                self._forward.clicked.connect(view.forward)
                self._reload.clicked.connect(view.reload)
                view.urlChanged.connect(lambda _url: self._update_browser())
                view.loadFinished.connect(self._browser_loaded)
                browser.open(HONGGUO_HOME)
            self._update_browser()
        return self._browser

    def _update_browser(self) -> None:
        view = self._browser.view
        self._home.setEnabled(view is not None)
        self._reload.setEnabled(view is not None)
        self._back.setEnabled(bool(view and view.history().canGoBack()))
        self._forward.setEnabled(bool(view and view.history().canGoForward()))
        identifier = series_id(view.url().toString()) if view is not None else None
        self._choose.setEnabled(bool(identifier))

    def _browser_loaded(self, ok: bool) -> None:
        self._update_browser()
        self._browse_status.setText(
            "Tìm phim trên Hongguo, mở trang phim rồi bấm Chọn tập để tải."
            if ok else "Không mở được trang Hongguo. Kiểm tra kết nối rồi bấm Tải lại."
        )

    def _open_browser(self, url: str) -> None:
        self.tabs.setCurrentIndex(0)
        self._ensure_browser().open(url)

    def _choose_episodes(self) -> None:
        browser = self._ensure_browser()
        if browser.view is None:
            return
        url = browser.view.url().toString()
        identifier = series_id(url)
        if not identifier:
            return
        if player_url(url):
            self._open_browser(detail_url(identifier))
            self._browse_status.setText("Đang mở trang phim. Bấm Chọn tập để tải khi danh sách hiện ra.")
            return
        browser.read_episodes(lambda result: self._show_episodes(url, identifier, result))

    def _show_episodes(self, source: str, identifier: str, result) -> None:
        if self._closing or not isinstance(result, dict) or self.tabs.currentIndex() != 0:
            return
        if self._browser.view.url().toString() != source or result.get("page") != source:
            return
        episodes = episode_links(result.get("links", []), identifier)
        if not episodes:
            self._browse_status.setText("Trang hiện chưa cung cấp liên kết phát tập nào.")
            return
        title = result.get("title") or f"Hongguo {identifier}"
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Chọn tập để tải")
        dialog.resize(440, 390)
        body = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(title)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        body.addWidget(label)
        body.addWidget(QtWidgets.QLabel("Các tập có liên kết phát trên trang web"))
        choices = QtWidgets.QListWidget()
        checked, unchecked = QtCore.Qt.CheckState.Checked, QtCore.Qt.CheckState.Unchecked
        for episode in episodes:
            item = QtWidgets.QListWidgetItem(f"Tập {episode.number:02}", choices)
            item.setCheckState(unchecked)
        body.addWidget(choices, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        all_button = buttons.addButton("Chọn tất cả", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        add_button = buttons.addButton("Tải các tập đã chọn", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        body.addWidget(buttons)

        def selected() -> list:
            return [episode for i, episode in enumerate(episodes) if choices.item(i).checkState() == checked]

        add_button.setEnabled(False)
        choices.itemChanged.connect(lambda _item: add_button.setEnabled(bool(selected())))

        def check_all() -> None:
            for i in range(choices.count()):
                choices.item(i).setCheckState(checked)

        all_button.clicked.connect(check_all)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._enqueue([(episode.url, f"{title} - Tập {episode.number:02}") for episode in selected()])
        dialog.deleteLater()

    def _build_links(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        label = QtWidgets.QLabel("Dán mỗi dòng một liên kết phim Hongguo, tập Hongguo hoặc tệp video trực tiếp.")
        layout.addWidget(label)
        self._links = QtWidgets.QPlainTextEdit()
        self._links.setPlaceholderText(f"{HONGGUO_HOME}detail?series_id=…")
        self._links.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._links, 1)
        action = QtWidgets.QHBoxLayout()
        self._link_status = QtWidgets.QLabel()
        self._link_status.setWordWrap(True)
        action.addWidget(self._link_status, 1)
        action.addWidget(push_button("Xử lý liên kết", self._submit_links, "download", "Primary"))
        layout.addLayout(action)
        return tab

    def _submit_links(self) -> None:
        urls = list(dict.fromkeys(line.strip() for line in self._links.toPlainText().splitlines() if line.strip()))
        entries, remaining = [], []
        film = None
        for url in urls:
            player = player_url(url)
            if player:
                entries.append((player, f"Hongguo - {player.rsplit('/', 1)[-1]}"))
            elif is_hongguo(url) and series_id(url) and film is None:
                film = detail_url(series_id(url))
            elif http_url(url) and not is_hongguo(url):
                entries.append((url, ""))
            else:
                remaining.append(url)
        self._links.setPlainText("\n".join(remaining))
        message = "" if urls else "Hãy dán liên kết trước."
        if remaining:
            message = f"Còn {len(remaining)} dòng. Mở từng link phim hoặc kiểm tra lại địa chỉ."
        self._link_status.setText(message)
        if entries:
            self._enqueue(entries)
        if film:
            self._open_browser(film)

    def _build_downloads(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        toolbar = QtWidgets.QHBoxLayout()
        location = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DownloadLocation)
        self._folder = Path(location or Path.home()) / "ReviewTrans"
        self._folder_field = QtWidgets.QLineEdit(str(self._folder))
        self._folder_field.setReadOnly(True)
        self._folder_field.setAccessibleName("Thư mục lưu video")
        toolbar.addWidget(QtWidgets.QLabel("Lưu vào"))
        toolbar.addWidget(self._folder_field, 1)
        toolbar.addWidget(push_button("Đổi thư mục", self._change_folder, "folder"))
        toolbar.addWidget(push_button("Mở thư mục", self._open_folder, "open"))
        layout.addLayout(toolbar)

        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Video", "Trạng thái", "Tiến độ", "Chi tiết"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().hide()
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._update_actions)
        layout.addWidget(self._table, 1)
        actions = QtWidgets.QHBoxLayout()
        self._queue_status = QtWidgets.QLabel("Chưa có video trong hàng đợi.")
        actions.addWidget(self._queue_status, 1)
        self._cancel_button = push_button("Hủy mục đã chọn", self._cancel_selected, "stop")
        self._retry_button = push_button("Thử lại", self._retry_selected, "refresh")
        self._import_button = push_button("Thêm vào Project", self._import_selected, "add", "Primary")
        for button in (self._cancel_button, self._retry_button, self._import_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        return tab

    def _change_folder(self) -> None:
        start = self._folder if self._folder.is_dir() else self._folder.parent
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu video", str(start))
        if path:
            self._folder = Path(path)
            self._folder_field.setText(path)

    def _open_folder(self) -> None:
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            open_path(str(self._folder))
        except OSError as exc:
            self._queue_status.setText(str(exc))

    def _enqueue(self, entries: list[tuple[str, str]]) -> None:
        existing = {job.url for job in self._downloads if job.folder == self._folder}
        added = 0
        for url, label in entries:
            if url in existing:
                continue
            existing.add(url)
            row = len(self._downloads)
            self._downloads.append(_Download(url, label, self._folder))
            self._table.insertRow(row)
            for column in range(self._table.columnCount()):
                self._table.setItem(row, column, QtWidgets.QTableWidgetItem())
            self._render(row)
            added += 1
        self._queue_status.setText(f"Đã thêm {added} video." if added else "Các liên kết đã có trong hàng đợi.")
        self.tabs.setCurrentIndex(2)
        self._pump()

    def _pump(self) -> None:
        if self._closing:
            return
        for row, job in enumerate(self._downloads):
            if len(self._workers) + (self._resolving is not None) >= _PARALLEL_DOWNLOADS:
                break
            if job.state != PENDING or row in self._workers:
                continue
            if player_url(job.url):
                if self._resolving is None:
                    self._resolving = row
                    job.state = _RESOLVING
                    self._render(row)
                    self._ensure_browser().resolve(job.url, lambda media, i=row: self._resolved(i, media))
            else:
                self._start(row, Media(job.url))

    def _resolved(self, row: int, media: Media | None) -> None:
        if self._closing or self._resolving != row:
            return
        self._resolving = None
        if media is None:
            self._failed(row, "Trang phát chưa cung cấp luồng tải. Kiểm tra tập trong Browse rồi thử lại.")
        else:
            self._start(row, media)
        self._pump()

    def _start(self, row: int, media: Media) -> None:
        job = self._downloads[row]
        job.state = RUNNING
        job.message = ""
        thread = FunctionThread(lambda progress, stop: download_video(media, job.folder, job.label, progress, stop), self)
        self._workers[row] = thread
        thread.progress.connect(lambda progress, message: self._progress(row, progress, message))
        thread.done.connect(lambda path: self._finished(row, path))
        thread.failed.connect(lambda message: self._failed(row, message))
        thread.finished.connect(lambda: self._release_worker(row))
        self._render(row)
        thread.start()

    def _progress(self, row: int, value: float, message: str) -> None:
        job = self._downloads[row]
        if job.state == RUNNING:
            job.progress = min(99, int(value))
            job.message = message
            self._render(row)

    def _finished(self, row: int, path: Path) -> None:
        job = self._downloads[row]
        job.state, job.path, job.progress, job.message = DONE, path, 100, str(path)
        self._render(row)

    def _failed(self, row: int, message: str) -> None:
        job = self._downloads[row]
        job.state = CANCELLED if job.state == _STOPPING else ERROR
        job.message = message
        self._render(row)

    def _release_worker(self, row: int) -> None:
        self._workers.pop(row).deleteLater()
        self._pump()

    def _render(self, row: int) -> None:
        job = self._downloads[row]
        name = job.label or QtCore.QUrl(job.url).fileName() or QtCore.QUrl(job.url).host()
        values = (name, _STATUS[job.state], f"{job.progress}%", job.message)
        for column, value in enumerate(values):
            item = self._table.item(row, column)
            item.setText(value)
            item.setToolTip(value)
        self._update_actions()

    def _selected_rows(self) -> list[int]:
        return sorted(index.row() for index in self._table.selectionModel().selectedRows())

    def _update_actions(self) -> None:
        states = {self._downloads[row].state for row in self._selected_rows()}
        self._cancel_button.setEnabled(bool(states.intersection({PENDING, RUNNING, _RESOLVING})))
        self._retry_button.setEnabled(bool(states.intersection({ERROR, CANCELLED})))
        self._import_button.setEnabled(DONE in states and self.state.project is not None)

    def _cancel_selected(self) -> None:
        for row in self._selected_rows():
            job = self._downloads[row]
            if job.state == RUNNING:
                job.state = _STOPPING
                self._workers[row].stop_event.set()
            elif job.state in {PENDING, _RESOLVING}:
                if self._resolving == row:
                    self._browser.cancel()
                    self._resolving = None
                job.state = CANCELLED
            self._render(row)
        self._pump()

    def _retry_selected(self) -> None:
        for row in self._selected_rows():
            job = self._downloads[row]
            if job.state in {ERROR, CANCELLED}:
                job.state, job.progress, job.message = PENDING, 0, ""
                self._render(row)
        self._pump()

    def _import_selected(self) -> None:
        paths = [self._downloads[row].path for row in self._selected_rows()]
        existing = [str(path) for path in paths if path is not None and path.is_file()]
        if existing:
            self.importRequested.emit(existing)
        elif paths:
            self._queue_status.setText("Tệp đã chọn không còn trong thư mục lưu.")

    def is_busy(self) -> bool:
        return bool(self._workers) or self._resolving is not None

    def shutdown(self) -> None:
        self._closing = True
        if self._browser is not None:
            self._browser.cancel()
        for worker in self._workers.values():
            worker.stop_event.set()
        for worker in self._workers.values():
            worker.wait()
