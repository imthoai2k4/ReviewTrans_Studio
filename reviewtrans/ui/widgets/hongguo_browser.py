from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from ...core.video_downloads import HONGGUO_HOME, hongguo_media, is_hongguo, player_url

try:
    from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None


_EPISODES = """
({
    page: location.href,
    title: document.querySelector('h1')?.textContent.trim() || '',
    links: Array.from(document.querySelectorAll('a[href*="/player/"]'),
                      a => [(a.innerText || a.textContent).trim(), a.href])
})
"""

_MEDIA = """
(() => {
    const players = Array.from(document.querySelectorAll('video'));
    if (!players.length || players.some(video => video.mediaKeys)) return null;
    return {
        page: location.href,
        urls: [
            ...players.map(video => video.currentSrc || video.src),
            ...performance.getEntriesByType('resource').map(entry => entry.name)
        ].filter(url => url.includes('.qznovelvod.com/'))
    };
})()
"""


if QWebEngineView is not None:
    class _HongguoPage(QWebEnginePage):
        def __init__(self, profile, parent):
            super().__init__(profile, parent)
            settings = self.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, False)
            settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
            self.newWindowRequested.connect(self._open_in_place)

        def acceptNavigationRequest(self, url, _kind, is_main_frame):  # noqa: N802
            return not is_main_frame or (url.scheme() == "https" and is_hongguo(url.toString()))

        def _open_in_place(self, request) -> None:
            if request.isUserInitiated() and is_hongguo(request.requestedUrl().toString()):
                request.openIn(self)


class HongguoBrowser(QtCore.QObject):
    """One visible browser and one short-lived player document per chosen episode."""

    def __init__(self, layout: QtWidgets.QVBoxLayout, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.view = None
        self._player = None
        self._result = None
        self._reading = False
        self._sample_timer = QtCore.QTimer(self, interval=450)
        self._sample_timer.timeout.connect(self._sample)
        self._deadline = QtCore.QTimer(self, singleShot=True, interval=18000)
        self._deadline.timeout.connect(lambda: self._complete(None))
        if QWebEngineView is None:
            label = QtWidgets.QLabel("Cần cài PyQt6-WebEngine để mở Hongguo trong ReviewTrans.")
            label.setWordWrap(True)
            layout.addWidget(label)
            layout.addStretch()
            return

        self.view = QWebEngineView(parent)
        layout.addWidget(self.view, 1)
        self._profile = QWebEngineProfile(parent)
        self.view.setPage(_HongguoPage(self._profile, self.view))
        self.view.setUrl(QtCore.QUrl(HONGGUO_HOME))

    def open(self, url: str) -> None:
        if self.view is not None:
            self.view.setUrl(QtCore.QUrl(url))

    def read_episodes(self, callback) -> None:
        if self.view is not None:
            self.view.page().runJavaScript(_EPISODES, callback)

    def resolve(self, url: str, callback) -> None:
        if self.view is None:
            QtCore.QTimer.singleShot(0, lambda: callback(None))
            return
        self._result = callback
        self._player = page = _HongguoPage(self._profile, self)
        page.setAudioMuted(True)
        page.setVisible(True)
        self._source = url
        self._reading = False
        page.loadFinished.connect(lambda ok: self._load_finished(page, ok))
        self._deadline.start()
        self._sample_timer.start()
        page.load(QtCore.QUrl(url))

    def _load_finished(self, page, ok: bool) -> None:
        if self._player is not page:
            return
        if ok:
            self._sample()
        else:
            self._complete(None)

    def _sample(self) -> None:
        page = self._player
        if page is None or self._reading:
            return
        self._reading = True
        page.runJavaScript(_MEDIA, lambda result: self._read_media(page, result))

    def _read_media(self, page, result) -> None:
        if self._player is not page:
            return
        self._reading = False
        if isinstance(result, dict) and player_url(result.get("page", "")) == self._source:
            media = hongguo_media(result.get("urls", []), self._source)
            if media:
                self._complete(media)

    def cancel(self) -> None:
        self._deadline.stop()
        self._sample_timer.stop()
        self._result = None
        page, self._player = self._player, None
        if page is not None:
            page.triggerAction(QWebEnginePage.WebAction.Stop)
            page.deleteLater()

    def _complete(self, media) -> None:
        callback = self._result
        self.cancel()
        if callback is not None:
            callback(media)
