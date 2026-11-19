from __future__ import annotations

import threading
from pathlib import Path

import pytest

from reviewtrans.core import video_downloads as downloads
from reviewtrans.core.paths import find_tool
from reviewtrans.core.proc import probe, run_checked


_HOME = downloads.HONGGUO_HOME
_CDN = "https://v26-hgweb.qznovelvod.com/"


def test_episode_picker_uses_only_numbered_links_for_the_open_film():
    assert downloads.episode_links([
        ["播放正片", _HOME + "player/111"],
        ["2", _HOME + "player/111/222"],
        ["第 1 集", _HOME + "player/111"],
        ["10", _HOME + "player/222/999"],
        ["3", "https://other.invalid/player/111/333"],
        ["2", _HOME + "player/111/222?source=detail"],
    ], "111") == [downloads.Episode(1, _HOME + "player/111"), downloads.Episode(2, _HOME + "player/111/222")]


def test_player_media_requires_a_complete_pair_from_the_expected_cdn():
    page = _HOME + "player/111"
    video = _CDN + "video-avc1/item?mime_type=video_mp4"
    audio = _CDN + "audio-mp4a/item?mime_type=video_mp4"
    assert downloads.hongguo_media([video], page) is None
    assert downloads.hongguo_media([video, audio, f"{audio}&retry=1"], page) == downloads.Media(video, audio, page)
    assert downloads.hongguo_media([video.replace(".com/", ".com.invalid/"), audio], page) is None


class _Response:
    def __init__(self, data: bytes, kind="video/mp4", status=200):
        self.data = data
        self.status_code = status
        self.headers = {"Content-Type": kind, "Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.data), chunk_size):
            yield self.data[offset:offset + chunk_size]


@pytest.mark.skipif(not (find_tool("ffmpeg") and find_tool("ffprobe")), reason="cần FFmpeg và ffprobe")
def test_download_joins_real_tracks_without_overwriting_existing_output(tmp_path, monkeypatch):
    root = tmp_path / "out"
    root.mkdir()
    existing = root / "Tập 01.mp4"
    existing.write_bytes(b"existing")
    video, audio = tmp_path / "v.mp4", tmp_path / "a.m4a"
    ffmpeg = str(find_tool("ffmpeg"))
    for inputs, output in (
        (["testsrc2=size=64x64:rate=25:duration=1", "-c:v", "libx264"], video),
        (["sine=frequency=440:duration=1", "-c:a", "aac"], audio),
    ):
        run_checked([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", *inputs, str(output)], "tạo mẫu")
    sources = {"https://media.invalid/v.mp4": video, "https://media.invalid/a.m4a": audio}
    monkeypatch.setattr(downloads.requests.Session, "get", lambda _session, url, **_: _Response(sources[url].read_bytes()))

    output = downloads.download_video(
        downloads.Media(*sources), root, "Tập 01", lambda *_: None, threading.Event(),
    )
    assert output.name == "Tập 01 (2).mp4"
    info = probe(output)
    assert info.has_video and info.has_audio
    assert existing.read_bytes() == b"existing"
    assert sorted(root.iterdir()) == sorted([existing, output])


@pytest.mark.parametrize("kind,status", [("text/html", 200), ("video/mp4", 206), ("video/mp4", 403)])
def test_web_page_and_incomplete_or_denied_responses_leave_no_output(tmp_path, monkeypatch, kind, status):
    monkeypatch.setattr(downloads.requests.Session, "get", lambda *_args, **_kwargs: _Response(b"preview", kind, status))
    with pytest.raises(downloads.DownloadError):
        downloads.download_video(downloads.Media("https://example.invalid/media"), tmp_path, "Preview",
                                 lambda *_: None, threading.Event())
    assert not list(tmp_path.iterdir())


def test_download_reads_validated_http_ranges_in_order(tmp_path, monkeypatch):
    payload = b"0123456789"
    requests_seen = []

    def get(_session, _url, headers, **_kwargs):
        requests_seen.append(dict(headers))
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        end = min(end, len(payload) - 1)
        response = _Response(payload[start:end + 1], status=206)
        response.headers.update({"Content-Range": f"bytes {start}-{end}/{len(payload)}", "ETag": '"original"'})
        return response

    monkeypatch.setattr(downloads, "_RANGE_SIZE", 4)
    monkeypatch.setattr(downloads.requests.Session, "get", get)
    target = tmp_path / "video.part"
    downloads._transfer("https://media.invalid/video.mp4", target, None, lambda *_: None, threading.Event())
    assert target.read_bytes() == payload
    assert [headers["Range"] for headers in requests_seen] == ["bytes=0-3", "bytes=4-7", "bytes=8-11"]
    assert requests_seen[1]["If-Range"] == '"original"'


def test_download_page_limits_transfers_and_releases_next_item(qapp, home, tmp_path, monkeypatch):
    from PyQt6 import QtCore

    from reviewtrans.core.config import SettingsStore
    from reviewtrans.ui.pages import video_downloads
    from reviewtrans.ui.state import AppState

    state = AppState(SettingsStore())
    page = video_downloads.VideoDownloadsPage(state)
    page._folder = tmp_path
    gate = threading.Event()
    monkeypatch.setattr(video_downloads, "download_video", lambda media, *_: (
        gate.wait(3) and tmp_path / Path(media.video).name
    ))
    try:
        assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == ["Browse", "Links", "Download"]
        urls = [f"https://media.invalid/tap{i}.mp4" for i in range(5)]
        page._links.setPlainText("\n".join(urls + [urls[0]]))
        page._submit_links()
        assert len(page._downloads) == 5
        assert len(page._workers) == 3
        assert [job.state for job in page._downloads][3:] == ["pending", "pending"]
        gate.set()
        timer = QtCore.QElapsedTimer()
        timer.start()
        while page.is_busy() and timer.elapsed() < 3000:
            qapp.processEvents()
        assert [job.state for job in page._downloads] == ["done"] * 5
    finally:
        gate.set()
        page.shutdown()
        state.jobs.shutdown()
