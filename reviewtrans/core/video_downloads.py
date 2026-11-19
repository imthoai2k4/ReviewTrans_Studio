"""Hongguo web links and local video transfers."""
from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import requests

from .proc import StopRequested, probe, require_tool, run_checked
from .store import slugify

HONGGUO_HOME = "https://hongguoduanju.com/"
HONGGUO_HOSTS = {"hongguoduanju.com", "www.hongguoduanju.com"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v"}
_PLAYER = re.compile(r"/player/(\d{1,25})(?:/\d{1,25})?/?")
_RANGE_SIZE = 1024 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class DownloadError(RuntimeError):
    pass


def http_url(text: str) -> bool:
    try:
        url = urlsplit(text)
        return (
            url.scheme in {"http", "https"} and bool(url.hostname)
            and url.username is None and url.port != 0
            and not any(char.isspace() for char in text)
        )
    except ValueError:
        return False


def is_hongguo(text: str) -> bool:
    return http_url(text) and urlsplit(text).hostname in HONGGUO_HOSTS


def series_id(text: str) -> str | None:
    if not is_hongguo(text):
        return None
    url = urlsplit(text)
    match = _PLAYER.fullmatch(url.path)
    if match:
        return match[1]
    if url.path.rstrip("/") == "/detail":
        identifier = parse_qs(url.query).get("series_id", [""])[0]
        if re.fullmatch(r"[0-9]{1,25}", identifier):
            return identifier
    return None


def player_url(text: str) -> str | None:
    if not is_hongguo(text):
        return None
    url = urlsplit(text)
    if _PLAYER.fullmatch(url.path):
        return HONGGUO_HOME.rstrip("/") + url.path.rstrip("/")
    return None


def detail_url(identifier: str) -> str:
    return f"{HONGGUO_HOME}detail?series_id={identifier}"


@dataclass(frozen=True)
class Episode:
    number: int
    url: str


def episode_links(items: list[list[str]], identifier: str) -> list[Episode]:
    episodes: dict[int, Episode] = {}
    seen = set()
    for label, href in items:
        number = re.fullmatch(r"(?:第\s*)?([0-9]{1,5})(?:\s*集)?", label.strip())
        link = player_url(href)
        if number and link and series_id(link) == identifier and link not in seen:
            index = int(number[1])
            episodes.setdefault(index, Episode(index, link))
            seen.add(link)
    return sorted(episodes.values(), key=lambda episode: episode.number)


@dataclass(frozen=True)
class Media:
    video: str
    audio: str | None = None
    page: str | None = None


def hongguo_media(urls: list[str], page: str) -> Media | None:
    """Select the first video/audio pair requested by this player document."""
    video = audio = combined = None
    for href in urls:
        if not http_url(href):
            continue
        url = urlsplit(href)
        if url.scheme != "https" or url.port not in {None, 443} or not url.hostname.endswith(".qznovelvod.com"):
            continue
        query = parse_qs(url.query)
        if query.get("mime_type") != ["video_mp4"] and not url.path.lower().endswith(".mp4"):
            continue
        markers = set(re.split(r"[^a-z0-9]+", url.path.lower()))
        if markers.intersection({"avc1", "hev1", "hvc1", "av01"}):
            video = video or href
        elif "mp4a" in markers:
            audio = audio or href
        else:
            combined = combined or href
    if video and audio:
        return Media(video, audio, page)
    if combined:
        return Media(combined, page=page)
    return None


def _check_stop(stop) -> None:
    if stop.is_set():
        raise StopRequested()


def _transfer(url: str, path: Path, page: str | None, progress, stop) -> None:
    headers = {"Accept-Encoding": "identity"}
    if page:
        headers["Referer"] = page
    received = total = 0
    reported_at = 0.0
    try:
        with requests.Session() as session, path.open("wb") as handle:
            while True:
                _check_stop(stop)
                headers["Range"] = f"bytes={received}-{received + _RANGE_SIZE - 1}"
                with session.get(url, stream=True, timeout=(6, 8), headers=headers,
                                 allow_redirects=page is None) as response:
                    total, expected_end, partial = _response_extent(response, received, total)
                    if partial and not received:
                        etag = response.headers.get("ETag", "")
                        validator = etag if etag and not etag.startswith("W/") else response.headers.get("Last-Modified")
                        if validator:
                            headers["If-Range"] = validator
                    for chunk in response.iter_content(chunk_size=128 * 1024):
                        _check_stop(stop)
                        handle.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - reported_at >= 0.2:
                            progress(received / total if total else 0, f"{received / 1e6:.1f} MB")
                            reported_at = now
                if not received or (expected_end is not None and received != expected_end):
                    raise DownloadError("Tệp tải về rỗng hoặc chưa đủ dữ liệu.")
                if not partial or received == total:
                    progress(1, f"{received / 1e6:.1f} MB")
                    break
    except requests.RequestException:
        raise DownloadError("Kết nối tải xuống bị gián đoạn hoặc liên kết đã hết hạn.") from None


def _response_extent(response, offset: int, total: int) -> tuple[int, int | None, bool]:
    if response.status_code not in {200, 206}:
        raise DownloadError(f"Máy chủ trả về HTTP {response.status_code}.")
    content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
    if content_type.startswith(("text/", "image/")) or content_type == "application/json":
        raise DownloadError("Liên kết trả về một trang web hoặc dữ liệu khác thay vì tệp video.")
    if response.status_code == 206:
        match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if match and not response.headers.get("Content-Encoding"):
            start, end, size = map(int, match.groups())
            if start == offset and start <= end < size and (not total or size == total):
                return size, end + 1, True
        raise DownloadError("Máy chủ trả về dải dữ liệu không hợp lệ.")
    if offset:
        raise DownloadError("Nguồn video đã thay đổi trong khi tải. Hãy thử lại.")
    length = response.headers.get("Content-Length", "")
    size = int(length) if length.isdecimal() and not response.headers.get("Content-Encoding") else 0
    return size, size or None, False


def _publish(staged: Path, folder: Path, name: str, suffix: str) -> Path:
    stem = slugify(name)[:110].rstrip(" .") or "Video"
    if re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", stem.split(".")[0], re.I):
        stem = f"Video - {stem}"
    for index in count(1):
        basename = stem if index == 1 else f"{stem} ({index})"
        target = folder / f"{basename}{suffix}"
        try:
            with target.open("xb"):
                pass
        except FileExistsError:
            continue
        try:
            return staged.replace(target)
        except OSError:
            target.unlink(missing_ok=True)
            raise


def download_video(media: Media, folder: Path, title: str, progress, stop) -> Path:
    """Download complete files, merge separate tracks and publish a verified video."""
    path = Path(unquote(urlsplit(media.video).path))
    suffix = path.suffix.lower() if path.suffix.lower() in VIDEO_SUFFIXES else ".mp4"
    name = title or path.stem or "Video"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reviewtrans-", dir=folder) as workspace:
        work = Path(workspace)
        video = work / f"video{suffix}"
        video_end = 48 if media.audio else 96
        _transfer(media.video, video, media.page, lambda p, msg: progress(p * video_end, msg), stop)
        output = video
        if media.audio:
            audio = work / "audio.m4a"
            output = work / "merged.mp4"
            _transfer(media.audio, audio, media.page, lambda p, msg: progress(48 + p * 48, msg), stop)
            progress(97, "Đang ghép hình và tiếng…")
            run_checked([
                str(require_tool("ffmpeg")), "-nostdin", "-hide_banner", "-loglevel", "error",
                "-protocol_whitelist", "file", "-f", "mov", "-i", str(video),
                "-protocol_whitelist", "file", "-f", "mov", "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
                "-c", "copy", "-movflags", "+faststart", str(output),
            ], "Ghép hình và tiếng", stop_event=stop)
        info = probe(output, local_formats="mov,matroska,webm,avi,flv,mpegts")
        if not info.has_video or (media.audio and not info.has_audio):
            raise DownloadError("Tệp tải về thiếu luồng hình hoặc tiếng cần thiết.")
        _check_stop(stop)
        return _publish(output, folder, name, output.suffix)
