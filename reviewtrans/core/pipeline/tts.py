from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import isfinite
from pathlib import Path
from uuid import uuid4

from ..audio_quality import audio_quality_error
from ..config import ProviderProfile
from ..models import ProjectContext, Segment, VideoDoc
from ..proc import StopRequested, call_interruptibly, probe, require_tool, run
from ..providers import ProviderError
from ..providers.tts import TTSProvider, create_tts
from ..resolve import Resolved, resolve
from . import RunContext


def tts_resolved(ctx: RunContext, doc: VideoDoc | None = None) -> Resolved:
    resolved = resolve(ctx.settings, "tts", ctx.project, doc)
    if resolved.profile is None:
        raise ProviderError("Chưa cấu hình provider TTS (trang Providers).")
    return resolved


def tts_profile(ctx: RunContext, doc: VideoDoc | None = None) -> ProviderProfile:
    return tts_resolved(ctx, doc).profile


def resolve_voice(seg: Segment, context: ProjectContext, provider: TTSProvider, default_voice: str = "") -> str:
    """Giọng của câu: giọng riêng của câu > giọng nhân vật > giọng video/project > giọng của profile."""
    if seg.voice:
        return seg.voice
    if seg.speaker:
        for char in context.characters:
            if char.voice and seg.speaker in (char.source, char.target):
                return char.voice
    return default_voice or provider.voice


def synthesis_text(text: str) -> str:
    """Kết câu cho TTS để model đọc trọn âm tiết cuối, giữ nguyên nội dung phụ đề."""
    text = text.strip()
    if text and text[-1].isalnum():
        text += "."
    return text


def segment_key(seg: Segment, profile: ProviderProfile, voice: str, speed: float) -> str:
    parts = [
        synthesis_text(seg.text), profile.id, profile.kind, profile.model, profile.base_url, voice, f"{speed:.2f}",
        str(sorted((k, str(v)) for k, v in profile.options.items() if k not in ("voices",))),
    ]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def tts_path(ctx: RunContext, doc: VideoDoc, seg: Segment) -> Path | None:
    if seg.tts_error or not seg.tts_file or not isfinite(seg.tts_duration) or seg.tts_duration <= 0:
        return None
    path = ctx.store.tts_dir(doc.id) / seg.tts_file
    if not path.exists() or audio_quality_error(path, seg.text, doc.audio.speed, ctx.project.target_language):
        return None
    return path


def needs_tts(ctx: RunContext, doc: VideoDoc, seg: Segment, key: str) -> bool:
    return bool(seg.text.strip()) and (seg.tts_key != key or tts_path(ctx, doc, seg) is None)


def _convert_to_wav(raw: Path, target: Path, ctx: RunContext) -> None:
    ffmpeg = require_tool("ffmpeg")
    code = run(
        [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", "44100",
         "-c:a", "pcm_s16le", str(target)],
        log=ctx.log, stop_event=ctx.stop_event, quiet=True,
    )
    if code != 0 or not target.exists():
        raise ProviderError("Chuyển đổi âm thanh TTS thất bại")


def _remove_cached_file(path: Path, ctx: RunContext) -> None:
    """Dọn cache không được làm hỏng kết quả TTS đã tạo, kể cả khi player giữ file."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        ctx.log(f"  Giữ lại file cache chưa xóa được {path.name}: {exc}")


def run_tts(
    ctx: RunContext,
    doc: VideoDoc,
    segments: list[Segment],
    targets: list[Segment] | None = None,
    force: bool = False,
    save=None,
    completed_ids: set[int] | None = None,
) -> tuple[int, int]:
    """Tạo TTS cho các câu cần. Trả về (thành công, lỗi)."""
    resolved = tts_resolved(ctx, doc)
    profile = resolved.profile
    provider = create_tts(profile)
    context = ctx.store.load_context()
    speed = doc.audio.speed
    tts_dir = ctx.store.tts_dir(doc.id)
    pool = targets if targets is not None else segments
    jobs: list[tuple[Segment, str, str]] = []
    for seg in pool:
        if completed_ids is not None and seg.id in completed_ids:
            continue
        voice = resolve_voice(seg, context, provider, resolved.value)
        key = segment_key(seg, profile, voice, speed)
        if force or needs_tts(ctx, doc, seg, key):
            if seg.text.strip():
                jobs.append((seg, voice, key))
    if not jobs:
        ctx.log("Tất cả câu đã có lồng tiếng mới nhất.")
        return 0, 0
    ctx.log(f"Tạo lồng tiếng cho {len(jobs)} câu bằng {profile.name}")
    lock = threading.Lock()
    done = failed = 0
    errors: list[str] = []

    def work(seg: Segment, voice: str, key: str) -> tuple[Segment, str, float, str]:
        # Mỗi lần tạo dùng file mới để không ghi đè âm thanh player đang đọc.
        suffix = f"{seg.id}_{key[:10]}_{uuid4().hex}"
        base = tts_dir / f"raw_{suffix}"
        target = tts_dir / f"seg_{suffix}.wav"
        last: Exception | None = None
        for attempt in range(3):
            ctx.check_stop()
            text = synthesis_text(seg.text)
            if attempt == 2 and text[-1:] not in ".!?。！？…":
                text += "."
            raw = None
            valid_audio = False
            try:
                raw = call_interruptibly(
                    lambda: provider.synthesize(text, voice, speed, base, ctx.stop_event),
                    ctx.stop_event,
                    discard=lambda path: path.unlink(missing_ok=True),
                )
                ctx.check_stop()
                _convert_to_wav(raw, target, ctx)
                info = probe(target)
                if not info.has_audio or not isfinite(info.duration) or info.duration <= 0:
                    raise ProviderError("TTS trả về âm thanh rỗng hoặc không hợp lệ")
                quality_error = audio_quality_error(target, seg.text, speed, ctx.project.target_language)
                if quality_error:
                    raise ProviderError(quality_error)
                valid_audio = True
                return seg, target.name, info.duration, key
            except StopRequested:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < 2:
                    ctx.log(f"  câu #{seg.id}: {exc} — tạo lại ({attempt + 2}/3)")
            finally:
                if raw is not None and raw != target:
                    _remove_cached_file(raw, ctx)
                if not valid_audio:
                    _remove_cached_file(target, ctx)
        raise ProviderError(f"câu #{seg.id}: {last}")

    concurrency = max(1, int(profile.option("concurrency", ctx.settings.tts_concurrency)))
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(work, *job): job[0] for job in jobs}
        try:
            for future in as_completed(futures):
                try:
                    seg, name, duration, key = future.result()
                except StopRequested:
                    raise
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        failed += 1
                        futures[future].tts_error = str(exc) or "Lỗi tạo giọng"
                        errors.append(str(exc))
                    ctx.progress((done + failed) * 100.0 / len(jobs), f"Lồng tiếng {done + failed}/{len(jobs)}")
                    ctx.log(f"  lỗi TTS {exc}")
                    continue
                with lock:
                    old = seg.tts_file
                    seg.tts_file, seg.tts_duration, seg.tts_key = name, duration, key
                    seg.tts_error = ""
                    if old and old != name:
                        _remove_cached_file(tts_dir / old, ctx)
                    done += 1
                    if completed_ids is not None:
                        completed_ids.add(seg.id)
                    ctx.progress((done + failed) * 100.0 / len(jobs), f"Lồng tiếng {done + failed}/{len(jobs)}")
                    if save and done % 5 == 0:
                        save()
        except StopRequested:
            for future in futures:
                future.cancel()
            raise
        finally:
            if save:
                save()
    if failed:
        ctx.log(f"{failed} câu lỗi TTS: " + "; ".join(errors[:5]))
    return done, failed
