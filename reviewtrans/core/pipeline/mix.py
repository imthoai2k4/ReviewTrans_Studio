from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..models import AudioSettings, Segment, VideoDoc
from ..proc import atempo_chain, require_tool, run_checked
from . import RunContext

BATCH = 24


@dataclass
class Clip:
    segment: Segment
    file: str
    start: float
    tempo: float
    length: float  # thời lượng sau khi tăng tốc


def plan_clips(doc: VideoDoc, segments: list[Segment]) -> list[Clip]:
    """Tính vị trí & hệ số tăng tốc của từng đoạn lồng tiếng (UI timeline cũng dùng)."""
    ordered = sorted((s for s in segments if not s.tts_error and s.tts_file and s.tts_duration > 0), key=lambda s: s.start)
    all_sorted = sorted(segments, key=lambda s: s.start)
    next_start: dict[int, float] = {}
    for current, following in zip(all_sorted, all_sorted[1:]):
        next_start[id(current)] = following.start
    audio = doc.audio
    clips: list[Clip] = []
    for seg in ordered:
        slot_end = seg.end
        if audio.use_gap:
            slot_end = max(seg.end, next_start.get(id(seg), doc.duration or seg.end))
        start = seg.start
        if audio.fit_mode == "fit" and clips:
            previous = clips[-1]
            start = max(start, previous.start + previous.length)
        slot = max(0.1, slot_end - start)
        tempo = 1.0
        if audio.fit_mode == "fit" and seg.tts_duration > slot:
            tempo = min(max(1.0, audio.max_speed), seg.tts_duration / slot)
        clips.append(Clip(seg, seg.tts_file, start, tempo, seg.tts_duration / tempo))
    return clips


def filter_script_args(work_dir: Path, name: str, graph: str) -> list[str]:
    script = work_dir / name
    script.write_text(graph, encoding="utf-8")
    return ["-/filter_complex", str(script)]


def _merge_intervals(intervals: list[tuple[float, float]], gap: float = 0.35) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def original_volume_filter(audio: AudioSettings, clips: list[Clip]) -> str:
    """Âm lượng cố định hoặc giảm/tăng dần theo thời gian thực của các câu TTS."""
    base = max(0, audio.original_volume) / 100.0
    # Give the low end of the original-audio control a finer, quieter range.
    if base < 1.0:
        base = base ** 2
    if audio.original_mode != "duck" or not clips:
        return f"volume={base:.6f}"
    remaining = max(0, min(100, audio.duck_volume)) / 100.0
    attack = max(0.0, audio.duck_attack)
    release = max(0.0, audio.duck_release)
    envelopes = []
    for start, end in _merge_intervals([(c.start, c.start + c.length) for c in clips]):
        rise = f"clip((t-({start - attack:.6f}))/{attack:.6f},0,1)" if attack else f"gte(t,{start:.6f})"
        fall = f"clip(({end + release:.6f}-t)/{release:.6f},0,1)" if release else f"lte(t,{end:.6f})"
        envelopes.append(f"min({rise},{fall})")
    # Các khoảng giảm giao nhau vẫn giữ đúng mức đã chọn, không nhân giảm nhiều lần.
    envelope = f"min(1,{'+'.join(envelopes)})"
    return f"volume='{base:.6f}*(1-{1 - remaining:.6f}*{envelope})':eval=frame"


def build_dub_track(ctx: RunContext, doc: VideoDoc, clips: list[Clip]) -> Path | None:
    if not clips:
        return None
    ffmpeg = require_tool("ffmpeg")
    cache = ctx.store.cache_dir(doc.id)
    tts_dir = ctx.store.tts_dir(doc.id)
    duration = max(doc.duration, max(c.start + c.length for c in clips) + 0.5)
    signature = hashlib.md5(
        (
            "|".join(f"{c.file}:{c.start:.3f}:{c.tempo:.4f}" for c in clips)
            + f"|{duration:.2f}|{doc.audio.pitch:.2f}"
        ).encode("utf-8")
    ).hexdigest()
    # tên file theo nội dung → preview và job xuất có thể dùng chung, không ghi đè file đang được đọc
    dub = cache / f"dub_{signature[:12]}.wav"
    if dub.exists():
        ctx.log("Dùng lại track lồng tiếng đã trộn.")
        return dub

    work = Path(tempfile.mkdtemp(prefix="mixwork_", dir=cache))
    base = work / "base.wav"
    run_checked(
        [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-t", f"{duration:.3f}",
         "-i", "anullsrc=channel_layout=mono:sample_rate=44100", str(base)],
        "Tạo nền âm thanh", log=ctx.log, stop_event=ctx.stop_event,
    )
    total = len(clips)
    for offset in range(0, total, BATCH):
        ctx.check_stop()
        batch = clips[offset:offset + BATCH]
        command = [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-i", str(base)]
        parts = []
        labels = ["[0:a]"]
        for index, clip in enumerate(batch, start=1):
            command += ["-i", str(tts_dir / clip.file)]
            delay = int(round(clip.start * 1000))
            chain = []
            if abs(clip.tempo - 1.0) > 1e-3:
                chain.append(atempo_chain(clip.tempo))
            chain.append(f"adelay={delay}:all=1")
            parts.append(f"[{index}:a]{','.join(chain)}[a{index}]")
            labels.append(f"[a{index}]")
        parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0[out]")
        out = work / "step.wav"
        command += filter_script_args(work, "dub.txt", ";".join(parts))
        command += ["-map", "[out]", "-ac", "1", "-ar", "44100", str(out)]
        run_checked(command, "Trộn lồng tiếng", log=ctx.log, stop_event=ctx.stop_event)
        out.replace(base)
        ctx.progress(min(total, offset + BATCH) * 70.0 / total, f"Trộn lồng tiếng {min(total, offset + BATCH)}/{total}")

    if abs(doc.audio.pitch) > 1e-3:
        ratio = 2 ** (doc.audio.pitch / 12.0)
        pitched = work / "pitched.wav"
        run_checked(
            [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-i", str(base),
             "-af", f"rubberband=pitch={ratio:.5f}", str(pitched)],
            "Đổi cao độ", log=ctx.log, stop_event=ctx.stop_event,
        )
        pitched.replace(base)
    try:
        base.replace(dub)
    except OSError:
        if not dub.exists():  # lượt trộn khác vừa tạo cùng file thì dùng luôn file đó
            raise
    shutil.rmtree(work, ignore_errors=True)
    for old in cache.glob("dub_*.wav"):
        if old != dub:
            try:
                old.unlink()
            except OSError:
                pass  # đang được đọc ở chỗ khác, lần sau dọn
    return dub


def run_mix(ctx: RunContext, doc: VideoDoc, segments: list[Segment], target: Path | None = None) -> Path:
    """Tạo file trộn = âm gốc (giảm/tắt) + lồng tiếng + nhạc nền. Mặc định cache/mix.wav (dùng khi xuất);
    editor truyền target riêng cho bản nghe thử để không đụng file mà job xuất đang dùng."""
    ffmpeg = require_tool("ffmpeg")
    cache = ctx.store.cache_dir(doc.id)
    audio = doc.audio
    tracks = doc.tracks
    use_original = doc.has_audio and tracks.original_audio and audio.original_mode != "mute"
    if use_original:
        mode = "tự giảm thêm khi có lồng tiếng" if audio.original_mode == "duck" else "giữ cố định suốt video"
        ctx.log(f"Âm lượng video gốc: {audio.original_volume}% — {mode}.")
    else:
        ctx.log("Không dùng âm thanh video gốc trong bản trộn.")
    clips = plan_clips(doc, segments) if tracks.dub else []
    for clip, following in zip(clips, clips[1:]):
        overlap = clip.start + clip.length - following.start
        if overlap > 0.05:
            ctx.log(
                f"CẢNH BÁO: voice câu #{clip.segment.id} chồng câu #{following.segment.id} {overlap:.2f}s "
                f"ở tốc độ x{clip.tempo:.2f}. Hãy dịch gọn hoặc tạo lại voice; giữ đầy đủ tiếng, không cắt đuôi câu."
            )
    duration = max(doc.duration, 0.1, *(c.start + c.length for c in clips))
    if duration > doc.duration + 0.05 and clips:
        ctx.log(f"Lồng tiếng dài hơn video {duration - doc.duration:.2f}s. Khi xuất sẽ giữ khung hình cuối.")
    dub = build_dub_track(ctx, doc, clips) if clips else None
    ctx.progress(75, "Trộn âm thanh cuối")

    command = [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error"]
    parts: list[str] = []
    labels: list[str] = []
    index = 0

    if use_original:
        command += ["-i", doc.source_path]
        parts.append(f"[{index}:a]aresample=48000,{original_volume_filter(audio, clips)}[orig]")
        labels.append("[orig]")
        index += 1
    if dub is not None:
        command += ["-i", str(dub)]
        parts.append(f"[{index}:a]aresample=48000,volume={audio.dub_volume / 100.0:.3f}[dub]")
        labels.append("[dub]")
        index += 1
    if tracks.bgm and audio.bgm_path and Path(audio.bgm_path).exists():
        command += ["-stream_loop", "-1", "-i", audio.bgm_path]
        parts.append(f"[{index}:a]aresample=48000,volume={audio.bgm_volume / 100.0:.3f}[bgm]")
        labels.append("[bgm]")
        index += 1

    target = target or cache / "mix.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not labels:
        command += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        command += ["-ac", "2", str(target)]
    else:
        parts.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0,"
            f"atrim=0:{duration:.3f},asetpts=N/SR/TB[out]"
        )
        command += filter_script_args(target.parent, f"{target.stem}_graph.txt", ";".join(parts))
        command += ["-map", "[out]", "-ac", "2", "-ar", "48000", str(target)]
    run_checked(command, "Trộn âm thanh", log=ctx.log, stop_event=ctx.stop_event)
    ctx.progress(100, "Đã trộn âm thanh")
    return target
