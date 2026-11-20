"""ASR via Paraformer; SRT inputs bypass model inference."""
from pathlib import Path
from ..langs import asr_language_code
from ..models import Segment, VideoDoc
from ..proc import probe, require_tool, run_checked
from ..srt import parse_srt
from . import RunContext
from .funasr import run_paraformer

def ensure_media_info(doc: VideoDoc) -> None:
    if doc.width and doc.height and doc.duration:
        return
    info = probe(doc.source_path)
    doc.duration = info.duration or doc.duration
    doc.width = info.width or doc.width
    doc.height = info.height or doc.height
    doc.fps = info.fps or doc.fps
    doc.has_audio = info.has_audio

def extract_audio(ctx: RunContext, doc: VideoDoc) -> Path:
    target = ctx.store.cache_dir(doc.id) / "audio16k.wav"
    source = Path(doc.source_path)
    if target.exists() and source.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    ffmpeg = require_tool("ffmpeg")
    partial = target.with_suffix(".part.wav")
    ctx.progress(2, "Tách âm thanh")
    run_checked(
        [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(partial)],
        "Tách âm thanh",
        log=ctx.log,
        stop_event=ctx.stop_event,
    )
    partial.replace(target)
    return target



def resolve_asr_engine(ctx: RunContext, doc: VideoDoc) -> str:
    language = asr_language_code(doc.source_language or ctx.project.source_language)
    if language not in ('zh', 'en', 'auto', ''):
        raise ValueError('Paraformer-zh chỉ hỗ trợ tiếng Trung/Anh. Với ngôn ngữ khác, hãy nhập SRT có sẵn.')
    return 'paraformer'


def run_asr(ctx: RunContext, doc: VideoDoc) -> list[Segment]:
    ctx.check_stop()
    if doc.source_mode == 'srt':
        if not doc.srt_path or not Path(doc.srt_path).exists():
            raise FileNotFoundError('Chưa chọn file SRT cho video này.')
        items = parse_srt(Path(doc.srt_path).read_text(encoding='utf-8', errors='replace'))
        return [Segment(id=i + 1, start=s, end=e, source=t) for i, (s,e,t) in enumerate(items)]
    resolve_asr_engine(ctx, doc)
    return run_paraformer(ctx, doc, extract_audio(ctx, doc))
