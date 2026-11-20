from __future__ import annotations

import threading
import time

from ..config import ProviderProfile, THINKING_OPTIONS
from ..context import UPDATE_SYSTEM, build_update_prompt, context_block, extract_json, glossary_pairs, merge_update
from ..models import Segment, VideoDoc
from ..providers import ProviderError
from ..providers.translate import TranslateJob, Translator, create_translator
from ..resolve import Resolved, resolve
from . import RunContext

CONTEXT_REQUEST_TIMEOUT = 60
CONTEXT_UPDATE_TIMEOUT = 120
CONTEXT_ATTEMPTS = 2
CONTEXT_PROGRESS_INTERVAL = 5
CONTEXT_LOG_INTERVAL = 15


def translate_resolved(ctx: RunContext, doc: VideoDoc | None = None) -> Resolved:
    resolved = resolve(ctx.settings, "translate", ctx.project, doc)
    if resolved.profile is None:
        raise ProviderError("Chưa cấu hình provider dịch (trang Providers).")
    return resolved


def translate_profile(ctx: RunContext, doc: VideoDoc | None = None) -> ProviderProfile:
    return translate_resolved(ctx, doc).profile


def make_translator(ctx: RunContext, doc: VideoDoc | None = None) -> Translator:
    resolved = translate_resolved(ctx, doc)
    return create_translator(resolved.profile, resolved.value)


def translate_segments(
    ctx: RunContext,
    doc: VideoDoc,
    segments: list[Segment],
    targets: list[Segment],
    save=None,
    completed_ids: set[int] | None = None,
) -> int:
    """Dịch danh sách `targets` (thuộc `segments`). Trả về số câu đã dịch."""
    if not targets:
        return 0
    translator = make_translator(ctx, doc)
    context = ctx.store.load_context()
    job = TranslateJob(
        source_lang=doc.source_language or ctx.project.source_language,
        target_lang=ctx.project.target_language,
        context_text=context_block(context, ctx.project.instructions) if translator.uses_context else "",
        glossary=glossary_pairs(context),
        log=ctx.log,
        stop_event=ctx.stop_event,
    )
    batch_size = max(1, min(ctx.settings.translate_batch_size, translator.max_batch))
    index_of = {id(seg): i for i, seg in enumerate(segments)}
    done = 0
    ctx.log(f"Dịch {len(targets)} câu bằng {translator.profile.name} ({translator.model or translator.profile.kind})")
    ctx.log(f"Tối đa {batch_size} câu/lần dịch.")
    if translator.profile.kind == "openai" and translator.thinking_mode():
        ctx.log("Thinking khi dịch phụ đề: " + dict(THINKING_OPTIONS)[translator.thinking_mode()])
    for start in range(0, len(targets), batch_size):
        ctx.check_stop()
        batch = targets[start:start + batch_size]
        first = index_of.get(id(batch[0]), 0)
        job.previous = [
            (s.source, s.text) for s in segments[max(0, first - 6):first] if s.text and s.source
        ]
        job.durations = [s.duration for s in batch]
        results = translator.translate([s.source for s in batch], job)
        for seg, text in zip(batch, results):
            seg.text = text.strip()
        done += len(batch)
        ctx.progress(done * 100.0 / len(targets), f"Đã dịch {done}/{len(targets)} câu")
        if save:
            save()
        if completed_ids is not None:
            completed_ids.update(seg.id for seg in batch)
    return done


def run_translate(
    ctx: RunContext, doc: VideoDoc, segments: list[Segment], only_missing: bool = True, save=None,
    completed_ids: set[int] | None = None,
) -> None:
    targets = [
        s for s in segments
        if s.source.strip() and not s.locked and (not only_missing or not s.text.strip())
        and (completed_ids is None or s.id not in completed_ids)
    ]
    if not targets:
        ctx.log("Không có câu nào cần dịch.")
    translate_segments(ctx, doc, segments, targets, save, completed_ids)


def context_llm(ctx: RunContext, doc: VideoDoc | None = None) -> Translator | None:
    if ctx.project.context_profile:
        profile = ctx.settings.find_translate(ctx.project.context_profile)
        if profile is not None and profile.is_llm:
            return create_translator(profile)
    resolved = translate_resolved(ctx, doc)
    if not resolved.profile.is_llm:
        return None
    return create_translator(resolved.profile, resolved.value)


def update_context(ctx: RunContext, doc: VideoDoc, segments: list[Segment]) -> list[str]:
    ctx.check_stop()
    llm = context_llm(ctx, doc)
    if llm is None:
        ctx.log("Bỏ qua cập nhật ngữ cảnh: cần một provider LLM (OpenAI/Gemini/Anthropic).")
        return []
    ctx.progress(0, "Cập nhật ngữ cảnh project")
    context = ctx.store.load_context()
    prompt = build_update_prompt(
        context, segments, doc.source_language or ctx.project.source_language, ctx.project.target_language, doc.name
    )
    raw = _complete_context(ctx, llm, prompt)
    try:
        data = extract_json(raw)
    except ValueError as exc:
        raise ProviderError(f"Không đọc được JSON ngữ cảnh: {raw[:200]}") from exc
    if not isinstance(data, dict):
        raise ProviderError("LLM trả về ngữ cảnh không đúng định dạng.")
    ctx.check_stop()
    # đọc lại ngay trước khi gộp để giữ các sửa tay mới nhất
    context = ctx.store.load_context()
    changes = merge_update(context, data, doc.name)
    if doc.id not in context.processed_videos:
        context.processed_videos.append(doc.id)
    ctx.store.save_context(context)
    ctx.context_changed()
    ctx.log("Ngữ cảnh: " + ("; ".join(changes) if changes else "không có thay đổi"))
    ctx.progress(100, "Đã cập nhật ngữ cảnh")
    return changes


def _complete_context(ctx: RunContext, llm: Translator, prompt: str) -> str:
    """Chờ API có tiến độ, cho dừng ngay và giới hạn tổng thời gian cập nhật."""
    llm.request_timeout = CONTEXT_REQUEST_TIMEOUT
    llm.purpose = "context"
    request_stop = threading.Event()
    finished = threading.Event()
    response: list[str] = []
    errors: list[Exception] = []
    job = TranslateJob(
        log=lambda message: ctx.log(message) if not request_stop.is_set() else None,
        stop_event=request_stop,
    )

    def request() -> None:
        # Chỉ gọi API ở luồng này. Kết quả về muộn không được tự ghi ngữ cảnh.
        try:
            response.append(llm.complete(UPDATE_SYSTEM, prompt, job=job, attempts=CONTEXT_ATTEMPTS))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            finished.set()

    ctx.log(
        f"Cập nhật ngữ cảnh bằng {llm.profile.name} ({llm.model or llm.profile.kind}): "
        f"timeout {CONTEXT_REQUEST_TIMEOUT}s/lần, tối đa {CONTEXT_ATTEMPTS} lần; "
        f"giới hạn tổng {CONTEXT_UPDATE_TIMEOUT}s."
    )
    if llm.profile.kind == "openai" and llm.thinking_mode():
        ctx.log("Thinking khi cập nhật ngữ cảnh: " + dict(THINKING_OPTIONS)[llm.thinking_mode()])
    started = time.monotonic()
    next_progress = CONTEXT_PROGRESS_INTERVAL
    next_log = CONTEXT_LOG_INTERVAL
    threading.Thread(target=request, name="context-api", daemon=True).start()
    try:
        while not finished.wait(min(0.25, max(0.001, CONTEXT_UPDATE_TIMEOUT))):
            ctx.check_stop()
            elapsed = time.monotonic() - started
            if elapsed >= CONTEXT_UPDATE_TIMEOUT:
                raise ProviderError(f"Cập nhật ngữ cảnh quá {CONTEXT_UPDATE_TIMEOUT}s chờ API.")
            message = f"Đang cập nhật ngữ cảnh: chờ API {elapsed:.0f}s/{CONTEXT_UPDATE_TIMEOUT}s"
            if elapsed >= next_progress:
                ctx.progress(0, message)
                next_progress += CONTEXT_PROGRESS_INTERVAL
            if elapsed >= next_log:
                ctx.log(message)
                next_log += CONTEXT_LOG_INTERVAL
        ctx.check_stop()
        if time.monotonic() - started >= CONTEXT_UPDATE_TIMEOUT:
            raise ProviderError(f"Cập nhật ngữ cảnh quá {CONTEXT_UPDATE_TIMEOUT}s chờ API.")
        if errors:
            raise errors[0]
        return response[0]
    finally:
        request_stop.set()
